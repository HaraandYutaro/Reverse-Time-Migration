# <PROJECT NAME>
# Copyright (C) <2025>  <Yutaro Hara>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

# 1. 順方向モデリング
import time

import numpy as np
import cupy as cp  # CuPy is imported as cp for compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import os

from wave_simulation_base import WaveSimulation
from optimal_FD_operator import optimize_c

"""
    i,j               i+1,j
u,w:o---------o--------o
 rho|sxx               |
mulm|szz               |
    |                  |
  v:o        mxz       o
syz |        sxz       |
myz |                  |
    |                  |
    o------------------o
i,j+1                  i+1,j+1
"""
_DERIVATIVE_STYLE_MAP: dict[str, int] = {
    '2points':   2,   # 2-point staggered (order-2)
    'L6stencil': 6,   # 6th-order staggered (L=6)
}


class forward_modeling(WaveSimulation):
    """
    Parameters
    ----------
    nx : int
        x方向のグリッド数
    nz : int
        z方向のグリッド数
    dx : float
        x方向のグリッド間隔
    dz : float
        z方向のグリッド間隔
    nt : int
        シミュレーション時間ステップ数
    fs : float
        サンプリング周波数
    receiver_loc : list
        受信機の位置 [[i1,j1],[i2,j2],...]
    src_loc : list, optional
        震源の位置 [[i1,j1],[i2,j2],...]  default: [[nx//2, 0]]
    absorbing_frame : int, optional
        吸収境界の幅  default: 60
    isnap : int, optional
        途中経過の表示ステップ数  default: 10
    derivative_style : str, optional
        空間微分スキームの選択  '2points' (default) or 'L6stencil'
        '2points'   — 2点スタガード差分 (order-2)
        'L6stencil' — 6次精度スタガード差分 (L=6)
    fmax : float, optional
        最大周波数 [Hz]  default: 50.0
        'L6stencil' 選択時に λmin = vs_min / fmax を計算し、
        optimal_FD_operator.optimize_c() で分散誤差を最小化した係数を
        自動設定する。fd_coefficients を明示した場合はそちらが優先される。
    **kwargs
        追加パラメータ (vs, vp, rho, wavelet_u, wavelet_v, wavelet_w,
        f0, receivers_height, surface_matrix, steepness_array 等)
        はすべてキーワード引数として渡す。
    """

    def __init__(
        self,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        nt: int,
        fs: float,
        receiver_loc: list,
        src_loc: list | None = None,
        absorbing_frame: int = 60,
        isnap: int = 10,
        derivative_style: str = '2points',
        fmax: float = 50.0,
        **kwargs,
    ) -> None:
        if derivative_style not in _DERIVATIVE_STYLE_MAP:
            raise ValueError(
                f"derivative_style must be one of {list(_DERIVATIVE_STYLE_MAP)}, "
                f"got {derivative_style!r}"
            )
        kwargs.pop('order', None)  # derivative_style takes precedence over order
        super_kwargs: dict = dict(
            nx=nx, nz=nz, dx=dx, dz=dz, nt=nt, fs=fs,
            receiver_loc=receiver_loc,
            absorbing_frame=absorbing_frame,
            isnap=isnap,
            order=_DERIVATIVE_STYLE_MAP[derivative_style],
            **kwargs,
        )
        if src_loc is not None:
            super_kwargs['src_loc'] = src_loc
        super().__init__(**super_kwargs)
        self.derivative_style: str = derivative_style
        self.fmax: float = fmax
        self.wavelet_u = kwargs.get('wavelet_u')
        self.wavelet_v = kwargs.get('wavelet_v')
        self.wavelet_w = kwargs.get('wavelet_w')
        self.f0 = kwargs.get('f0')

    def initialize(self):
        self.seismogram_u = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)
        self.seismogram_v = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)
        self.seismogram_w = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)

        self.mu = self.rho * self.vs ** 2
        self.lam = ((self.vp / self.vs) ** 2 - 2) * self.mu
        self.dt = 1 / self.fs

        # stress
        # for p-sv wave propagation, sxx, sxz, szz
        self.sxx = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.sxz = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.szz = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        # for sh wave propagation, syx, syz
        self.syx = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.syz = cp.zeros((self.nx, self.nz), dtype=cp.float32)

        # velocity
        # u, v, w for each x,y,z,axis
        self.u = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.v = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.w = cp.zeros((self.nx, self.nz), dtype=cp.float32)

        # shear modulus mu
        # mxx,mzz=self.mu for p-sv wave propagation
        self.mxz = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.myx = cp.zeros((self.nx, self.nz), dtype=cp.float32)
        self.myz = cp.zeros((self.nx, self.nz), dtype=cp.float32)

        self.myx, self.myz = self.shear_avg_SH()
        self.mxz = self.shear_avg_PSV()

        # rho for timestep updating
        self.rho_u = self.rhou()
        self.rho_w = self.rhow()

        # Bulk modulus lambda
        # lxx, lzz = self.lam for p-sv wave propagation

        # absorbing coefficient
        self.absorb_coeff = self.absorb()

        # Precompute index arrays and scaling factors for source injection and
        # seismogram extraction — avoids Python loops inside the time loop
        src_loc_arr = cp.array(self.src_loc)
        self._src_i = src_loc_arr[:, 0]
        self._src_j = src_loc_arr[:, 1]
        rec_loc_arr = cp.array(self.receiver_loc)
        self._rec_i = rec_loc_arr[:, 0]
        self._rec_j = rec_loc_arr[:, 1]
        scale = self.dt * self.dx * self.dz
        self._src_scale_u = scale / self.rho_u[self._src_i, self._src_j]
        self._src_scale_v = scale / self.rho[self._src_i, self._src_j]
        self._src_scale_w = scale / self.rho_w[self._src_i, self._src_j]

        self.wavelet_u = self.initialize_wavelet(self.wavelet_u)
        self.wavelet_v = self.initialize_wavelet(self.wavelet_v)
        self.wavelet_w = self.initialize_wavelet(self.wavelet_w)

        # Compile fused CUDA kernels for order-2 FDM updates
        self._compile_fdm_kernels()

    def initialize_wavelet(self, wavelet, show=False):
        if wavelet is None:
            print('wavelet is None. make gaussian source of f0 = ', self.f0)
            wavelets = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float32)
            for i, src in enumerate(self.src_loc):
                wavelets[i, :] = self._gaussian_src(self.f0)
                if show:
                    plt.figure(figsize=(5, 4))
                    plt.plot(wavelets[i, :])
                    plt.title('source wavelet')
                    plt.show()
        else:
            if wavelet.shape[0] != len(self.src_loc):
                raise ValueError('wavelet shape is not match with src_loc')
            if wavelet.shape[1] != self.nt:
                print('wavelet shape is different to simulation duration. zero padding or cut')
                wavelets = wavelet[:self.nt] if len(wavelet) > self.nt else cp.pad(wavelet, (0, self.nt - len(wavelet)))
            else:
                wavelets = wavelet
        return wavelets

    def plot_wavefield(self):
        # Call base implementation then overlay source/receiver positions
        super().plot_wavefield()
        ax1, ax2, ax3 = self.fig.axes
        for loc in self.receiver_loc:
            ax1.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='+', color='y')
            ax2.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='+', color='y')
            ax3.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='+', color='y')
        for loc in self.src_loc:
            ax1.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='*', color='g')
            ax2.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='*', color='g')
            ax3.scatter(float(loc[0] * self.dx), float(loc[1] * self.dz), marker='*', color='g')
        # Forward modeling uses equal aspect ratio
        for ax in (ax1, ax2, ax3):
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.92, hspace=0.023, wspace=0.12)

    def display_wavefield(self):
        # Forward modeling's simpler no-argument version delegates to base
        super().display_wavefield()

    def update_vel(self, order):
        if order == 2:
            self._update_vel_order2()
        elif order == 3:
            self._update_vel_order3()
        elif order == 6:
            self._launch_vel_kernel_order6(1.0)
        else:
            raise ValueError('order must be 2, 3, or 6')
        self.apply_absorb(self.u)
        self.apply_absorb(self.v)
        self.apply_absorb(self.w)

    def update_str(self, order):
        if order == 2:
            self._update_str_order2()
        elif order == 3:
            self._update_str_order3()
        elif order == 6:
            self._launch_str_kernel_order6(1.0)
        else:
            raise ValueError('order must be 2, 3, or 6')
        self.apply_absorb(self.sxx)
        self.apply_absorb(self.sxz)
        self.apply_absorb(self.szz)
        self.apply_absorb(self.syx)
        self.apply_absorb(self.syz)

    def _update_vel_order2(self):
        # Forward: backward-difference stencil, sign = +1
        self._launch_vel_kernel(1.0, 0, -1, 0, -1)

    def _update_vel_order3(self):
        # インデックス範囲を設定
        i_start = 3
        i_end = self.v.shape[0] - 3  # nx - 3
        j_start = 3
        j_end = self.v.shape[1] - 3  # nz - 3

        # syx_x の計算
        syx_x = (
            (1/3)  * self.sx[i_start+1:i_end+1, j_start:j_end]
          - (2/3)  * self.sx[i_start:i_end, j_start:j_end]
          +  3     * self.sx[i_start-1:i_end-1, j_start:j_end]
          - (11/6) * self.sx[i_start-2:i_end-2, j_start:j_end]
        ) / self.dx

        # syz_z の計算
        syz_z = (
            (1/3) * self.sz[i_start:i_end, j_start+1:j_end+1]
            - (2/3) * self.sz[i_start:i_end, j_start:j_end]
            + 3 * self.sz[i_start:i_end, j_start-1:j_end-1]
            - (11/6) * self.sz[i_start:i_end, j_start-2:j_end-2]
        ) / self.dz

        # 速度場の更新
        dv = (syx_x + syz_z) * (self.dt / self.rho[i_start:i_end, j_start:j_end])
        self.v[i_start:i_end, j_start:j_end] += dv

    def _update_str_order2(self):
        # Forward: forward-difference stencil, sign = +1
        self._launch_str_kernel(1.0, 1, 0, 1, 0)

    def _update_str_order3(self):
        # インデックス範囲を設定
        i_start = 3
        i_end = self.v.shape[0] - 3  # nx - 3
        j_start = 3
        j_end = self.v.shape[1] - 3  # nz - 3
        # v_x の計算
        v_x = (
            (1/3)   * self.v[i_start+1:i_end+1, j_start:j_end]
            - (2/3) * self.v[i_start:i_end, j_start:j_end]
            + 3     * self.v[i_start-1:i_end-1, j_start:j_end]
            - (11/6) * self.v[i_start-2:i_end-2, j_start:j_end]
        ) / self.dx
        # v_z の計算
        v_z = (
            (1/3) * self.v[i_start:i_end, j_start+1:j_end+1]
            - (2/3) * self.v[i_start:i_end, j_start:j_end]
            + 3 * self.v[i_start:i_end, j_start-1:j_end-1]
            - (11/6) * self.v[i_start:i_end, j_start-2:j_end-2]
        ) / self.dz

        # 応力場の更新
        self.sx[i_start:i_end, j_start:j_end] += self.dt * self.mux[i_start:i_end, j_start:j_end] * v_x
        self.sz[i_start:i_end, j_start:j_end] += self.dt * self.muz[i_start:i_end, j_start:j_end] * v_z

    def _gaussian_src(self, f0=10):
        time = cp.linspace(0 * self.dt, self.nt * self.dt, self.nt)
        t0 = 3 / f0
        src = -2. * (time - t0) * (f0 ** 2) * (cp.exp(-(f0 ** 2) * (time - t0) ** 2))
        return src

    def run(
        self,
        show: bool = True,
        save: bool = False,
        fd_coefficients: tuple[float, float, float] | None = None,
    ) -> int:
        """
        run forward modeling

        Parameters
        ----------
        show : bool
            True の場合、wavefield をリアルタイム表示する。
        save : bool
            True の場合、wavefield スナップショットを float16 で保存する。
        fd_coefficients : tuple[float, float, float] | None
            L=6 stencil の係数 (c1, c2, c3) を上書きする場合に指定。
            None の場合は Taylor 6次係数 (75/64, -25/384, 3/640) を使用。
            optimal_FD_operator.optimize_c() で得た係数をここに渡せる。

        Returns
        -------
        int
            0: 正常終了, 1: u が発散, 2: v が発散, 3: w が発散
        """
        cp.get_default_memory_pool().free_all_blocks()

        if save:  # allocate saved u,v,w as float16 to halve VRAM usage
            self.u_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.v_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.w_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.isnaps = cp.zeros(self.nt // self.isnap, dtype=cp.int32)

        # L6stencil: update grid spacing to λmin/8 before initialize() so that
        # _compile_fdm_kernels() picks up the correct inv_dx / inv_dz
        _lambda_min: float | None = None
        if self.derivative_style == 'L6stencil':
            vmin = float(self.vs.min())
            _lambda_min = vmin / self.fmax
            self.dx = _lambda_min / 8.0
            self.dz = _lambda_min / 8.0

        self.initialize()

        if fd_coefficients is not None:
            self._c1, self._c2, self._c3 = (np.float32(c) for c in fd_coefficients)
        elif _lambda_min is not None:
            _, res, _ = optimize_c(_lambda_min, self.dx)
            self._c1, self._c2, self._c3 = (np.float32(c) for c in res.x)
            print(f'L6stencil: λmin={_lambda_min:.1f}m, dx={self.dx:.2f}m, '
                  f'c=({float(self._c1):.6f}, {float(self._c2):.6f}, {float(self._c3):.6f})')
        if show:
            self.plot_wavefield()
        for it in tqdm(range(self.nt), desc='Forward modeling', unit='step'):

            self.set_boundary_condition()
            self.update_vel(order=self.order)

            # Source injection (vectorized — no Python loop over sources)
            self.u[self._src_i, self._src_j] += self.wavelet_u[:, it] * self._src_scale_u
            self.v[self._src_i, self._src_j] += self.wavelet_v[:, it] * self._src_scale_v
            self.w[self._src_i, self._src_j] += self.wavelet_w[:, it] * self._src_scale_w

            self.update_str(order=self.order)

            # Seismogram extraction (vectorized, stays on GPU until after loop)
            self.seismogram_u[:, it] = self.u[self._rec_i, self._rec_j]
            self.seismogram_v[:, it] = self.v[self._rec_i, self._rec_j]
            self.seismogram_w[:, it] = self.w[self._rec_i, self._rec_j]

            if it % self.isnap == 0:
                if show:
                    self.display_wavefield()

            if it % 100 == 0:
                if not cp.all(cp.isfinite(self.u)):
                    return 1
                if not cp.all(cp.isfinite(self.v)):
                    return 2
                if not cp.all(cp.isfinite(self.w)):
                    return 3

            # save wavefield snapshot as float16
            if save and it % self.isnap == 0 and it != 0:
                idx = it // self.isnap - 1
                self.u_save[:, :, idx] = self.u.astype(cp.float16)
                self.v_save[:, :, idx] = self.v.astype(cp.float16)
                self.w_save[:, :, idx] = self.w.astype(cp.float16)
                self.isnaps[idx] = it

        if show:
            self.show_seismogram(self.seismogram_u.get(), 'u')
            self.show_seismogram(self.seismogram_v.get(), 'v')
            self.show_seismogram(self.seismogram_w.get(), 'w')

        print('end forward modeling')
        return 0

    def show_seismogram(self, seismogram_cpu, title='seismogram'):
        fig, axes = plt.subplots(len(seismogram_cpu), 1, figsize=(5, 4), sharey=False)
        for i in range(len(seismogram_cpu)):
            axes[i].plot(seismogram_cpu[i])
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

            if i == 0:
                axes[i].set_xlabel('time [s]')

        plt.suptitle(title)
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.92, hspace=0.023, wspace=0)
        plt.show()
        plt.pause(1.)

    def run_gui(self, save=False):
        """
        Run forward modeling with a Dear PyGui interactive viewer.

        The simulation advances one time-step per rendered GUI frame.
        Wavefield textures are updated on the GPU->CPU transfer only when
        ``step % gui.isnap == 0`` (controlled live by the isnap slider).

        return:
             0: simulation completed,
             1: u diverged (infinite),
             2: v diverged (infinite),
             3: w diverged (infinite).
        save: bool, default False – if True, snapshot arrays are stored in
              self.u_save / v_save / w_save (float32, shape nx×nz×(nt//isnap)).
        """
        from visualizer import SimulationGUI  # optional DPG dependency

        # Free VRAM before large allocations (G14 memory constraint)
        cp.get_default_memory_pool().free_all_blocks()

        if save:
            self.u_save = cp.zeros(
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.v_save = cp.zeros(
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.w_save = cp.zeros(
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float16)
            self.isnaps = cp.zeros(self.nt // self.isnap, dtype=cp.int32)

        self.initialize()

        gui = SimulationGUI(self.nx, self.nz, self.dx, self.dz)
        gui.isnap = self.isnap   # sync initial slider value with class default
        gui.setup(self.nt)

        it = 0
        step_time_ms = 0.0
        pbar = tqdm(total=self.nt, desc='Forward modeling (GUI)', unit='step')

        while gui.is_running():
            if gui.playing and it < self.nt:
                t0 = time.perf_counter()

                self.set_boundary_condition()
                self.update_vel(order=self.order)

                # Source injection (vectorized)
                self.u[self._src_i, self._src_j] += (
                    self.wavelet_u[:, it] * self._src_scale_u)
                self.v[self._src_i, self._src_j] += (
                    self.wavelet_v[:, it] * self._src_scale_v)
                self.w[self._src_i, self._src_j] += (
                    self.wavelet_w[:, it] * self._src_scale_w)

                self.update_str(order=self.order)

                # Seismogram extraction (stays on GPU)
                self.seismogram_u[:, it] = self.u[self._rec_i, self._rec_j]
                self.seismogram_v[:, it] = self.v[self._rec_i, self._rec_j]
                self.seismogram_w[:, it] = self.w[self._rec_i, self._rec_j]

                step_time_ms = (time.perf_counter() - t0) * 1000.0

                # GPU→CPU transfer only when display update is due
                if it % gui.isnap == 0:
                    u_cpu = cp.asnumpy(self.u)
                    v_cpu = cp.asnumpy(self.v)
                    w_cpu = cp.asnumpy(self.w)
                    gui.update_textures(u_cpu, v_cpu, w_cpu)
                    gui.update_status(it, step_time_ms)

                if save and it % self.isnap == 0 and it != 0:
                    idx = it // self.isnap - 1
                    self.u_save[:, :, idx] = self.u.astype(cp.float16)
                    self.v_save[:, :, idx] = self.v.astype(cp.float16)
                    self.w_save[:, :, idx] = self.w.astype(cp.float16)
                    self.isnaps[idx] = it

                if it % 100 == 0:
                    if not cp.all(cp.isfinite(self.u)):
                        pbar.close()
                        gui.cleanup()
                        return 1
                    if not cp.all(cp.isfinite(self.v)):
                        pbar.close()
                        gui.cleanup()
                        return 2
                    if not cp.all(cp.isfinite(self.w)):
                        pbar.close()
                        gui.cleanup()
                        return 3

                it += 1
                pbar.update(1)
                pbar.set_postfix(ms=f'{step_time_ms:.1f}')

                if it >= self.nt:
                    # Simulation finished — push final status and stay open
                    gui.update_status(it, step_time_ms)

            gui.render_frame()

        pbar.close()
        gui.cleanup()
        print('end forward modeling (GUI)')
        return 0
