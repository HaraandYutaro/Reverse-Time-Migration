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

import cupy as cp  # CuPy is imported as cp for compatibility
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

from wave_simulation_base import WaveSimulation

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
class forward_modeling(WaveSimulation):
    """
    kwargs:
    nx:int   x方向のグリッド数
    nz:int   z方向のグリッド数
    dx:float x方向のグリッド間隔
    dz:float z方向のグリッド間隔
    nt:int   シミュレーション時間ステップ数
    fs:float サンプリング周波数
    vs:cp.array S波速度
    rho:cp.array 密度
    absorbing_frame:int 吸収境界の幅
    src_loc:list 震源の位置  [[i1,j1],[i2,j2],...]
    wavelet:cp.array 震源波形
    receiver_loc:list 受信機の位置 [[i1,j1],[i2,j2],...]

    isnap:int 途中経過の表示ステップ数 default:10
    order:int 空間微分のオーダー(2 or 3) dedault:2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wavelet_u = kwargs['wavelet_u'] if 'wavelet_u' in kwargs else None
        self.wavelet_v = kwargs['wavelet_v'] if 'wavelet_v' in kwargs else None
        self.wavelet_w = kwargs['wavelet_w'] if 'wavelet_w' in kwargs else None
        self.f0 = kwargs['f0'] if 'f0' in kwargs else None

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
        else:
            raise ValueError('order must be 2 or 3')
        self.u *= self.absorb_coeff
        self.v *= self.absorb_coeff
        self.w *= self.absorb_coeff

    def update_str(self, order):
        if order == 2:
            self._update_str_order2()
        elif order == 3:
            self._update_str_order3()
        else:
            raise ValueError('order must be 2 or 3')
        self.sxx *= self.absorb_coeff
        self.sxz *= self.absorb_coeff
        self.szz *= self.absorb_coeff
        self.syx *= self.absorb_coeff
        self.syz *= self.absorb_coeff

    def _update_vel_order2(self):
        # P-SV wave update:
        sxx_x = (self.sxx[1:-1, 1:-1] - self.sxx[0:-2, 1:-1]) / self.dx
        szz_z = (self.szz[1:-1, 1:-1] - self.szz[1:-1, 0:-2]) / self.dz
        sxz_x = (self.sxz[1:-1, 1:-1] - self.sxz[0:-2, 1:-1]) / self.dx
        sxz_z = (self.sxz[1:-1, 1:-1] - self.sxz[1:-1, 0:-2]) / self.dz
        du = (sxx_x + sxz_z) * (self.dt / self.rho_u[1:-1, 1:-1])
        dw = (sxz_x + szz_z) * (self.dt / self.rho_w[1:-1, 1:-1])
        self.u[1:-1, 1:-1] += du
        self.w[1:-1, 1:-1] += dw
        # SH wave update:
        syx_x = (self.syx[1:-1, 1:-1] - self.syx[0:-2, 1:-1]) / self.dx
        syz_z = (self.syz[1:-1, 1:-1] - self.syz[1:-1, 0:-2]) / self.dz
        dv = (syx_x + syz_z) * (self.dt / self.rho[1:-1, 1:-1])
        self.v[1:-1, 1:-1] += dv

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
        # P-SV wave update:
        u_x = (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        u_z = (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dz
        w_x = (self.w[2:, 1:-1] - self.w[1:-1, 1:-1]) / self.dx
        w_z = (self.w[1:-1, 2:] - self.w[1:-1, 1:-1]) / self.dz
        dsxx = self.dt * (self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0 * self.mu[1:-1, 1:-1] * u_x)
        dszz = self.dt * (self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0 * self.mu[1:-1, 1:-1] * w_z)
        dsxz = self.dt * (self.mxz[1:-1, 1:-1] * (u_z + w_x))
        self.sxx[1:-1, 1:-1] += dsxx
        self.szz[1:-1, 1:-1] += dszz
        self.sxz[1:-1, 1:-1] += dsxz

        # SH wave update:
        v_x = (self.v[2:, 1:-1] - self.v[1:-1, 1:-1]) / self.dx
        v_z = (self.v[1:-1, 2:] - self.v[1:-1, 1:-1]) / self.dz
        dsyx = self.dt * self.myx[1:-1, 1:-1] * v_x
        dsyz = self.dt * self.myz[1:-1, 1:-1] * v_z
        self.syx[1:-1, 1:-1] += dsyx
        self.syz[1:-1, 1:-1] += dsyz

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

    def run(self, show=True, save=False):
        """
        run forward modeling
        return:
             0: simulation was conducted safely,
             1: u faced infinite,
             2: v faced infinite,
             3: w faced infinite.
        show: bool, default True, if True, show wavefield
        save: bool, default False, if True, save wavefield
        """
        cp.get_default_memory_pool().free_all_blocks()

        if save:  # allocate saved u,v,w, whose sizes are (nx,nz,nt//isnap)
            self.u_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.v_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.w_save = cp.zeros((self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.isnaps = cp.zeros(self.nt // self.isnap, dtype=cp.int32)

        self.initialize()
        if show:
            self.plot_wavefield()
        for it in range(self.nt):

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

            if not cp.all(cp.isfinite(self.u)):
                return 1
            if not cp.all(cp.isfinite(self.v)):
                return 2
            if not cp.all(cp.isfinite(self.w)):
                return 3

            # save wavefield if save
            if save and it % self.isnap == 0 and it != 0:
                self.u_save[:, :, it // self.isnap - 1] = self.u
                self.v_save[:, :, it // self.isnap - 1] = self.v
                self.w_save[:, :, it // self.isnap - 1] = self.w
                self.isnaps[it // self.isnap - 1] = it

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
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.v_save = cp.zeros(
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.w_save = cp.zeros(
                (self.nx, self.nz, self.nt // self.isnap), dtype=cp.float32)
            self.isnaps = cp.zeros(self.nt // self.isnap, dtype=cp.int32)

        self.initialize()

        gui = SimulationGUI(self.nx, self.nz, self.dx, self.dz)
        gui.isnap = self.isnap   # sync initial slider value with class default
        gui.setup(self.nt)

        it = 0
        step_time_ms = 0.0

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
                    self.u_save[:, :, it // self.isnap - 1] = self.u
                    self.v_save[:, :, it // self.isnap - 1] = self.v
                    self.w_save[:, :, it // self.isnap - 1] = self.w
                    self.isnaps[it // self.isnap - 1] = it

                if not cp.all(cp.isfinite(self.u)):
                    gui.cleanup()
                    return 1
                if not cp.all(cp.isfinite(self.v)):
                    gui.cleanup()
                    return 2
                if not cp.all(cp.isfinite(self.w)):
                    gui.cleanup()
                    return 3

                it += 1

                if it >= self.nt:
                    # Simulation finished — push final status and stay open
                    gui.update_status(it, step_time_ms)

            gui.render_frame()

        gui.cleanup()
        print('end forward modeling (GUI)')
        return 0
