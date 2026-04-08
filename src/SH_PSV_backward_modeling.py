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

# 1. backward modeling
import time

import cupy as cp  # CuPy is imported as cp for compatibility
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('fast')

from wave_simulation_base import WaveSimulation


class backward_modeling(WaveSimulation):
    """
    kwargs:
    observed data:cp.array 観測データ
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
    receiver_loc:list 受信機の位置 [[i1,j1],[i2,j2],...]

    isnap:int 途中経過の表示ステップ数 default:10
    order:int 空間微分のオーダー(2 or 3) dedault:2
    ###
    snap:cp.array 途中経過の波場スナップショット for correlationg
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obsdata_u = kwargs['observed_data_u'] if 'observed_data_u' in kwargs else None
        self.obsdata_v = kwargs['observed_data_v'] if 'observed_data_v' in kwargs else None
        self.obsdata_w = kwargs['observed_data_w'] if 'observed_data_w' in kwargs else None

        if self.surface_matrix is not None:
            if self.surface_matrix.shape != (self.nx, self.nz):
                raise ValueError('surface_matrix shape must be equal to (nx, nz)')

    def show(self, v: cp.array, suptitle):
        """
        v:cp.array 途中経過の波場スナップショット
        """
        v = v.get()
        mvmax = v.max()
        plt.figure(figsize=(8, 7))
        plt.imshow(v.T, aspect='equal', cmap='seismic', interpolation='nearest', vmin=-mvmax, vmax=mvmax)
        plt.colorbar()
        plt.title(suptitle)
        plt.show()

    def initialize(self):
        self.synsrc_u = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float32)
        self.synsrc_v = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float32)
        self.synsrc_w = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float32)

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

        self.maxAD = 2 ** 23  # ADC full-scale count for raw data scaling

        # Compile fused CUDA kernels for order-2 FDM updates
        self._compile_fdm_kernels()

    def update_vel(self, order):
        if order == 2:
            self._update_vel_order2()
        elif order == 3:
            self._update_vel_order3()
        elif order == 6:
            # Backward time: velocity uses forward spatial stencil (str-type) with sign=-1
            self._launch_str_kernel_order6(-1.0)
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
            # Backward time: stress uses backward spatial stencil (vel-type) with sign=-1
            self._launch_vel_kernel_order6(-1.0)
        else:
            raise ValueError('order must be 2, 3, or 6')
        self.apply_absorb(self.sxx)
        self.apply_absorb(self.sxz)
        self.apply_absorb(self.szz)
        self.apply_absorb(self.syx)
        self.apply_absorb(self.syz)

    def _update_vel_order2(self):
        # Backward: forward-difference stencil, sign = -1
        self._launch_vel_kernel(-1.0, 1, 0, 1, 0)

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
        # Backward: backward-difference stencil, sign = -1
        self._launch_str_kernel(-1.0, 0, -1, 0, -1)

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

    def run(self, show=True):
        """
        run backward modeling
        return:
            0: simulation was conducted safely,
            4: u faced infinite,
            5: v faced infinite,
            6: w faced infinite.

        Show: bool, default=True
            show wavefield animation or not
        """
        print('start backward modeling')
        self.initialize()
        if show:
            self.plot_wavefield()

        # Precompute index arrays and scaling factors for vectorized injection/extraction
        rec_loc_arr = cp.array(self.receiver_loc)
        rec_i = rec_loc_arr[:, 0]
        rec_j = rec_loc_arr[:, 1]
        scale = self.dt * self.dx * self.dz / self.maxAD
        rec_scale_u = scale / self.rho_u[rec_i, rec_j]
        rec_scale_v = scale / self.rho[rec_i, rec_j]
        rec_scale_w = scale / self.rho_w[rec_i, rec_j]
        src_loc_arr = cp.array(self.src_loc)
        src_i = src_loc_arr[:, 0]
        src_j = src_loc_arr[:, 1]

        for it in tqdm(range(self.nt), desc='Backward modeling', unit='step'):
            self.set_boundary_condition()

            t = self.nt - it - 1  # real timestep
            self.update_vel(order=self.order)

            self.u[rec_i, rec_j] += self.obsdata_u[:, t] * rec_scale_u
            self.v[rec_i, rec_j] += self.obsdata_v[:, t] * rec_scale_v
            self.w[rec_i, rec_j] += self.obsdata_w[:, t] * rec_scale_w

            self.update_str(order=self.order)

            # Stay on GPU; single .get() per component after the loop via show_src()
            self.synsrc_u[:, t] = self.u[src_i, src_j]
            self.synsrc_v[:, t] = self.v[src_i, src_j]
            self.synsrc_w[:, t] = self.w[src_i, src_j]

            if it % self.isnap == 0:
                print(f'i={it}/{self.nt}')
                if show:
                    self.display_wavefield()

            if it % 100 == 0:
                if not cp.all(cp.isfinite(self.u)):
                    return 4
                if not cp.all(cp.isfinite(self.v)):
                    return 5
                if not cp.all(cp.isfinite(self.w)):
                    return 6

        print('end backward modeling')
        return 0

    def run_calc(self, import_fwdata_u: cp.array, import_fwdata_v: cp.array, import_fwdata_w: cp.array, isnaps: cp.array, show=True, method='cross_correlation', save=False):
        """
        run backward modeling and calculate correlation with synthetic forward data
        parameters:
            import_fwdata_u:cp.array,
                forward modeling data u
            import_fwdata_v:cp.array,
                forward modeling data v
            import_fwdata_w:cp.array,
                forward modeling data w
            isnaps:cp.array,
                snapshot timesteps of forward modeling u,v,w
            method:str, default='cross_correlation',
                'cross_correlation' or 'convolution' only (20241002)
            show: bool, default=True
                show wavefield animation or not
            save: bool, default=False
                save correlation data or not

        return:
            0: simulation was conducted safely,
            4: u faced infinite,
            5: v faced infinite,
            6: w faced infinite.

        if simulation was conducted safely,
        result_u, result_v, result_w:cp.array
            correlation data of u,v,w with synthetic forward data
        """
        self.initialize()
        if show:
            self.plot_wavefield()

        ## make result image:
        result_u = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        result_v = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        result_w = cp.zeros((self.nx, self.nz), dtype=cp.float64)

        # Precompute index arrays for vectorized injection/extraction
        receiver_loc_array = cp.array(self.receiver_loc)
        rec_i = receiver_loc_array[:, 0]
        rec_j = receiver_loc_array[:, 1]
        src_loc_array = cp.array(self.src_loc)
        src_i = src_loc_array[:, 0]
        src_j = src_loc_array[:, 1]

        # Precompute snapshot lookup on CPU to avoid GPU linear search every step
        isnap_dict = {int(s): idx for idx, s in enumerate(isnaps.get())}

        for it in tqdm(range(self.nt), desc='Backward modeling (calc)', unit='step'):
            # free surface boundary condition Z=0
            self.set_boundary_condition()

            self.update_vel(order=self.order)

            t = (self.nt - 1) - it  # real timestep
            delta_u = self.obsdata_u[:, t]
            delta_v = self.obsdata_v[:, t]
            delta_w = self.obsdata_w[:, t]

            self.u[rec_i, rec_j] += delta_u
            self.v[rec_i, rec_j] += delta_v
            self.w[rec_i, rec_j] += delta_w

            self.update_str(order=self.order)

            self.synsrc_u[:, t] = self.u[src_i, src_j]
            self.synsrc_v[:, t] = self.v[src_i, src_j]
            self.synsrc_w[:, t] = self.w[src_i, src_j]

            if t in isnap_dict:
                indices = isnap_dict[t]
                if method == 'cross_correlation':
                    delta_result_u = (import_fwdata_u[:, :, indices] * self.u[:, :]).squeeze()
                    delta_result_v = (import_fwdata_v[:, :, indices] * self.v[:, :]).squeeze()
                    delta_result_w = (import_fwdata_w[:, :, indices] * self.w[:, :]).squeeze()
                elif method == 'convolution':
                    delta_result_u = (import_fwdata_u[:, :, indices] * self.u[:, :]).squeeze() / (import_fwdata_u[:, :, indices] ** 2).squeeze()
                    delta_result_v = (import_fwdata_v[:, :, indices] * self.v[:, :]).squeeze() / (import_fwdata_v[:, :, indices] ** 2).squeeze()
                    delta_result_w = (import_fwdata_w[:, :, indices] * self.w[:, :]).squeeze() / (import_fwdata_w[:, :, indices] ** 2).squeeze()
                else:
                    raise ValueError('method must be cross_correlation or convolution, now method is ' + str(method))

                result_u += delta_result_u
                result_v += delta_result_v
                result_w += delta_result_w

                if show:
                    sm_u = import_fwdata_u[:, :, indices].squeeze() * self.u[:, :]
                    sm_v = import_fwdata_v[:, :, indices].squeeze() * self.v[:, :]
                    sm_w = import_fwdata_w[:, :, indices].squeeze() * self.w[:, :]
                    self.display_wavefield(u_cpu=sm_u.get(), v_cpu=sm_v.get(), w_cpu=sm_w.get(), suptitle=f'fw x bw at {t}timestep')

            # check if the wavefield is infinite: flag (every 100 steps)
            if it % 100 == 0:
                if not cp.all(cp.isfinite(self.u)):
                    return 4
                if not cp.all(cp.isfinite(self.v)):
                    return 5
                if not cp.all(cp.isfinite(self.w)):
                    return 6

        print('end backward modeling')
        self.result_u = result_u
        self.result_v = result_v
        self.result_w = result_w
        return 0

    def run_gui_calc(self, import_fwdata_u: cp.array, import_fwdata_v: cp.array, import_fwdata_w: cp.array, isnaps: cp.array, method='cross_correlation'):
        """
        Run backward modeling with correlation calculation and Dear PyGui interactive viewer.

        Combines run_gui (interactive GPU viewer) with run_calc (cross-correlation).
        Snapshot steps show the fw×bw correlation image; other steps show the raw wavefield.

        Parameters
        ----------
        import_fwdata_u/v/w : cp.array  (nx, nz, n_snaps)
            Forward modeling wavefield snapshots.
        isnaps : cp.array
            Timestep indices at which snapshots were saved.
        method : str
            'cross_correlation' or 'convolution'.

        Returns
        -------
        0  : simulation completed safely
        4  : u diverged
        5  : v diverged
        6  : w diverged

        On success, sets self.result_u / result_v / result_w.
        """
        from visualizer import SimulationGUI

        cp.get_default_memory_pool().free_all_blocks()
        self.initialize()

        result_u = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        result_v = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        result_w = cp.zeros((self.nx, self.nz), dtype=cp.float64)

        receiver_loc_array = cp.array(self.receiver_loc)
        rec_i = receiver_loc_array[:, 0]
        rec_j = receiver_loc_array[:, 1]
        src_loc_array = cp.array(self.src_loc)
        src_i = src_loc_array[:, 0]
        src_j = src_loc_array[:, 1]

        # Precompute snapshot lookup on CPU to avoid GPU linear search every step
        isnap_dict = {int(s): idx for idx, s in enumerate(isnaps.get())}

        gui = SimulationGUI(self.nx, self.nz, self.dx, self.dz)
        gui.isnap = self.isnap
        gui.setup(self.nt)

        it = 0
        step_time_ms = 0.0
        pbar = tqdm(total=self.nt, desc='Backward modeling (GUI calc)', unit='step')

        while gui.is_running():
            if gui.playing and it < self.nt:
                t0 = time.perf_counter()

                self.set_boundary_condition()
                self.update_vel(order=self.order)

                t = (self.nt - 1) - it  # real (reversed) timestep
                self.u[rec_i, rec_j] += self.obsdata_u[:, t]
                self.v[rec_i, rec_j] += self.obsdata_v[:, t]
                self.w[rec_i, rec_j] += self.obsdata_w[:, t]

                self.update_str(order=self.order)

                self.synsrc_u[:, t] = self.u[src_i, src_j]
                self.synsrc_v[:, t] = self.v[src_i, src_j]
                self.synsrc_w[:, t] = self.w[src_i, src_j]

                # Cross-correlation at snapshot steps
                if t in isnap_dict:
                    idx = isnap_dict[t]
                    fw_u = import_fwdata_u[:, :, idx].squeeze()
                    fw_v = import_fwdata_v[:, :, idx].squeeze()
                    fw_w = import_fwdata_w[:, :, idx].squeeze()
                    if method == 'cross_correlation':
                        delta_u = fw_u * self.u
                        delta_v = fw_v * self.v
                        delta_w = fw_w * self.w
                    elif method == 'convolution':
                        delta_u = (fw_u * self.u) / (fw_u ** 2)
                        delta_v = (fw_v * self.v) / (fw_v ** 2)
                        delta_w = (fw_w * self.w) / (fw_w ** 2)
                    else:
                        gui.cleanup()
                        raise ValueError('method must be cross_correlation or convolution')
                    result_u += delta_u
                    result_v += delta_v
                    result_w += delta_w

                    # Show correlation image at snapshot steps
                    gui.update_textures(delta_u.get(), delta_v.get(), delta_w.get())
                    gui.update_status(it, step_time_ms)
                elif it % gui.isnap == 0:
                    gui.update_textures(cp.asnumpy(self.u), cp.asnumpy(self.v), cp.asnumpy(self.w))
                    gui.update_status(it, step_time_ms)

                step_time_ms = (time.perf_counter() - t0) * 1000.0

                if it % 100 == 0:
                    if not cp.all(cp.isfinite(self.u)):
                        pbar.close()
                        gui.cleanup()
                        return 4
                    if not cp.all(cp.isfinite(self.v)):
                        pbar.close()
                        gui.cleanup()
                        return 5
                    if not cp.all(cp.isfinite(self.w)):
                        pbar.close()
                        gui.cleanup()
                        return 6

                it += 1
                pbar.update(1)
                pbar.set_postfix(ms=f'{step_time_ms:.1f}')
                if it >= self.nt:
                    gui.update_status(it, step_time_ms)

            gui.render_frame()

        pbar.close()
        gui.cleanup()
        print('end backward modeling (GUI calc)')
        self.result_u = result_u
        self.result_v = result_v
        self.result_w = result_w
        return 0

    def show_src(self):
        plt.figure(figsize=(8, 7))
        plt.plot(self.synsrc_u.get()[0, :], label='u')
        plt.plot(self.synsrc_v.get()[0, :], label='v')
        plt.plot(self.synsrc_w.get()[0, :], label='w')
        plt.legend()
        plt.show()

    def run_gui(self):
        """
        Run backward modeling with a Dear PyGui interactive viewer.

        The simulation advances one time-step per rendered GUI frame in reverse
        time order (t = nt-1 → 0). Wavefield textures are updated on the
        GPU→CPU transfer only when ``step % gui.isnap == 0``.

        return:
             0: simulation completed,
             4: u diverged (infinite),
             5: v diverged (infinite),
             6: w diverged (infinite).
        """
        from visualizer import SimulationGUI  # optional DPG dependency

        cp.get_default_memory_pool().free_all_blocks()

        self.initialize()

        # Precompute index arrays and scaling factors (mirrors run())
        rec_loc_arr = cp.array(self.receiver_loc)
        rec_i = rec_loc_arr[:, 0]
        rec_j = rec_loc_arr[:, 1]
        scale = cp.float32(self.dt * self.dx * self.dz / self.maxAD)
        rec_scale_u = (scale / self.rho_u[rec_i, rec_j]).astype(cp.float32)
        rec_scale_v = (scale / self.rho[rec_i, rec_j]).astype(cp.float32)
        rec_scale_w = (scale / self.rho_w[rec_i, rec_j]).astype(cp.float32)
        src_loc_arr = cp.array(self.src_loc)
        src_i = src_loc_arr[:, 0]
        src_j = src_loc_arr[:, 1]

        gui = SimulationGUI(self.nx, self.nz, self.dx, self.dz)
        gui.isnap = self.isnap
        gui.setup(self.nt)

        it = 0
        step_time_ms = 0.0
        pbar = tqdm(total=self.nt, desc='Backward modeling (GUI)', unit='step')

        while gui.is_running():
            if gui.playing and it < self.nt:
                t0 = time.perf_counter()

                self.set_boundary_condition()

                t = self.nt - it - 1  # real (reversed) timestep
                self.update_vel(order=self.order)

                # Observed data injection at receiver locations (time-reversed)
                self.u[rec_i, rec_j] += self.obsdata_u[:, t] * rec_scale_u
                self.v[rec_i, rec_j] += self.obsdata_v[:, t] * rec_scale_v
                self.w[rec_i, rec_j] += self.obsdata_w[:, t] * rec_scale_w

                self.update_str(order=self.order)

                # Synthetic source extraction (stays on GPU)
                self.synsrc_u[:, t] = self.u[src_i, src_j]
                self.synsrc_v[:, t] = self.v[src_i, src_j]
                self.synsrc_w[:, t] = self.w[src_i, src_j]

                step_time_ms = (time.perf_counter() - t0) * 1000.0

                # GPU→CPU transfer only when display update is due
                if it % gui.isnap == 0:
                    u_cpu = cp.asnumpy(self.u)
                    v_cpu = cp.asnumpy(self.v)
                    w_cpu = cp.asnumpy(self.w)
                    gui.update_textures(u_cpu, v_cpu, w_cpu)
                    gui.update_status(it, step_time_ms)

                if it % 100 == 0:
                    if not cp.all(cp.isfinite(self.u)):
                        pbar.close()
                        gui.cleanup()
                        return 4
                    if not cp.all(cp.isfinite(self.v)):
                        pbar.close()
                        gui.cleanup()
                        return 5
                    if not cp.all(cp.isfinite(self.w)):
                        pbar.close()
                        gui.cleanup()
                        return 6

                it += 1
                pbar.update(1)
                pbar.set_postfix(ms=f'{step_time_ms:.1f}')

                if it >= self.nt:
                    gui.update_status(it, step_time_ms)

            gui.render_frame()

        pbar.close()
        gui.cleanup()
        print('end backward modeling (GUI)')
        return 0
