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
import cupy as cp  # CuPy is imported as cp for compatibility
import matplotlib.pyplot as plt
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
        self.synsrc_u = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float64)
        self.synsrc_v = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float64)
        self.synsrc_w = cp.zeros((len(self.src_loc), self.nt), dtype=cp.float64)

        self.mu = self.rho * self.vs ** 2
        self.lam = ((self.vp / self.vs) ** 2 - 2) * self.mu
        self.dt = 1 / self.fs

        # stress
        # for p-sv wave propagation, sxx, sxz, szz
        self.sxx = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.sxz = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.szz = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        # for sh wave propagation, syx, syz
        self.syx = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.syz = cp.zeros((self.nx, self.nz), dtype=cp.float64)

        # velocity
        # u, v, w for each x,y,z,axis
        self.u = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.v = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.w = cp.zeros((self.nx, self.nz), dtype=cp.float64)

        # shear modulus mu
        # mxx,mzz=self.mu for p-sv wave propagation
        self.mxz = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.myx = cp.zeros((self.nx, self.nz), dtype=cp.float64)
        self.myz = cp.zeros((self.nx, self.nz), dtype=cp.float64)

        self.myx, self.myz = self.shear_avg_SH()
        self.mxz = self.shear_avg_PSV()

        # rho for timestep updating
        self.rho_u = self.rhou()
        self.rho_w = self.rhow()

        # Bulk modulus lambda
        # lxx, lzz = self.lam for p-sv wave propagation

        # absorbing coefficient
        self.absorb_coeff = self.absorb()

        ## obsdata scaling
        gain = 16
        ms_V = 28.8
        maxV = 2.5
        self.maxAD = 2 ** 23

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
        sxx_x = (self.sxx[2:, 1:-1] - self.sxx[1:-1, 1:-1]) / self.dx
        szz_z = (self.szz[1:-1, 2:] - self.szz[1:-1, 1:-1]) / self.dz
        sxz_x = (self.sxz[2:, 1:-1] - self.sxz[1:-1, 1:-1]) / self.dx
        sxz_z = (self.sxz[1:-1, 2:] - self.sxz[1:-1, 1:-1]) / self.dz
        du = -(sxx_x + sxz_z) * (self.dt / self.rho_u[1:-1, 1:-1])
        dw = -(sxz_x + szz_z) * (self.dt / self.rho_w[1:-1, 1:-1])
        self.u[1:-1, 1:-1] += du
        self.w[1:-1, 1:-1] += dw
        # SH wave update:
        syx_x = (self.syx[2:, 1:-1] - self.syx[1:-1, 1:-1]) / self.dx
        syz_z = (self.syz[1:-1, 2:] - self.syz[1:-1, 1:-1]) / self.dz
        dv = -(syx_x + syz_z) * (self.dt / self.rho[1:-1, 1:-1])
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
        u_x = (self.u[1:-1, 1:-1] - self.u[0:-2, 1:-1]) / self.dx
        u_z = (self.u[1:-1, 1:-1] - self.u[1:-1, 0:-2]) / self.dz
        w_x = (self.w[1:-1, 1:-1] - self.w[0:-2, 1:-1]) / self.dx
        w_z = (self.w[1:-1, 1:-1] - self.w[1:-1, 0:-2]) / self.dz
        dsxx = -self.dt * (self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0 * self.mu[1:-1, 1:-1] * u_x)
        dszz = -self.dt * (self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0 * self.mu[1:-1, 1:-1] * w_z)
        dsxz = -self.dt * (self.mxz[1:-1, 1:-1] * (u_z + w_x))
        self.sxx[1:-1, 1:-1] += dsxx
        self.szz[1:-1, 1:-1] += dszz
        self.sxz[1:-1, 1:-1] += dsxz

        # SH wave update:
        v_x = (self.v[1:-1, 1:-1] - self.v[0:-2, 1:-1]) / self.dx
        v_z = (self.v[1:-1, 1:-1] - self.v[1:-1, 0:-2]) / self.dz
        dsyx = -self.dt * self.myx[1:-1, 1:-1] * v_x
        dsyz = -self.dt * self.myz[1:-1, 1:-1] * v_z
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
        for it in range(self.nt):
            # free surface boundary condition Z=0
            self.set_boundary_condition()

            t = self.nt - it - 1  # real timestep
            self.update_vel(order=self.order)
            # add source term at the source location
            for k, loc in enumerate(self.receiver_loc):
                i, j = loc
                self.u[i, j] += self.obsdata_u[k, t] * self.dt / self.rho_u[i, j] * self.dx * self.dz / self.maxAD
                self.v[i, j] += self.obsdata_v[k, t] * self.dt / self.rho[i, j] * self.dx * self.dz / self.maxAD
                self.w[i, j] += self.obsdata_w[k, t] * self.dt / self.rho_w[i, j] * self.dx * self.dz / self.maxAD
            self.update_str(order=self.order)
            for l, loc in enumerate(self.src_loc):
                i, j = loc
                self.synsrc_u[l, t] = self.u[i, j].get()
                self.synsrc_v[l, t] = self.v[i, j].get()
                self.synsrc_w[l, t] = self.w[i, j].get()

            if it % self.isnap == 0:
                print(f'i={it}/{self.nt}')
                if show:
                    self.display_wavefield()

            # check if the wavefield is infinite: flag
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

        for it in range(self.nt):
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

            snapt = t
            if snapt in isnaps:
                indices = cp.where(isnaps == snapt)[0][0]
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
                    self.display_wavefield(u_cpu=sm_u.get(), v_cpu=sm_v.get(), w_cpu=sm_w.get(), suptitle=f'fw x bw at {snapt}timestep')

            # check if the wavefield is infinite: flag
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

    def show_src(self):
        plt.figure(figsize=(8, 7))
        plt.plot(self.synsrc_u.get()[0, :], label='u')
        plt.plot(self.synsrc_v.get()[0, :], label='v')
        plt.plot(self.synsrc_w.get()[0, :], label='w')
        plt.legend()
        plt.show()
