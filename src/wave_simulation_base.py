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

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WaveSimulation:
    """
    Base class for forward and backward elastic wave simulation.
    Holds parameters and methods shared by both forward_modeling and backward_modeling.

    kwargs:
    nx:int              x方向のグリッド数
    nz:int              z方向のグリッド数
    dx:float            x方向のグリッド間隔
    dz:float            z方向のグリッド間隔
    nt:int              シミュレーション時間ステップ数
    fs:float            サンプリング周波数
    vs:cp.array         S波速度
    vp:cp.array         P波速度
    rho:cp.array        密度
    absorbing_frame:int 吸収境界の幅
    src_loc:list        震源の位置 [[i1,j1],[i2,j2],...]
    receiver_loc:list   受信機の位置 [[i1,j1],[i2,j2],...]
    isnap:int           途中経過の表示ステップ数 default:10
    order:int           空間微分のオーダー(2 or 3) default:2
    receivers_height:cp.array  受信機の高さ (optional)
    surface_matrix:cp.array    表面マトリックス (optional)
    steepness_array:cp.array   傾斜配列 (optional)
    """

    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = kwargs['dx']
        self.dz = kwargs['dz']
        self.nt = kwargs['nt']
        self.fs = kwargs['fs']
        self.vs = kwargs['vs'] if 'vs' in kwargs else cp.ones((self.nx, self.nz), dtype=cp.float32) * 200
        self.vp = kwargs['vp'] if 'vp' in kwargs else self.vs * cp.sqrt(6)
        self.rho = kwargs['rho'] if 'rho' in kwargs else cp.ones((self.nx, self.nz), dtype=cp.float32) * 1800
        self.absorbing_frame = kwargs['absorbing_frame'] if 'absorbing_frame' in kwargs else 60
        self.src_loc = kwargs['src_loc'] if 'src_loc' in kwargs else [self.nx // 2, 0]
        self.receiver_loc = kwargs['receiver_loc']
        self.isnap = kwargs['isnap'] if 'isnap' in kwargs else 10
        self.order = kwargs['order'] if 'order' in kwargs else 2
        self.receivers_height = kwargs['receivers_height'] if 'receivers_height' in kwargs else None
        self.surface_matrix = kwargs['surface_matrix'] if 'surface_matrix' in kwargs else None
        self.steepness_array = kwargs['steepness_array'] if 'steepness_array' in kwargs else None

    # ------------------------------------------------------------------
    # Shared elastic parameter helpers
    # ------------------------------------------------------------------

    def shear_avg_SH(self):
        mux = cp.copy(self.mu)
        muz = cp.copy(self.mu)
        mu_i_j = self.mu[1:-1, 1:-1]
        mu_ip1_j = self.mu[2:, 1:-1]
        mu_i_jp1 = self.mu[1:-1, 2:]
        mux[1:-1, 1:-1] = 2 / (1 / mu_i_j + 1 / mu_ip1_j)
        muz[1:-1, 1:-1] = 2 / (1 / mu_i_j + 1 / mu_i_jp1)
        return mux, muz

    def shear_avg_PSV(self):
        muxz = cp.copy(self.mu)
        mu_i_j = self.mu[1:-1, 1:-1]
        mu_ip1_j = self.mu[2:, 1:-1]
        mu_i_jp1 = self.mu[1:-1, 2:]
        mu_ip1_jp1 = self.mu[2:, 2:]
        muxz[1:-1, 1:-1] = 4 / (1 / mu_i_j + 1 / mu_ip1_j + 1 / mu_i_jp1 + 1 / mu_ip1_jp1)
        return muxz

    def rhou(self):
        rho_u = cp.copy(self.rho)
        rho_i_j = self.rho[1:-1, 1:-1]
        rho_ip1_j = self.rho[2:, 1:-1]
        rho_u[1:-1, 1:-1] = 0.5 * (rho_i_j + rho_ip1_j)
        return rho_u

    def rhow(self):
        rho_w = cp.copy(self.rho)
        rho_i_j = self.rho[1:-1, 1:-1]
        rho_i_jp1 = self.rho[1:-1, 2:]
        rho_w[1:-1, 1:-1] = 0.5 * (rho_i_j + rho_i_jp1)
        return rho_w

    def absorb(self):
        """
        Define simple absorbing boundary frame based on wavefield damping
        according to Cerjan et al., 1985, Geophysics, 50, 705-708
        """
        FW = self.absorbing_frame
        a = 0.0053
        nx = self.nx
        nz = self.nz

        # Precompute 1D coefficient vector
        i_range = cp.arange(FW, dtype=cp.float64)
        coeff = cp.exp(-(a ** 2) * (FW - i_range) ** 2)

        absorb_coeff = cp.ones((nx, nz))

        # Left x-boundary: rows 0..FW-1
        # For row i, coeff[i] applies to columns 0..nz-i-2
        row_idx = cp.arange(FW)                          # shape (FW,)
        col_idx = cp.arange(nz)                          # shape (nz,)
        left_mask = col_idx[None, :] < (nz - row_idx[:, None] - 1)  # (FW, nz)
        absorb_coeff[:FW, :] = cp.where(left_mask, coeff[:, None], absorb_coeff[:FW, :])

        # Right x-boundary: rows nx-FW..nx-1 (mirrored)
        right_rows = nx - 1 - row_idx                    # (FW,) descending row indices
        right_mask = col_idx[None, :] < (nz - row_idx[:, None] - 1)
        absorb_coeff[right_rows, :] = cp.where(right_mask, coeff[:, None], absorb_coeff[right_rows, :])

        # Bottom z-boundary: columns nz-FW..nz-1
        # For column j (index nz-j-1), coeff[j] applies to rows j..nx-j-1
        for j in range(FW):
            absorb_coeff[j:nx - j, nz - j - 1] = coeff[j]

        return absorb_coeff

    def set_boundary_condition(self):
        if self.receivers_height is None:
            # free surface boundary condition at Z=0
            self.syz[:, 0] = 0
            self.sxz[:, 0] = 0
            self.szz[:, 0] = 0
        else:
            self.syz = self.syz * self.surface_matrix
            self.sxz = self.sxz * self.surface_matrix
            self.szz = self.szz * self.surface_matrix

    # ------------------------------------------------------------------
    # Shared visualisation helpers
    # ------------------------------------------------------------------

    def plot_wavefield(self):
        u_cpu = np.asarray(self.u.get()).T
        v_cpu = np.asarray(self.v.get()).T
        w_cpu = np.asarray(self.w.get()).T

        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        extent = [0.0, float(self.nx * self.dx), float(self.nz * self.dz), 0.0]

        self.im_u = ax1.imshow(u_cpu, cmap='seismic', extent=extent, animated=True)
        ax1.set_title('U Wavefield')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')

        self.im_v = ax2.imshow(v_cpu, cmap='seismic', extent=extent, animated=True)
        ax2.set_title('V Wavefield')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('z [m]')

        self.im_w = ax3.imshow(w_cpu, cmap='seismic', extent=extent, animated=True)
        ax3.set_title('W Wavefield')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top=0.92, hspace=0.023, wspace=0.12)
        plt.ion()
        plt.show(block=False)

    def display_wavefield(self, u_cpu=None, v_cpu=None, w_cpu=None, suptitle='Wavefield'):
        """
        Display wavefield. Optionally accepts pre-computed CPU arrays;
        falls back to self.u/v/w if not provided.
        """
        plt.suptitle(suptitle)
        u_cpu = self.u.get() if u_cpu is None else u_cpu
        v_cpu = self.v.get() if v_cpu is None else v_cpu
        w_cpu = self.w.get() if w_cpu is None else w_cpu

        self.im_u.set_data(u_cpu.T)
        self.im_v.set_data(v_cpu.T)
        self.im_w.set_data(w_cpu.T)

        u_max = np.abs(u_cpu).max()
        v_max = np.abs(v_cpu).max()
        w_max = np.abs(w_cpu).max()
        self.im_u.set_clim(-u_max, u_max)
        self.im_v.set_clim(-v_max, v_max)
        self.im_w.set_clim(-w_max, w_max)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
