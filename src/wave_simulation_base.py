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


# ---------------------------------------------------------------------------
# CUDA RawKernel sources for fused FDM update (order 2, SH + P-SV)
# ---------------------------------------------------------------------------
_UPDATE_VEL_ORDER2_SRC = r'''
extern "C" __global__
void update_vel_order2(
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    const float* __restrict__ sxx,
    const float* __restrict__ szz,
    const float* __restrict__ sxz,
    const float* __restrict__ syx,
    const float* __restrict__ syz,
    const float* __restrict__ dt_over_rho_u,
    const float* __restrict__ dt_over_rho_w,
    const float* __restrict__ dt_over_rho,
    const float inv_dx, const float inv_dz,
    const float sign,
    const int a_di, const int b_di,
    const int a_dj, const int b_dj,
    const int nx, const int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= nz - 1) return;

    int ij    = i * nz + j;
    int ij_ax = (i + a_di) * nz + j;
    int ij_bx = (i + b_di) * nz + j;
    int ij_az = i * nz + (j + a_dj);
    int ij_bz = i * nz + (j + b_dj);

    /* P-SV: u */
    float sxx_x = (sxx[ij_ax] - sxx[ij_bx]) * inv_dx;
    float sxz_z = (sxz[ij_az] - sxz[ij_bz]) * inv_dz;
    u[ij] += sign * (sxx_x + sxz_z) * dt_over_rho_u[ij];

    /* P-SV: w */
    float sxz_x = (sxz[ij_ax] - sxz[ij_bx]) * inv_dx;
    float szz_z = (szz[ij_az] - szz[ij_bz]) * inv_dz;
    w[ij] += sign * (sxz_x + szz_z) * dt_over_rho_w[ij];

    /* SH: v */
    float syx_x = (syx[ij_ax] - syx[ij_bx]) * inv_dx;
    float syz_z = (syz[ij_az] - syz[ij_bz]) * inv_dz;
    v[ij] += sign * (syx_x + syz_z) * dt_over_rho[ij];
}
'''

_UPDATE_STR_ORDER2_SRC = r'''
extern "C" __global__
void update_str_order2(
    float* __restrict__ sxx,
    float* __restrict__ szz,
    float* __restrict__ sxz,
    float* __restrict__ syx,
    float* __restrict__ syz,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ lam,
    const float* __restrict__ mu,
    const float* __restrict__ mxz,
    const float* __restrict__ myx,
    const float* __restrict__ myz,
    const float dt, const float inv_dx, const float inv_dz,
    const float sign,
    const int a_di, const int b_di,
    const int a_dj, const int b_dj,
    const int nx, const int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= nz - 1) return;

    int ij    = i * nz + j;
    int ij_ax = (i + a_di) * nz + j;
    int ij_bx = (i + b_di) * nz + j;
    int ij_az = i * nz + (j + a_dj);
    int ij_bz = i * nz + (j + b_dj);

    /* P-SV */
    float u_x = (u[ij_ax] - u[ij_bx]) * inv_dx;
    float u_z = (u[ij_az] - u[ij_bz]) * inv_dz;
    float w_x = (w[ij_ax] - w[ij_bx]) * inv_dx;
    float w_z = (w[ij_az] - w[ij_bz]) * inv_dz;

    float div  = u_x + w_z;
    float sdt  = sign * dt;
    float l_ij = lam[ij];
    float m_ij = mu[ij];

    sxx[ij] += sdt * (l_ij * div + 2.0f * m_ij * u_x);
    szz[ij] += sdt * (l_ij * div + 2.0f * m_ij * w_z);
    sxz[ij] += sdt * mxz[ij] * (u_z + w_x);

    /* SH */
    float v_x = (v[ij_ax] - v[ij_bx]) * inv_dx;
    float v_z = (v[ij_az] - v[ij_bz]) * inv_dz;
    syx[ij] += sdt * myx[ij] * v_x;
    syz[ij] += sdt * myz[ij] * v_z;
}
'''


_UPDATE_VEL_L6STENCIL_SRC = r'''
/* L=6 staggered velocity update — backward stencil for stress derivatives
 * df/dx at i   = (c1*(f[i]-f[i-1])   + c2*(f[i+1]-f[i-2]) + c3*(f[i+2]-f[i-3])) / dx
 * Active range: i in [3, nx-3), j in [3, nz-3)
 * c1, c2, c3 are passed as scalars (Taylor: 75/64, -25/384, 3/640)
 */
extern "C" __global__
void update_vel_l6stencil(
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    const float* __restrict__ sxx,
    const float* __restrict__ szz,
    const float* __restrict__ sxz,
    const float* __restrict__ syx,
    const float* __restrict__ syz,
    const float* __restrict__ dt_over_rho_u,
    const float* __restrict__ dt_over_rho_w,
    const float* __restrict__ dt_over_rho,
    const float inv_dx, const float inv_dz,
    const float sign,
    const float c1, const float c2, const float c3,
    const int nx, const int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 3 || i >= nx - 3 || j < 3 || j >= nz - 3) return;

    int ij = i * nz + j;

    /* Backward stencil in x (uses i-3..i+2) */
    #define BWD_X(f) ( c1*((f)[ij]         - (f)[(i-1)*nz+j]) \
                     + c2*((f)[(i+1)*nz+j] - (f)[(i-2)*nz+j]) \
                     + c3*((f)[(i+2)*nz+j] - (f)[(i-3)*nz+j]) ) * inv_dx

    /* Backward stencil in z (uses j-3..j+2) */
    #define BWD_Z(f) ( c1*((f)[ij]          - (f)[i*nz+(j-1)]) \
                     + c2*((f)[i*nz+(j+1)]  - (f)[i*nz+(j-2)]) \
                     + c3*((f)[i*nz+(j+2)]  - (f)[i*nz+(j-3)]) ) * inv_dz

    /* P-SV: u */
    u[ij] += sign * (BWD_X(sxx) + BWD_Z(sxz)) * dt_over_rho_u[ij];

    /* P-SV: w */
    w[ij] += sign * (BWD_X(sxz) + BWD_Z(szz)) * dt_over_rho_w[ij];

    /* SH: v */
    v[ij] += sign * (BWD_X(syx) + BWD_Z(syz)) * dt_over_rho[ij];

    #undef BWD_X
    #undef BWD_Z
}
'''

_UPDATE_STR_L6STENCIL_SRC = r'''
/* L=6 staggered stress update — forward stencil for velocity derivatives
 * df/dx at i+1/2 = (c1*(f[i+1]-f[i])   + c2*(f[i+2]-f[i-1]) + c3*(f[i+3]-f[i-2])) / dx
 * Active range: i in [3, nx-3), j in [3, nz-3)
 * c1, c2, c3 are passed as scalars (Taylor: 75/64, -25/384, 3/640)
 */
extern "C" __global__
void update_str_l6stencil(
    float* __restrict__ sxx,
    float* __restrict__ szz,
    float* __restrict__ sxz,
    float* __restrict__ syx,
    float* __restrict__ syz,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ lam,
    const float* __restrict__ mu,
    const float* __restrict__ mxz,
    const float* __restrict__ myx,
    const float* __restrict__ myz,
    const float dt, const float inv_dx, const float inv_dz,
    const float sign,
    const float c1, const float c2, const float c3,
    const int nx, const int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 3 || i >= nx - 3 || j < 3 || j >= nz - 3) return;

    int ij = i * nz + j;

    /* Forward stencil in x (uses i-2..i+3) */
    #define FWD_X(f) ( c1*((f)[(i+1)*nz+j] - (f)[ij]) \
                     + c2*((f)[(i+2)*nz+j] - (f)[(i-1)*nz+j]) \
                     + c3*((f)[(i+3)*nz+j] - (f)[(i-2)*nz+j]) ) * inv_dx

    /* Forward stencil in z (uses j-2..j+3) */
    #define FWD_Z(f) ( c1*((f)[i*nz+(j+1)] - (f)[ij]) \
                     + c2*((f)[i*nz+(j+2)] - (f)[i*nz+(j-1)]) \
                     + c3*((f)[i*nz+(j+3)] - (f)[i*nz+(j-2)]) ) * inv_dz

    /* P-SV */
    float u_x = FWD_X(u);
    float u_z = FWD_Z(u);
    float w_x = FWD_X(w);
    float w_z = FWD_Z(w);

    float div  = u_x + w_z;
    float sdt  = sign * dt;
    float l_ij = lam[ij];
    float m_ij = mu[ij];

    sxx[ij] += sdt * (l_ij * div + 2.0f * m_ij * u_x);
    szz[ij] += sdt * (l_ij * div + 2.0f * m_ij * w_z);
    sxz[ij] += sdt * mxz[ij] * (u_z + w_x);

    /* SH */
    syx[ij] += sdt * myx[ij] * FWD_X(v);
    syz[ij] += sdt * myz[ij] * FWD_Z(v);

    #undef FWD_X
    #undef FWD_Z
}
'''


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

        self._absorb_FW = FW
        return absorb_coeff

    # ------------------------------------------------------------------
    # Fused CUDA kernel helpers (order 2)
    # ------------------------------------------------------------------

    def _compile_fdm_kernels(self):
        """Compile CUDA kernels and precompute per-cell dt/rho ratios.

        Must be called at the end of subclass initialize() after dt, rho_u,
        rho_w, rho, lam, mu, mxz, myx, myz are all set.

        L=6 stencil coefficients (Taylor 6th-order defaults):
            self._c1 = 75/64,  self._c2 = -25/384,  self._c3 = 3/640
        Override these after calling initialize() to use optimised coefficients
        (e.g. from optimal_FD_operator.optimize_c) before the time loop.
        """
        self._vel_kernel    = cp.RawKernel(_UPDATE_VEL_ORDER2_SRC,   'update_vel_order2')
        self._str_kernel    = cp.RawKernel(_UPDATE_STR_ORDER2_SRC,   'update_str_order2')
        self._vel_kernel_l6 = cp.RawKernel(_UPDATE_VEL_L6STENCIL_SRC, 'update_vel_l6stencil')
        self._str_kernel_l6 = cp.RawKernel(_UPDATE_STR_L6STENCIL_SRC, 'update_str_l6stencil')

        # Taylor 6th-order staggered FD coefficients (c1+3c2+5c3 = 1)
        self._c1 = np.float32(75.0  / 64.0)
        self._c2 = np.float32(-25.0 / 384.0)
        self._c3 = np.float32(3.0   / 640.0)

        # Precompute dt/rho so the kernel avoids per-thread division
        self._dt_over_rho_u = (self.dt / self.rho_u).astype(cp.float32)
        self._dt_over_rho_w = (self.dt / self.rho_w).astype(cp.float32)
        self._dt_over_rho_v = (self.dt / self.rho).astype(cp.float32)

        # Cache scalar args as numpy types (avoid repeated conversion each step)
        self._inv_dx = np.float32(1.0 / float(self.dx))
        self._inv_dz = np.float32(1.0 / float(self.dz))
        self._dt_f32 = np.float32(float(self.dt))
        self._nx_i32 = np.int32(self.nx)
        self._nz_i32 = np.int32(self.nz)

        # Grid / block for order-2 interior points [1, nx-1) x [1, nz-1)
        self._block = (16, 16)
        self._grid = ((self.nx - 2 + 15) // 16, (self.nz - 2 + 15) // 16)
        # Grid for order-6: boundary guard is inside the kernel (i>=3, i<nx-3)
        self._grid_o6 = ((self.nx + 15) // 16, (self.nz + 15) // 16)

    def _launch_vel_kernel(self, sign, a_di, b_di, a_dj, b_dj):
        self._vel_kernel(
            self._grid, self._block,
            (self.u, self.v, self.w,
             self.sxx, self.szz, self.sxz,
             self.syx, self.syz,
             self._dt_over_rho_u, self._dt_over_rho_w, self._dt_over_rho_v,
             self._inv_dx, self._inv_dz,
             np.float32(sign),
             np.int32(a_di), np.int32(b_di),
             np.int32(a_dj), np.int32(b_dj),
             self._nx_i32, self._nz_i32))

    def _launch_str_kernel(self, sign, a_di, b_di, a_dj, b_dj):
        self._str_kernel(
            self._grid, self._block,
            (self.sxx, self.szz, self.sxz,
             self.syx, self.syz,
             self.u, self.v, self.w,
             self.lam, self.mu, self.mxz,
             self.myx, self.myz,
             self._dt_f32, self._inv_dx, self._inv_dz,
             np.float32(sign),
             np.int32(a_di), np.int32(b_di),
             np.int32(a_dj), np.int32(b_dj),
             self._nx_i32, self._nz_i32))

    def _launch_vel_kernel_order6(self, sign):
        self._vel_kernel_l6(
            self._grid_o6, self._block,
            (self.u, self.v, self.w,
             self.sxx, self.szz, self.sxz,
             self.syx, self.syz,
             self._dt_over_rho_u, self._dt_over_rho_w, self._dt_over_rho_v,
             self._inv_dx, self._inv_dz,
             np.float32(sign),
             self._c1, self._c2, self._c3,
             self._nx_i32, self._nz_i32))

    def _launch_str_kernel_order6(self, sign):
        self._str_kernel_l6(
            self._grid_o6, self._block,
            (self.sxx, self.szz, self.sxz,
             self.syx, self.syz,
             self.u, self.v, self.w,
             self.lam, self.mu, self.mxz,
             self.myx, self.myz,
             self._dt_f32, self._inv_dx, self._inv_dz,
             np.float32(sign),
             self._c1, self._c2, self._c3,
             self._nx_i32, self._nz_i32))

    def apply_absorb(self, field):
        """Apply absorbing coefficient only at boundary slices (not the full grid)."""
        FW = self._absorb_FW
        nz = self.nz
        # Left x-boundary
        field[:FW, :] *= self.absorb_coeff[:FW, :]
        # Right x-boundary
        field[-FW:, :] *= self.absorb_coeff[-FW:, :]
        # Bottom z-boundary (middle rows only, left/right already handled)
        field[FW:-FW, -FW:] *= self.absorb_coeff[FW:-FW, -FW:]

    def set_boundary_condition(self):
        if self.receivers_height is None:
            # free surface boundary condition at Z=0
            self.syz[:, 0] = 0
            self.sxz[:, 0] = 0
            self.szz[:, 0] = 0
        else:
            self.syz *= self.surface_matrix
            self.sxz *= self.surface_matrix
            self.szz *= self.surface_matrix

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
