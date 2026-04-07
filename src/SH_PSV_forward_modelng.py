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
import cupy as cp  # CuPy is imported as cp for compatibility
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg') 
import os

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
class forward_modeling:
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
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = kwargs['dx']
        self.dz = kwargs['dz']
        self.nt = kwargs['nt']
        self.fs = kwargs['fs']
        self.vs = kwargs['vs'] if 'vs' in kwargs else cp.ones((self.nx,self.nz), dtype =cp.float32)*200
        self.vp = kwargs['vp'] if 'vp' in kwargs else self.vs*cp.sqrt(6) # at least root2 times larger than vs, poisson ratio = 0.25, vp/vs = 1.7320508, 
        self.rho= kwargs['rho']if 'rho'in kwargs else cp.ones((self.nx,self.nz), dtype =cp.float32)*1800
        self.absorbing_frame = kwargs['absorbing_frame'] if 'absorbing_frame' in kwargs else 60
        self.src_loc = kwargs['src_loc']if 'src_loc'in kwargs else [self.nx // 2,0] #source location, (i,j)
        self.wavelet_u = kwargs['wavelet_u']if 'wavelet_u'in kwargs else None
        self.wavelet_v = kwargs['wavelet_v']if 'wavelet_v'in kwargs else None
        self.wavelet_w = kwargs['wavelet_w']if 'wavelet_w'in kwargs else None
        self.f0 = kwargs['f0'] if 'f0' in kwargs else None
        self.receiver_loc = kwargs['receiver_loc'] #receiver location, (i,j)
        self.isnap = kwargs['isnap']if 'isnap'in kwargs else 10
        self.order = kwargs['order']if 'order'in kwargs else 2
        self.receivers_height = kwargs['receivers_height'] if 'receivers_height' in kwargs else None ##
        self.surface_matrix = kwargs['surface_matrix'] if 'surface_matrix' in kwargs else None
        self.steepness_array = kwargs['steepness_array'] if 'steepness_array' in kwargs else None

    def initialize(self):
        self.seismogram_u = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)
        self.seismogram_v = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)
        self.seismogram_w = cp.zeros((len(self.receiver_loc), self.nt), dtype=cp.float32)

        self.mu = self.rho*self.vs**2
        self.lam = ((self.vp/self.vs)**2 - 2)*self.mu
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

        self.wavelet_u = self.initialize_wavelet(self.wavelet_u)
        self.wavelet_v = self.initialize_wavelet(self.wavelet_v)
        self.wavelet_w = self.initialize_wavelet(self.wavelet_w)
    
    def initialize_wavelet(self, wavelet, show=False):
        if wavelet is None:
            print('wavelet is None. make gaussian source of f0 = ', self.f0)
            wavelets = cp.zeros((len(self.src_loc), self.nt), dtype = cp.float32)
            for i, src in enumerate(self.src_loc):
                wavelets[i,:] = self._gaussian_src(self.f0)
                if show:
                    plt.figure(figsize=(5, 4))
                    plt.plot(wavelets[i,:])
                    plt.title('source wavelet')
                    plt.show()
        else:
            if wavelet.shape[0] != len(self.src_loc):
                raise ValueError('wavelet shape is not match with src_loc')
            if wavelet.shape[1] != self.nt:
                print('wavelet dhape is different to simulation duration. zero padding or cut')
                wavelets = wavelet[:self.nt] if len(wavelet) > self.nt else cp.pad(wavelet, (0,self.nt-len(wavelet)))
            else:
                wavelets = wavelet
        return wavelets
    
    def plot_wavefield(self):
        # 波動場の初期プロットを設定
        u_cpu = np.asarray(self.u.get()).T
        v_cpu = np.asarray(self.v.get()).T
        w_cpu = np.asarray(self.w.get()).T
         
        # 図と軸の設定
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        extent = [0.0, float(self.nx * self.dx), float(self.nz * self.dz), 0.0]

        # 初期イメージの作成
        self.im_u = ax1.imshow(u_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')    
        ax1.set_title('U Wavefield')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')

        self.im_v = ax2.imshow(v_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')
        ax2.set_title('V Wavefield')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('z [m]')

        self.im_w = ax3.imshow(w_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')
        ax3.set_title('W Wavefield')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')

        # source, receiverの位置をプロット
        for l, loc in enumerate(self.receiver_loc):
            ax1.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
            ax2.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
            ax3.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
        for k, loc in enumerate(self.src_loc):
            ax1.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')
            ax2.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')
            ax3.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top = 0.92, hspace= 0.023, wspace= 0.12)
        plt.ion()
        plt.show(block=False)

    def display_wavefield(self):
        # 波動場データをCPUに転送して更新
        u_cpu = self.u.get()
        v_cpu = self.v.get()
        w_cpu = self.w.get()

        # イメージのデータを更新
        self.im_u.set_data(u_cpu.T)
        self.im_v.set_data(v_cpu.T)
        self.im_w.set_data(w_cpu.T)

        # カラーバーの範囲を更新（必要に応じて）
        u_max = cp.max(u_cpu) if cp.max(u_cpu) > -cp.min(u_cpu) else -cp.min(u_cpu)
        v_max = cp.max(v_cpu) if cp.max(v_cpu) > -cp.min(v_cpu) else -cp.min(v_cpu)
        w_max = cp.max(w_cpu) if cp.max(w_cpu) > -cp.min(w_cpu) else -cp.min(w_cpu)
        self.im_u.set_clim(-u_max, u_max)
        self.im_v.set_clim(-v_max, v_max)
        self.im_w.set_clim(-w_max, w_max)

        # プロットを更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(0.001)
    
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
        dsxx = self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * u_x)
        dszz = self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * w_z)
        dsxz = self.dt*(self.mxz[1:-1, 1:-1] * (u_z + w_x))
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

    def shear_avg_SH(self):
        mux = cp.copy(self.mu)
        muz = cp.copy(self.mu)
        # Use vectorized operations
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
        muxz[1:-1,1:-1] = 4 / (1 / mu_i_j + 1 / mu_ip1_j + 1 / mu_i_jp1 + 1 / mu_ip1_jp1) 
        # for i in range(1, self.nx - 1):
        #     for j in range(1, self.nz - 1):
        #         muxz[i, j] = 4/(1/self.mu[i,j] + 1/self.mu[i+1,j] + 1/self.mu[i,j+1] + 1/self.mu[i+1,j+1])
        return muxz
        
    def rhou(self):
        """
        for i in range(1,self.nx-1):
            for j in range(1,self.nz-1):
                self.rho_u[i,j] = 0.5*(self.rho[i,j] + self.rho[i+1,j])        
        """
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
        # for i in range(1,self.nx-1):
        #     for j in range(1,self.nz-1):
        #         self.rho_w[i,j] = 0.5*(self.rho[i,j] + self.rho[i,j+1])
        return rho_w

    def absorb(self):
        """
        Define simple absorbing boundary frame based on wavefield damping
        according to Cerjan et al., 1985, Geophysics, 50, 705-708
        """
        FW = self.absorbing_frame # thickness of absorbing frame (gridpoints)
        a = 0.0053
        nx = self.nx
        nz = self.nz

        coeff = cp.zeros(FW)

        # define coefficients in absorbing frame
        for i in range(FW):
            coeff[i] = cp.exp(-(a**2 * (FW-i)**2))

        # initialize array of absorbing coefficients
        absorb_coeff = cp.ones((nx,nz))

        # compute coefficients for left grid boundaries (x-direction)
        zb=0
        for i in range(FW):
            ze = nz - i - 1
            for j in range(zb,ze):
                absorb_coeff[i,j] = coeff[i]

        # compute coefficients for right grid boundaries (x-direction)
        zb=0
        for i in range(FW):
            ii = nx - i - 1
            ze = nz - i - 1
            for j in range(zb,ze):
                absorb_coeff[ii,j] = coeff[i]

        # compute coefficients for bottom grid boundaries (z-direction)
        xb=0
        for j in range(FW):
            jj = nz - j - 1
            xb = j
            xe = nx - j
            for i in range(xb,xe):
                absorb_coeff[i,jj] = coeff[j]
        return absorb_coeff

    def _gaussian_src(self, f0=10):
        time = cp.linspace(0 * self.dt, self.nt * self.dt, self.nt)
        t0 = 3 / f0
        src = -2. * (time - t0) * (f0 ** 2) * (cp.exp(- (f0 ** 2) * (time - t0) ** 2))
        return src
    
    def set_boundary_conditions(self):
        if self.receivers_height is None:
            # surface: free surface boundary condition Z=0

            self.syz[:, 0] = 0
            self.sxz[:, 0] = 0
            self.szz[:, 0] = 0
        else: # receivers_height is not None
            self.syz =  self.syz * self.surface_matrix
            self.sxz =  self.sxz * self.surface_matrix
            self.szz =  self.szz * self.surface_matrix

    def run(self, show=True, save=False):
        """
        run forward modeling
        retrun:
             0: simulation was cinducted safely,
             1: u faced infinite,
             2: v faced infinite,
             3: w faced infinite.
        show: bool, default True, if True, show wavefield
        save: bool, default False, if True, save wavefield             
        """
        if save: #allocate saved u,v,w, whose sizes are (nx,nz,nt//isnap)
            self.u_save = cp.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=cp.float32)
            self.v_save = cp.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=cp.float32)
            self.w_save = cp.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=cp.float32)
            self.isnaps = cp.zeros(self.nt//self.isnap, dtype=cp.int32)
        
        #print('start forward modeling')
        self.initialize()
        if show:
            self.plot_wavefield()
        for it in range(self.nt):

            self.set_boundary_conditions()
            self.update_vel(order = self.order)
            # add source term at the source location
            for k, loc in enumerate(self.src_loc):
                i, j = loc
                self.u[i, j] += self.wavelet_u[k, it] * self.dt / self.rho_u[i, j] * self.dx * self.dz
                self.v[i, j] += self.wavelet_v[k, it] * self.dt / self.rho[i, j] * self.dx * self.dz
                self.w[i, j] += self.wavelet_w[k, it] * self.dt / self.rho_w[i, j] * self.dx * self.dz
            self.update_str(order = self.order)
            for l, loc in enumerate(self.receiver_loc):
                i, j = loc
                self.seismogram_u[l, it] = self.u[i, j].get()
                self.seismogram_v[l, it] = self.v[i, j].get()  # Transfer data to CPU
                self.seismogram_w[l, it] = self.w[i, j].get()  # Transfer data to CPU
    
            if it % self.isnap == 0:
                if show:
                    self.display_wavefield()
                # self.show(self.v, f'v :{it} step')
            
            if not cp.all(cp.isfinite(self.u)):
                return 1
            if not cp.all(cp.isfinite(self.v)):
                return 2                
            if not cp.all(cp.isfinite(self.w)):
                return 3
            
            # save wavefield if save
            if save and it % self.isnap == 0 and it != 0:
                self.u_save[:,:,it//self.isnap - 1] = self.u
                self.v_save[:,:,it//self.isnap - 1] = self.v
                self.w_save[:,:,it//self.isnap - 1] = self.w
                self.isnaps[it//self.isnap - 1] = it
        
        if show:
            self.show_seismogram(self.seismogram_u.get(), 'u')
            self.show_seismogram(self.seismogram_v.get(), 'v')
            self.show_seismogram(self.seismogram_w.get(), 'w')

        print('end forward modeling')
        return 0

    def show_seismogram(self, seismogram_cpu, title = 'seismogram'):
        fig, axes = plt.subplots(len(seismogram_cpu),1, figsize=(5,4), sharey=False)
        for i in range(len(seismogram_cpu)):
            axes[i].plot(seismogram_cpu[i])
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

            if i == 0:
                axes[i].set_xlabel('time [s]')
            else:
                # axes[i].set_yticklabels([])
                pass
                #axes[i].set_title(label = 'graph',fontdict = {"fontsize":'x-small' })

        plt.suptitle(title)
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top = 0.92, hspace= 0.023, wspace=0)
        #plt.ioff()
        plt.show()
        plt.pause(1.)
