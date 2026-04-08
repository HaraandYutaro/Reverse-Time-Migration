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

"""
This file is imaging code of the RTM npz results which are made by ReverseTimeMigration.py.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os

vel = 120

dir = f'examples/results2/data/'
dir_save = f'example/results/RTMimages/'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
rtm_list = glob.glob(f'{dir}*.npz')
cmap = 'gray'
mean = True

for i, rtm in enumerate(rtm_list):
    data = np.load(rtm)
    #image = data['image']
    offset = data['offset']
    dx = data['dx']
    dz = data['dz']
    nx = data['nx']
    nz = data['nz']
    u = data['u']
    v = data['v']
    w = data['w']

    if i == 0:# initialize
        u_sum = np.zeros_like(u)
        v_sum = np.zeros_like(v)
        w_sum = np.zeros_like(w)

    u_sum += u
    v_sum += v
    w_sum += w

# attenuate values at src_loc_step for imaging
# src_loc_step = data['src_loc_step']

# step = 20
# ratio = 1e-2
# for i in range(step):
#     ratio_i = ratio**(i/step)
#     u_sum[:,step - i - 1] *= ratio_i
#     v_sum[:,step - i - 1] *= ratio_i
#     w_sum[:,step - i - 1] *= ratio_i

# usum[:, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-1
# vsum[:, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-1
# wsum[:, src_loc_step[0][1]: src_loc_step[0][1]+10] *= 1e-1

# 軸の範囲を定義
xmin = -offset
zmin = 0
xmax = dx * nx - offset
zmax = dz * nz
xmin = float(xmin)
xmax = float(xmax)
zmin = float(zmin)
zmax = float(zmax)

umap = u_sum.T
vmap = v_sum.T
wmap = w_sum.T
if mean:
    umap = umap - np.mean(umap)
    vmap = vmap - np.mean(vmap)
    wmap = wmap - np.mean(wmap)

umax = np.max(umap) if np.max(umap) > -np.min(umap) else -np.min(umap)
vmax = np.max(vmap) if np.max(vmap) > -np.min(vmap) else -np.min(vmap)
wmax = np.max(wmap) if np.max(wmap) > -np.min(wmap) else -np.min(wmap)

# 図とサブプロットを作成
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 各サブプロットに画像を表示
im1 = axes[0].imshow(umap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = umax, vmin = -umax, aspect='auto')

axes[0].set_title('Reverse Time Migration, totla u')
axes[0].set_xlabel('x')
axes[0].set_ylabel('z')

im2 = axes[1].imshow(vmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = vmax, vmin = -vmax, aspect='auto')
axes[1].set_title('Reverse Time Migration, v')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')

im3 = axes[2].imshow(wmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = wmax, vmin = -wmax, aspect='auto')
axes[2].set_title('Reverse Time Migration, w')
axes[2].set_xlabel('x')
axes[2].set_ylabel('z')

# レイアウトを調整して表示
plt.tight_layout()
plt.savefig(f'{dir_save}{vel}.png')
plt.close()

plt.imshow(umap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = wmax, vmin = -wmax, aspect='auto')
plt.title('Reverse Time Migration, x axis')
plt.xlabel('x')
plt.ylabel('z')
plt.savefig(f'{dir_save}x_{vel}.png')
plt.close()  

plt.imshow(vmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = wmax, vmin = -wmax, aspect='auto')
plt.title('Reverse Time Migration, y axis')
plt.xlabel('x')
plt.ylabel('z')
plt.savefig(f'{dir_save}y_{vel}.png')
plt.close() 

plt.imshow(wmap, cmap=cmap, extent=[xmin, xmax, zmax, zmin], vmax = wmax, vmin = -wmax, aspect='auto')
plt.title('Reverse Time Migration, z axis')
plt.xlabel('x')
plt.ylabel('z')
plt.savefig(f'{dir_save}z_{vel}.png')
plt.close() 

# plt.savefig(dir_res + dir_res + 'stacked RTM image/' + f'{model_name}_rtm_integrated.png')
# np.savez(dir_res + dir_res + 'stacked RTM image/' + f'{model_name}_rtm_integrated.npz',
#         x = umap,
#         y = vmap,
#         z = wmap,
#         xmin = xmin,
#         xmax = xmax,
#         zmin = zmin,
#         zmax = zmax,
#             )
