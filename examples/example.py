import sys
sys.path.append('src/')
import ReverseTimeMigration as RT
import cupy as cp
import numpy as np
import os
import glob


def get_source_ch(distance:np.array, source_x):
    """
    Return the index of the nearest sensor to the source position.
    return: source_ch: list

    ~~example~~
    1D coorinates in sensor line x: 0---1---2---3---4---5---6---7---8---9---10--11--12--13--14--15--16--17--18--19--20--21--[m]
    Positions of each sensors    R:     o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o
    Source position              S:                         ☆

    source_x = 6.0 [m] # source position
    distance = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0] # array of sensor positions
    -> source_ch = [5] (=np.min(np.abs(distance - source_x)))
    """
    nearest_index = np.argmin(np.abs(np.array(distance) - source_x))
    return [nearest_index]

"""
Reverse time migration set-up
parameters:
    observed_u: cupy.array, observed velocity data in x-axis
    observed_v: cupy.array, observed velocity data in y-axis
    observed_w: cupy.array, observed velocity data in z-axis

    source_u: cupy.array, source velocity data in x-axis
    source_v: cupy.array, source velocity data in y-axis
    source_w: cupy.array, source velocity data in z-axis

    receiver_loc: cupy.array, sensor position in x-axis
    source_loc: cupy.array, source position in x-axis

    velocity: float, velocity of the medium
    fs: float, sampling frequency of acquisition data

relationships of u,v,w plane
    u: x-axis
    v: y-axis
    w: z-axis

       Source   |---sensors--|
        o-------#---#---#---#--> x -> wave propagation direction
       /|                       
      / |
     /  |
    /   |
   y    | 
        z 
        (vertical direction)

        

"""
# Here I use npz file.
dir = 'examples/npz data/' # write parent directry here
npzs_path_list = glob.glob(f'{dir}*.npz')

# data setup
sampling_freq = 100000 #[Hz] sampling frequency of acquisition data
time_to = 0.2 # [s] time to observe
velocity = 120 # [m/s] velocity of the medium

"""
for field survey data...
geophone = 28.8V/m/s
logger:+-2.5V, 24bit -> 
"""

dir_rtm= f'examples/results2/'
if not os.path.exists(dir_rtm):
    os.makedirs(dir_rtm)

for npz_path in npzs_path_list:
    npz = np.load(npz_path)
    distance = npz['distance']
    source_x = npz['source_x']
    source_ch = get_source_ch(distance, source_x)  # keep as Python list for numpy indexing
    receiver_loc = cp.array(distance)
    source_loc = cp.array(source_x)
    fs = cp.float32(sampling_freq)

    observed_u = npz['x'][:, :int(fs*time_to)]
    source_u = npz['x'][source_ch][0:int(fs*time_to)]
    observed_v = npz['y'][:, :int(fs*time_to)]
    source_v = npz['y'][source_ch][0:int(fs*time_to)]
    observed_w = npz['z'][:, :int(fs*time_to)]
    source_w = npz['z'][source_ch][0:int(fs*time_to)]

    # set data
    observed_u = cp.array(observed_u)
    source_u = cp.array(source_u)
    observed_v = cp.array(observed_v)
    source_v = cp.array(source_v)
    observed_w = cp.array(observed_w)
    source_w = cp.array(source_w)

    RTM = RT.ReverseTimeMigration(observed_u = observed_u,
                                observed_v = observed_v,
                                observed_w = observed_w,
                                source_u = source_u,
                                source_v = source_v,
                                source_w = source_w,
                                receiver_loc = receiver_loc,
                                source_loc = source_loc,
                                fs = fs,
                                vmin = 80,
                                vmax = 300,
                                vstep = 2000,
                                v_fix = velocity,
                                gui = False,
                                )
    RTM.run(total_memory = 20000, memory_merge = 4000)

    savename = f'{npz_path.split("/")[-1].split(".")[0]}'
    RTM.save_result(dir= dir_rtm +  'data/', savename = savename)