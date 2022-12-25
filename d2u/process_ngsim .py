"""
Created on Tue Jun 14 09:24:10 2022

@author: Qifan Xue
"""

import pandas as pd
import numpy as np
import os
import random
from scipy import spatial

print("Unidata Initializating...")
# Parameters
hist_len = 30 # History trajectory length (before sub-sampling)
fut_len = 50 # Future trajectory length (before sub-sampling)
total_len = hist_len + fut_len
np.random.seed(42)
random.seed(42)

# Which vehicle should be utilized?
padding = True
# True: Utilize vehicles that are accessible across all history frames. Zero-padding for possible missing frames in future frames.
# False: Utilize vehicles that are accessible across all frames(Both history and future).

# Unidata path
save_path = '/home/lsy/Public/04-Data/UniNgsim/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Srouce data path
data_dir = '/mnt/Data/Ngsim/'
txt_list = os.listdir(data_dir)

# Constructing unidata
print("Constructing Unidata...")
name_index = 0
count = np.zeros(500, dtype = np.int) # Maximum number of vehicles present in the scene
for i, name in enumerate(txt_list):
    # Load data ...
    print('File:'+str(i)+' Loading data...')
    data = np.loadtxt(data_dir + name, dtype = np.float64)
    
    # Transfer ...
    # NGSIM {0: veh_id, 3: global_time, 6: global_x, 7: global_y, 10: type, 11: v, 12: a, 13:lane}'
    # type: 1:motor 2:small 3:big
    data = pd.DataFrame(data).iloc[:, [0, 1, 4, 5, 10, 11, 12, 13]]
    data.columns = ['veh_id', 'frame_id', 'x', 'y', 'type', 'v', 'a', 'lane']
    
    # Index ...
    print('File:'+str(i)+' Indexing...')
    frame_feature = data.sort_values(by = 'frame_id')
    frame_list = pd.unique(frame_feature['frame_id'])
    
    # Build dict ...
    print('File:'+str(i)+' Dicting...')
    # All vehicle info is kept in Rec_dict in accordance with the frame.
    rec_dict = {}
    i_dicting = 0
    # Construct Rec_dict
    for frame in frame_list:        
        if i_dicting % 1000 == 1:
            print('File:', i, ' dicting:', i_dicting / len(frame_list))
        # Find all vehicles in the current frame.
        vehs = frame_feature[frame_feature['frame_id'] == frame]
        # All vehicle info in the currnt frame is stored in veh_dict
        veh_dict = {}
        for _, content in vehs.iterrows():
            veh_dict[content['veh_id']] = np.array(content)
        rec_dict[int(frame)] = veh_dict
        i_dicting += 1
        
    # Processing
    print('File:'+str(i)+' Processing...')
    # We create an unidata for every frame in the scene and distribute it at random among the test, train, and validation sets.
    i_process = 0
    for frame in frame_list[:-total_len + 1]:
        if i_process % 1000 == 1:
            print('File:', i, 'processing:', i_process / len(frame_list))
        start_frame = int(frame)
        mid_frame = start_frame + hist_len - 2 # Hist and fut frames are separated by the mid_frame.
        end_frame = start_frame + total_len -2
        
        # Filter vehicle data
        vehs_start = list(rec_dict[start_frame].keys())
        vehs_end = list(rec_dict[end_frame].keys())
        if padding:
            # Utilize vehicles that are accessible across all history frames. Zero-padding for possible missing frames in future frames.
            vehs_filt = vehs_start
            for f in range(start_frame, mid_frame + 1, 2):
                vehs_filt = list(set(vehs_filt) & set(rec_dict[f].keys()))
        else:
            # Utilize vehicles that are accessible across all frames(Both history and future).
            vehs_filt = list(set(vehs_start) & set(vehs_end))
            
        # Not enough vehicles for trajectory prediction
        if len(vehs_filt) < 2:
            continue
        
        # Find the maximum number of vehicles present in the scene.
        count[len(vehs_filt)] = count[len(vehs_filt)] + 1
        
        # Extract trajectory
        features = []
        for veh in vehs_filt:
            traj = []
            # Extract historical and future trajectories of vehicles.
            for f in range(start_frame, end_frame + 1, 2): # Sub-sampling at the same time with rate 2 Hz
                rec = rec_dict[f]
                if not rec.get(veh) is None: # if vehicle exists
                    traj.append(np.concatenate((rec.get(veh), [1]))) # Flag:1 vehicle exists
                else:
                    traj.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0])) # Flag:0 vehicle absenses
            traj = np.stack(traj)
            features.append(traj)
        # Construct unidata
        features = np.array(np.stack(features))
        
        # Distribute unidata at random among the test, train, and validation sets.
        ran = random.random()
        if ran < 0.7:
            path = save_path + 'train/'
        elif ran < 0.9:
            path = save_path + 'val/'
        else:
            path = save_path + 'test/'
        # Save ...
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + 'feature_' + str(name_index) + '.npy', features)
        
        i_process += 1
        name_index += 1

np.save('count_ngsim.npy', count) # Maximum number of vehicles present in the scene
