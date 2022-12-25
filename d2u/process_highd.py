#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:10 2022

@author: seuiv
"""

import pandas as pd
import numpy as np
import os
import random

from scipy import spatial


# Parameters
hist_len = 15 # History trajectory length (after sub-sampling)
fut_len = 25 # History trajectory length (after sub-sampling)
total_len = hist_len + fut_len
np.random.seed(42)
random.seed(42)

# Which vehicle should be utilized?
padding = True
# True: Utilize vehicles that are accessible across all history frames. Zero-padding for possible missing frames in future frames.
# False: Utilize vehicles that are accessible across all frames(Both history and future).

# Unidata path
save_path = '/home/lsy/Public/04-Data/UniHighd/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Srouce data path
data_dir = '/mnt/Data/Highd/'
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
    # Highd {0: veh_id, 1: frame_id, 2: x, 3: y, 4: lane}'
    data = pd.DataFrame(data).iloc[:, [0, 1, 2, 3,4]]
    data.columns = ['veh_id', 'frame_id', 'x', 'y', 'lane']
    
    # Index ...
    print('Indexing...')
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
        veh_dict = {}
        # All vehicle info in the currnt frame is stored in veh_dict
        for _, content in vehs.iterrows():
            veh_dict[content['veh_id']] = np.array(content)
        rec_dict[int(frame)] = veh_dict
        i_dicting += 1

    # Processing
    print('File:'+str(i)+' Processing...')
    # We create an unidata for every frame in the scene and distribute it at random among the test, train, and validation sets.
    i_process = 0
    features = []
    for frame in frame_list[:-total_len + 1]:
        i_process += 1
        if i_process % 1000 == 1:
            print('File:', i, 'processing:', i_process / len(frame_list))
        start_frame = int(frame)
        mid_frame = start_frame + hist_len - 1 # Hist and fut frames are separated by the mid_frame.
        end_frame = start_frame + total_len -1
        
        # Accquire vehicles in start and end frame
        if (mid_frame in rec_dict.keys()) and (end_frame in rec_dict.keys()): # If vehicles exist in middle frame still exist in end frame
            vehs_start = list(rec_dict[start_frame].keys())
            vehs_end = list(rec_dict[end_frame].keys())
        else:
            continue
        # Filter vehicle data
        if padding:
            # Utilize vehicles that are accessible across all history frames. Zero-padding for possible missing frames in future frames.
            vehs_filt = vehs_start
            for f in range(start_frame, mid_frame + 1, 1):
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
        jump = False
        for veh in vehs_filt:
            traj = []
            # Extract historical and future trajectories of vehicles.
            for f in range(start_frame, end_frame + 1, 1):
                rec = rec_dict[f]
                if not rec.get(veh) is None: # if vehicle exists
                    traj.append(np.concatenate((rec.get(veh), [1]))) # Flag:1 vehicle exists
                else:
                    traj.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0])) # Flag:0 vehicle absenses
                    # skip several frames
                    if f in [mid_frame + 1,  mid_frame + 2, mid_frame + 3]:
                        jump = True
            if jump:
                continue          
            traj = np.stack(traj)
            features.append(traj)
        if jump:
            continue
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
        
        name_index += 1

np.save('count_highd.npy', count) # Maximum number of vehicles present in the scene
