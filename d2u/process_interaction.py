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
  
# Source data path  
data_dir = '/mnt/Data/Interaction/'
folder_list = os.listdir(data_dir)

name_index = 0
print("Constructing Unidata...")
counter = np.zeros(500, dtype = np.int) # Maximum number of vehicles present in the scene
for i, folder in enumerate(folder_list):
    # Constructing source data path
    csv_list = os.listdir(data_dir + folder + '/')
    
    # Unidata path
    save_path = './home/lsy/Public/04-Data/UniInter/' + folder + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Constructing unidata
    for j, name in enumerate(csv_list):
        # 0: id 1: frame 3:type 4: x 5: y 6: vx 7:vy
        # Load data ...
        data = pd.read_csv(data_dir + folder + '/' + name, usecols = [0, 1, 3, 4, 5, 6, 7])
    
        # Index ...
        frame_feature = data.sort_values(by = 'frame_id')
        frame_list = data.loc[:, 'frame_id'].unique()
    
        # Build dict ...
        # All vehicle info is kept in Rec_dict in accordance with the frame.
        rec_dict = {}
        i_dicting = 0
        # Construct Rec_dict
        for index, frame in enumerate(frame_list):
            if index % 1000 == 1:
                print('folder: ', i, ' csv: ', j, ' dict_process:', i_dicting / len(frame_list))
            # Find all vehicles in the current frame.
            vehs = frame_feature[frame_feature['frame_id'] == frame]
            # All vehicle info in the currnt frame is stored in veh_dict
            veh_dict = {}
            for _, content in vehs.iterrows():
                veh_dict[content['track_id']] = np.array(content)
            rec_dict[frame] = veh_dict
            i_dicting += 1
        
        # Processing
        # We create an unidata for every frame in the scene and distribute it at random among the test, train, and validation sets.
        i_process = 0
        for frame in frame_list[:-total_len + 1]:
            if frame % 1000 == 1:
                print('folder: ', i, ' csv: ', j, 'final_process:', i_process / len(frame_list))
            start_frame = int(frame)
            mid_frame = start_frame + hist_len - 2 # Hist and fut frames are separated by the mid_frame.
            end_frame = start_frame + total_len -2
            
            # end_frame not available
            if not end_frame in frame_list:
                continue

            # Find vehicles in strat and end frames
            vehs_start = list(rec_dict[start_frame].keys())
            vehs_end = list(rec_dict[end_frame].keys())
            
            # Filter vehicle data
            jump = False
            if padding:
                # Utilize vehicles that are accessible across all history frames. Zero-padding for possible missing frames in future frames.
                vehs_filt = vehs_start
                for f in range(start_frame, mid_frame + 1, 2): # Sub-sampling with rate 2 Hz
                    if not f in frame_list: # Frame not available
                        jump = True
                        break   
                    vehs_filt = list(set(vehs_filt) & set(rec_dict[f].keys()))    
            else:
                # Utilize vehicles that are accessible across all frames(Both history and future).
                for f in range(start_frame, end_frame + 1, 2):
                    if not f in frame_list:
                        jump = True
                        break
                vehs_filt = list(set(vehs_start) & set(vehs_end))
                
            # Not enough vehicles for trajectory prediction    
            if len(vehs_filt) < 2 or jump:
                continue
            
            # Find the maximum number of vehicles present in the scene.
            counter[len(vehs_filt)] = counter[len(vehs_filt)] + 1
                
            # Extract trajectory
            features = []
            for veh in vehs_filt:
                traj = []
                # Extract historical and future trajectories of vehicles.
                for f in range(start_frame, end_frame + 1, 2):
                    if rec_dict.get(f) is None: # if vehicle not exists
                        rec = {}
                    else: # if vehicle exists    
                        rec = rec_dict[f]
                    if not rec.get(veh) is None: # if vehicle exists
                        traj.append(np.concatenate((rec.get(veh), [i, 1]))) # Flag:1 vehicle exists
                    else:
                        traj.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, i, 0])) # Flag:0 vehicle absences
                        # skip several frames
                        if f in [mid_frame + 2,  mid_frame + 4, mid_frame + 6]:
                            jump = True
                traj = np.stack(traj)
                features.append(traj)
            # Construct unidata    
            features = np.array(np.stack(features))
            if jump:
                continue
            
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
            np.save(path + str(i) + '_' + str(name_index) + '.npy', features)
            
            i_process += 1
            name_index += 1
            
np.save('count_interaction.npy', counter) # Maximum number of vehicles present in the scene


