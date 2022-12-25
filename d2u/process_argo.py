#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:10 2022

@author: seuiv
"""

import pandas as pd
import numpy as np
import os

counter = np.zeros(500, dtype = np.int) # Maximum number of vehicles present in the scene

process_list = ['train', 'val']
for pro in process_list:
    # Access unidata saving path
    save_path = '/home/lsy/Public/04-Data/UniArgo/' + pro + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Access original path
    data_dir = '/mnt/Data/Argo/' + pro + '/data/'
    csv_list = os.listdir(data_dir)
    
    # Constructing unidata
    for i, name in enumerate(csv_list):
        # Load data ...
        # 0: Time 1: ID, 2:type, 3: x, 4: y, 5: city
        data = pd.read_csv(data_dir + name, usecols = [0, 1, 2, 3, 4, 5])
    
        # Normalization
        frame_min_g = data['TIMESTAMP'].min()       
        data['TIMESTAMP'] -= frame_min_g
        
        # Renumber ID for convenience
        id_set = data.loc[:,'TRACK_ID'].unique()
        id_new = np.arange(len(id_set))
        dic = dict(zip(id_set, id_new))
        data['TRACK_ID'] = data['TRACK_ID'].map(dic)
        
        # Index...
        frame_feature = data.sort_values(by = 'TIMESTAMP')
        frame_list = data.loc[:, 'TIMESTAMP'].unique()
    
        # Build dict ...
        rec_dict = {}
        # All vehicle info is kept in Rec_dict in accordance with the frame.
        if i % 1000 == 1:
            print('pro:', pro, 'process:', i/len(csv_list) * 100, '%')
        # Construct Rec_dict
        i_dicting = 0
        for index, frame in enumerate(frame_list):
            # Find all vehicles in the current frame.
            vehs = frame_feature[frame_feature['TIMESTAMP'] == frame]
            veh_dict = {}
            # All vehicle info in the currnt frame is stored in veh_dict
            for _, content in vehs.iterrows():
                veh_dict[content['TRACK_ID']] = np.array(content)
            rec_dict[frame] = veh_dict
            
            # Only select vehicles that exist in all frames from history to future.
            if index == 0:
                vehs_visible = set(list(veh_dict.keys()))
            else:
                vehs_visible = list(set(vehs_visible) & set(list(veh_dict.keys())))
            # Not enough vehicles for trajectory prediction
            if len(vehs_visible) < 2:
                continue
            i_dicting += 1
        
        # # Maximum number of vehicles present in the scene
        counter[len(vehs_visible)] = counter[len(vehs_visible)] + 1
        
        # Extract trajectory
        features = []
        for veh in vehs_visible:
            jump = False
            traj = []
            count = 0
            for index, f in enumerate(frame_list):
                # Sub-sampling with rate 2 Hz
                if index % 2 == 1:
                    continue
                rec = rec_dict[f]
                if not rec.get(veh) is None: # if vehicle exists
                    traj.append(np.concatenate((rec.get(veh), [1]))) # Flag:1 vehicle exists
                else:
                    traj.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0])) # Flag:0 vehicle absenses    
                    if index in [20, 22, 24]:
                        jump = True
            if jump:
                continue
            traj = np.stack(traj)
            features.append(traj)
        # Construct unidata
        features = np.array(np.stack(features))  
        # Save ...
        np.save(save_path + '/feature_' + str(i) + '.npy', features)

    
np.save('count_argo.npy', counter)

