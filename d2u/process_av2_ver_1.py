#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:10 2022

@author: seuiv
"""

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import os
import random

from scipy import spatial
# from config import args_pre_av2
args_pre_av2 = {}
args_pre_av2['hist_len'] = 15
args_pre_av2['fut_len'] = 25
args_pre_av2['seed'] = 42
args_pre_av2['padding'] = True
args_pre_av2['source_path'] = '/mnt/data1/Argoverse2/train/'
args_pre_av2['dest_path'] = '/mnt/data1/Argoverse2/xqf/'
args_pre_av2['tr_ratio'] = 0.7
args_pre_av2['va_ratio'] = 0.1
args_pre_av2['veh_filter'] = 3

# Parameters
hist_len = args_pre_av2['hist_len']
fut_len = args_pre_av2['fut_len']
total_len = hist_len + fut_len

np.random.seed(args_pre_av2['seed'])
random.seed(args_pre_av2['seed'])
padding = args_pre_av2['padding']

source_path = args_pre_av2['source_path']
dest_path = args_pre_av2['dest_path']

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

parquet_list = os.listdir(source_path)

name_index = 0

for i, name in enumerate(parquet_list):
    
    # Load data ...
    scenario_path = source_path +'/'+ name + '/scenario_' + name + '.parquet'
    data = pd.read_parquet(scenario_path)

    # Transfer ...
    # Av2 {0: observability, 1:veh_id, 2: object_type, 3: object_category, 4: frame_id, 5: global_x, 6: global_y, 7: heading, 8: v_x, 9: v_y}
    # object_type: {vehicle, pedestrian, motorcyclist, cyclist, bus, STATIC}
    # object_category: {0: track_fragment, 1: unscored_track, 2: scored_track, 3: focal_track}
    data = pd.DataFrame(data).iloc[:, [1, 4, 5, 6]]
    data.columns = ['veh_id', 'frame_id', 'x', 'y']
    
    # Index ...
    print('Indexing...')
    frame_feature = data.sort_values(by = 'frame_id')
    frame_list = pd.unique(frame_feature['frame_id'])
    
    # Build dict ...
    rec_dict = {}
    
    i_dicting = 0
    for frame in frame_list:
        
        if i_dicting % (len(frame_list)//20) == 1:
            print('file:', i+1, '/', len(parquet_list),' dict_process: %.1f' % (100.0 * i_dicting / len(frame_list)), '%')
        
        vehs = frame_feature[frame_feature['frame_id'] == frame]
        veh_dict = {}
        for _, content in vehs.iterrows():
            veh_dict[content['veh_id']] = np.array(content)
        rec_dict[int(frame)] = veh_dict
        
        i_dicting += 1
    
    print('Generating...')
    
    i_process = 0
    for frame in frame_list[:-total_len + 1]:
        
        if i_dicting % (len(frame_list)//20) == 1:
            print('file:', i+1, '/', len(parquet_list), 'final_process: %.1f' % (100.0 * i_process / len(frame_list)), '%')
        start_frame = int(frame)
        mid_frame = start_frame + hist_len - 2
        end_frame = start_frame + total_len -2
        
        vehs_start = list(rec_dict[start_frame].keys())
        vehs_end = list(rec_dict[end_frame].keys())
        
        if padding:
            vehs_filt = vehs_start
            
            for f in range(start_frame, mid_frame + 1, 2):
                vehs_filt = list(set(vehs_filt) & set(rec_dict[f].keys()))
            
        else:
            for f in range(start_frame, end_frame + 1, 2):
                vehs_filt = list(set(vehs_filt) & set(rec_dict[f].keys()))
            

        if len(vehs_filt) < args_pre_av2['veh_filter']:
            continue
        
        features = []
        for veh in vehs_filt:
            traj = []
            
            for f in range(start_frame, end_frame + 1, 2):
                
                rec = rec_dict[f]
                
                if not rec.get(veh) is None: 
                    
                    traj.append(np.concatenate((rec.get(veh), [1])))
                else:
                    traj.append(np.array([np.nan, np.nan, np.nan, np.nan, 0]))
                
            traj = np.stack(traj)
            
            features.append(traj)
        features = np.array(np.stack(features))
        
        ran = random.random()
        if ran < args_pre_av2['tr_ratio']:
            path = dest_path + 'train/'
        elif ran < args_pre_av2['tr_ratio'] + args_pre_av2['va_ratio']:
            path = dest_path + 'val/'
        else:
            path = dest_path + 'test/'
        
        # Save ...
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + 'ngsim_' + str(name_index) + '.npy', features)
        
        i_process += 1
        name_index += 1
