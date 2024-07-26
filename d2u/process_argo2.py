#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:24:10 2022

@author: seuiv
"""

import pandas as pd
import numpy as np
import os

import ArgoverseV2Dataset

process_list = ['train', 'val']
for pro in process_list:
    # Access unidata saving path
    save_path = '/home/lsy/Public/05-Data/UniArgo2/' + pro + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Access original path
    data_dir = '/mnt/Data/Argo2/' + pro + '/data/'
    csv_list = os.listdir(data_dir)
    
    dataset = ArgoverseV2Dataset(root = data_dir, split = pro)

    length = len(dataset)
    
    for i in range(length):
        data = dataset.get(i)
        # Save ...
        np.save(save_path + '/feature_' + str(i) + '.npy', features)

    
