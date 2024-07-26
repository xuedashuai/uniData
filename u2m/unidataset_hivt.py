#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:36:06 2022

@author: seuiv
"""

import numpy as np
import os
import torch
import math
import random

class UniDataset:
    def __init__(self, folder, params, envs_num, seed):
        self.dir = folder
        self.hist_len = params['hist_len']
        self.fut_len = params['fut_len']
        self.limit = params['limit']
        self.grid = params['grid']
        self.enc_size = params['enc_size']
        self.pos_index = params['xy']
        self.data_list = os.listdir(self.dir)
        self.envs_num = envs_num
        self.shuffles = self.create_shuffle(envs_num, seed)
        
    def __len__(self):
        return len(self.data_list) 
    
    def create_shuffle(self, envs_num, seed):
        seeds = []
        shuffles = []
        
        random.seed(seed)
        for i in range(envs_num):
            seeds.append(random.randint(1, 9999))
            shuffles.append([j for j in range(len(self.data_list))])
            
        for k_i, k in enumerate(shuffles):
            random.seed(seeds[k_i])
            random.shuffle(k)
        
        return shuffles
    
    def __getitem__(self, idx):
        
        sample = []
        
        for shu in self.shuffles:
            ori = np.load(self.dir + self.data_list[shu[idx]], allow_pickle = True)
            
            hist = ori[:, :self.hist_len, self.pos_index:self.pos_index + 2]
            
            op_mask = ori[:, self.hist_len:, -1:]
            fut = ori[:, self.hist_len:, self.pos_index:self.pos_index + 2] - ori[:, self.hist_len - 1:self.hist_len, self.pos_index:self.pos_index + 2]
            
            nbrs = []
        
            sample.append([hist, fut, nbrs, op_mask])
        
        return sample
    
    def collate_fn(self, samples):
        sample_list = []
        batch = []
        for env_i in range(self.envs_num):
            sample_list.append([])
        
        for env_i in range(self.envs_num):
            for sample in samples:
                sample_list[env_i].append(sample[env_i])
        
        for env_i in range(self.envs_num):
            
            # veh_num = 0
            samplenum = 0
            
            for sample in sample_list[env_i]:
                hist = sample[0]
                
                
                num = hist.shape[0]
                
                            
                samplenum += num
                 

            
            # Initialize 
            hist_batch = torch.zeros(self.hist_len, samplenum, 2)
            fut_batch = torch.zeros(self.fut_len, samplenum, 2)
            op_mask_batch = torch.zeros(self.fut_len, samplenum, 2)
            
            
            count = 0
            sample_count = 0
            
            for sample in sample_list[env_i]:
                hist = sample[0]
                fut = sample[1] 
                # nbrs = sample[2]
                op_mask = sample[3] 
                
                num = hist.shape[0]
                
                for k in range(num):
    
                    hist_batch[:self.hist_len, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                    fut_batch[:self.fut_len, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                    
                    op_mask_batch[:self.fut_len, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                                       
                            
                sample_count += num
            
            nbrs_batch = []
            mask_batch = []
            
            batch.append([hist_batch, nbrs_batch, mask_batch, fut_batch, op_mask_batch])
        
        return batch
                    
   