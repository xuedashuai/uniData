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

class UniDataset:
    def __init__(self, folder, params):
        self.dir = folder # unidata location
        self.hist_len = params['hist_len'] # history trajectory length
        self.fut_len = params['fut_len'] # future trajectory length
        self.lane = params['use_lane'] # is lane available
        
        if self.lane:
            self.lane_index = params['lane_index'] # lane index
        self.enc_size = params['enc_size'] # size of lstm encoder
        
        self.data_list = os.listdir(self.dir) # unidata list
        self.params = params
        
        if self.params.get('type_map'): # is map available
            self.type_map = params['type_map'] # map
        else:
            self.type_map = None
    
    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        # Load unidata
        ori = np.load(self.dir + self.data_list[idx], allow_pickle = True)
        
        # Initialize index
        index1 = self.params['xy']
        index2 = self.params['type']
        # Is scene available
        if self.params.get('scene'):
            index3 = self.params['scene']
        else:
            index3 = 0
        if not self.lane: 
            # If lane is not available, grid is custmized
            self.limit = self.params['limit'] # ratio for grid positioning
            self.grid = self.params['grid']
        else:
            # If lane is available, grid is set by default
            self.grid = [3, 13]
        # Get data
        hist = ori[:, :self.hist_len, index1:index1 + 2]
        fut = ori[:, self.hist_len:, index1:index1 + 2] - ori[:, self.hist_len - 1:self.hist_len, index1:index1 + 2] # Relative location
        # Get op_mask to skip unavailable data for loss computing.
        op_mask = ori[:, self.hist_len:, -1:]
        tp = ori[:, self.hist_len: ,index2:index2 + 1] # type
        scene = ori[:, self.hist_len:, index3:index3 + 1] # scene
        num = hist.shape[0] # Number of vehicles in current unidataset
        
        # Construct Maneuver and Neighbor grid matrix for each agent in unidata
        if self.lane:
            # If lane is available
            # Get lane index and upbound
            lane =ori[:,:,self.lane_index] # shape: [N,T] 
            ub = ori.shape[1]-1 # time upbound
            # Construct Maneuver
            # Initialize longtidinal and lateral maneuver
            lon = np.zeros([num, 2])
            lat = np.zeros([num, 3])
            for i in range(num): 
                # Left-hand side, right-hand side or straight maneuver 
                if (lane[i,ub]>lane[i,int(ub/2)]) or(lane[i,int(ub/2)]>lane[i,0]):
                    lat[i, 2]=1
                elif (lane[i,ub]<lane[i,int(ub/2)]) or(lane[i,int(ub/2)]<lane[i,0]):
                    lat[i, 1]=1
                else:
                    lat[i, 0]=1
                # Compare history and future average velocity to determine lon 
                vhist=(hist[i, -1, 1]-hist[i, 0,1])/hist.shape[1]
                vfut=(fut[i, -1, 1]-fut[i, 0,1])/fut.shape[1]
                if (vfut/(vhist+0.01))<0.8:
                    lon[i, 1]=1
                else:
                    lon[i, 0]=1
            # Construct nbrs grid matrix
            # Initialize grid matrix (-1 indicate empty grid without vehicles)
            grid = np.zeros([num, 3, 13])-1
            for i in range(num): # construct nbrs grid matrix for vehicle i
                for j in range(num):
                    if j==i: # skip current vehicle
                        continue
                    tind = hist.shape[1]-1 # time index for final frame
                    y = hist[j,-1,1]-hist[i,-1,1] # y distance
                    # Grid positioning when lane is available
                    if lane[i,tind]<lane[j,tind]:
                        if abs(y)<90:
                            grid[i,2,int((y+90)/15)]=j
                    elif lane[i,tind]==lane[j,tind]:
                        if abs(y)<90:
                            grid[i,1,int((y+90)/15)]=j
                    elif lane[i,tind]>lane[j,tind]:
                        if abs(y)<90:      
                            grid[i,0,int((y+90)/15)]=j
                    else: # vehicle j is not nbrs of vehicle i
                        continue
        else:
            # if lane is unavailable, set lon and lat to zero
            lon = np.zeros([num, 2])
            lat = np.zeros([num, 3])
            # Construct grid matrix
            grid = np.zeros([num, self.grid[0], self.grid[1]])-1
            for i in range(num): # construct nbrs grid matrix for vehicle i
                x_i = hist[i, -1, 0]
                y_i = hist[i, -1, 1]
                for j in range(num): 
                    if j == i: # skip current vehicle
                        continue
                    x_j = hist[j, -1, 0]
                    y_j = hist[j, -1, 1]
                    # Grid positioning
                    g_x = (x_j - x_i) / self.limit
                    g_y = (y_j - y_i) / self.limit
                    # X axis positioning
                    if g_x > 0 and g_x < (self.grid[0] - 1)/2:
                        g_x = math.ceil(g_x) + (self.grid[0] - 1)/2
                    elif g_x < 0 and abs(g_x) < (self.grid[0] - 1)/2:
                        g_x = math.floor(g_x) + (self.grid[0] - 1)/2
                    else:
                        continue
                    # Y axis positioning
                    if g_y > 0 and g_y < (self.grid[1] - 1)/2:
                        g_y = math.ceil(g_y) + (self.grid[1] - 1)/2
                    elif g_y < 0 and abs(g_y) < (self.grid[1] - 1)/2:
                        g_y = math.floor(g_y) + (self.grid[1] - 1)/2
                    else:
                        continue
                    grid[i, int(g_x), int(g_y)] = j
        
        # Get Neighbor trajectories for each agents in unidata
        nbrs = []
        for i in range(num):
            # Convert nbrs grid matrix to line
            grid_line = grid.reshape([num, -1])
            # Get Neighbor trajectories for an agent
            nbr = []
            for j in grid_line[i]:
                # j is the index of nbrs in hist
                if j != -1:
                    # Relative position
                    nbr.append(hist[int(j)] - hist[int(i)][-1:, :])
                else:
                    # Empty nbr
                    nbr.append(np.empty([0,2]))
            nbrs.append(nbr)

        return hist, fut, num, lat, lon, nbrs, tp, op_mask, scene
    
    def collate_fn(self, samples):
        nbrs_num = 0 # Number of nbr vehicles in current batch 
        veh_num = 0 # Number of vehicles in current batch
        # Attention: Not all vehicles are neighbors.
        for _, _, num, _, _, nbrs, _, _, _ in samples:
            # Get total nbrs amount
            for j in nbrs:
                nbrs_num += sum([len(j[i])!=0 for i in range(len(j))]) 
            # Get total vehicles amount
            veh_num += num

        # Initialize batch size
        hist_batch = torch.zeros(self.hist_len, veh_num, 2)
        fut_batch = torch.zeros(self.fut_len, veh_num, 2)
        nbrs_batch = torch.zeros(self.hist_len, nbrs_num, 2)
        op_mask_batch = torch.ones(self.fut_len, veh_num, 2)
        lat_enc_batch = torch.zeros(veh_num,3)
        lon_enc_batch = torch.zeros(veh_num, 2)
        tp_batch   = torch.zeros(self.fut_len, veh_num, 2) 
        scene_batch   = torch.zeros(self.fut_len, veh_num, 2) 
        mask_batch = torch.zeros(veh_num, self.grid[0], self.grid[1], self.enc_size)
        mask_batch = mask_batch.byte()
        pos=[0,0]
        
        # Main Collate function
        count = 0
        sample_count = 0
        for hist, fut, num, lat, lon, nbrs, tp, op_mask, scene in samples:
            # Collate function for data about each agent
            # Set up hist, fut, op_mask, lat, lon, map and scene batches:
            for k in range(num):
                hist_batch[:self.hist_len, sample_count + k, :] = torch.from_numpy(hist[k].astype(float))
                fut_batch[:self.fut_len, sample_count + k, :] = torch.from_numpy(fut[k].astype(float))
                
                # print(op_mask.shape)
                op_mask_batch[:self.fut_len, sample_count + k,:] = torch.from_numpy(op_mask[k].astype(int))
                
                lat_enc_batch[sample_count + k,:] = torch.from_numpy(lat[k].astype(float))
                lon_enc_batch[sample_count + k, :] = torch.from_numpy(lon[k].astype(float))
                
                if not self.type_map == None:
                    tp_batch[:self.fut_len,sample_count + k,:] = torch.from_numpy(np.vectorize(self.type_map.get)(tp[k]).astype(int))
            
                scene_batch[:self.fut_len,sample_count + k,:] = torch.from_numpy(scene[k].astype(int))
              
            # Collate function for data about nbrs of each agnt        
            # Set up neighbor, neighbor sequence length, and mask batches:
            for k, singlenbr in enumerate(nbrs):
                for id,nbr in enumerate(singlenbr):
                    if len(nbr)!=0:
                        nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0].astype(float))
                        nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1].astype(float))
                        pos[0] = id % self.grid[1]
                        pos[1] = id // self.grid[1]
                        mask_batch[sample_count + k,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                        count+=1
            sample_count += num
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,fut_batch, op_mask_batch, tp_batch, scene_batch
                    
   