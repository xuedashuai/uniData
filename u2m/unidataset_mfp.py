#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:36:06 2022

@author: seuiv
"""

import numpy as np
import os
import torch

from scipy import spatial

class UniDataset:
    def __init__(self, folder, params):
        self.dir = folder # unidata location
        self.hist_len = params['hist_len'] # history trajectory length
        self.fut_len = params['fut_len'] # future trajectory length

        self.nbr_range = params['nbr_range'] 
        self.params = params
        
        self.data_list = os.listdir(self.dir)
        
        if self.params.get('type_map'): # is map available
            self.type_map = params['type_map']
        else:
            self.type_map = None
    
    def __len__(self):
        return len(self.data_list) 

        
    def __getitem__(self, idx):
        # Load unidata
        ori = np.load(self.dir + self.data_list[idx], allow_pickle=True)      
        # Number of vehicles in current unidataset
        num = ori.shape[0]
        # Initialize index
        index1 = self.params['xy']
        index2 = self.params['type']
        # Is scene available
        if self.params.get('scene'):
            index3 = self.params['scene']
        else:
            index3 = 0   
        # Judge neighbors
        xy = ori[:, self.hist_len - 1, index1:index1 + 2] # Reference frame (last frame of history)
        tp = ori[:, self.hist_len:, index2:index2 + 1] # Type of all vehicles
        scene = ori[:, self.hist_len:, index3:index3 + 1] # Scene of all vehicles
        op_mask = ori[:, self.hist_len:, -1:] # Get op_mask to skip unavailable data for loss computing.
        dist = spatial.distance.cdist(xy, xy) # Calculate distance between each vehicles in reference frame to construct distance matrix
        dist = (dist < self.nbr_range).astype(int) # Convert to binary mask matrix
        mask = np.eye(num) # Unit diagonal matrix
        dist = dist - mask # Construct distance mask matrix

        # Accquire grid index of nbrs
        nbrs = []        
        for i in range(num):
            dst = np.nonzero(dist[i])
            if len(list(dst[0])) == 0:
                nbrs.append(None)
            else:
                nbrs.append(list(dst[0]))
        
        # Extract history, future and nbrs trajectory
        hist = []
        fut = []
        neighbors = {} 
        for ind in range(num):
            hist.append(ori[ind, :self.hist_len, index1:index1 + 2]) # history trajectory   
            fut.append(ori[ind, self.hist_len:, index1:index1 + 2] - ori[ind, self.hist_len - 1:self.hist_len, index1:index1 + 2]) # future trajectory
            # Extract nbrs trajectory
            neighbors[ind] = []
            if not nbrs[ind] == None:
                for nbr in nbrs[ind]:
                    neighbors[ind].append((nbr, nbr, nbr)) # For MFP
        return hist, fut, neighbors, op_mask, tp, scene
    
    def collate_fn(self, samples):
        """Prepare a batch suitable for MFP training."""
        nbr_batch_size = 0 # Number of nbr vehicles in current batch  
        num_samples = 0 # Number of vehicles in current batch
        for _,_,nbrs, _ , _, _ in samples:
          # Get total nbrs amount
          nbr_batch_size +=  sum([len(nbr) for nbr in nbrs.values() ])
          # Get total vehicles amount
          num_samples += len(nbrs)
          
        # Initialize batch
        if nbr_batch_size <= 0: # nbrs batch      
          nbrs_batch = torch.zeros(self.hist_len,1,2)
        else:
          nbrs_batch = torch.zeros(self.hist_len,nbr_batch_size,2)
        nbr_inds_batch = None
        hist_batch = torch.zeros(self.hist_len,num_samples, 2)
        fut_batch  = torch.zeros(self.fut_len, num_samples, 2)
        mask_batch = torch.zeros(self.fut_len, num_samples, 2) 
        tp_batch   = torch.zeros(self.fut_len, num_samples, 2) 
        scene_batch= torch.zeros(self.fut_len, num_samples, 2) 
        context_batch = None 
        nbrs_infos = []
        
        # Construct batch
        count = 0
        samples_so_far = 0 # Calculate total vehicles in batch
        for sampleId,(hist, fut, nbrs, op_mask, tp, scene) in enumerate(samples):    
          num = len(nbrs) # number of vehicles
          for j in range(num):
            hist_batch[0:len(hist[j]), samples_so_far+j, :] = torch.from_numpy(hist[j].astype(float))
            fut_batch[0:len(fut[j]), samples_so_far+j, :] = torch.from_numpy(fut[j].astype(float))    
            mask_batch[0:len(fut[j]),samples_so_far+j,:] = torch.from_numpy(op_mask[j].astype(int))
            # Map available
            if not self.type_map == None:
                tp_batch[0:len(fut[j]),samples_so_far+j,:] = torch.from_numpy(np.vectorize(self.type_map.get)(tp[j]).astype(int))
            scene_batch[0:len(fut[j]),samples_so_far+j,:] = torch.from_numpy(scene[j].astype(int))
          samples_so_far += num
          # Construct nbrs batch
          nbrs_infos.append(nbrs)
          # nbrs is a dictionary of key to list of nbr (batch_index, veh_id, grid_ind)
          for batch_ind, list_of_nbr in nbrs.items():
            for batch_id, vehid, grid_ind in list_of_nbr:          
              if batch_id >= 0:
                nbr_hist = hist[batch_id]                                    
                nbrs_batch[0:len(nbr_hist),count,:] = torch.from_numpy( nbr_hist.astype(float) )
                count+=1
        return (hist_batch, nbrs_batch, nbr_inds_batch, fut_batch, mask_batch, context_batch, nbrs_infos, tp_batch, scene_batch)
        
   