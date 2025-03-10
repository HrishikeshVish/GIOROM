import torch
import torch_geometric as pyg
import json
import pickle
import os
import numpy as np
import random
from dgl.geometry import farthest_point_sampler
from scipy.spatial import Delaunay
from utils.data_utils import *



class OneStepDataset(pyg.data.Dataset):
    def __init__(self, train_dir=None, metadata_dir=None, window_length=7, noise_std=0.0, return_pos=False, sampling = False, sampling_strategy='random', graph_type='radius', sampling_percent_lb = 0.20, sampling_percent_ub=0.33, radius=None, sample_mesh_size=None):
        super().__init__()

        # load dataset from the disk
        #self.data_path = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
        self.data_path = train_dir
        with open(metadata_dir) as f:
            self.metadata = json.load(f)

        self.window_length = window_length
        self.noise_std = noise_std
        self.sample_mesh_size = sample_mesh_size
        self.return_pos = return_pos
        self.radius = radius
        self.graph_type = graph_type
        self.tensorpath = train_dir
        if('.pt' in self.tensorpath):
            dataset = torch.load(self.tensorpath)
        else:
            file = open(self.tensorpath, 'rb')
            dataset = pickle.load(file)

        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']
        self.sampling = sampling
        self.sampling_strategy = sampling_strategy

        self.dim = self.position[0].shape[2]

        self.sampling_percent_lb = sampling_percent_lb
        self.sampling_percent_ub = sampling_percent_ub

    def len(self):
        return len(self.position)

    def get(self, idx):
        # load corresponding data for this time slice


        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        target_position = torch.from_numpy(self.outputs[idx])

        if(self.sampling == True):
            
            mesh_size =  np.random.randint(int(self.sampling_percent_lb*particle_type.shape[0]), int(self.sampling_percent_ub*particle_type.shape[0]))
            while(mesh_size %10 !=0):
                mesh_size += 1
                
            if(self.sample_mesh_size!=None):
                mesh_size = self.sample_mesh_size

            if(self.sampling_strategy == 'random'):
                points = sorted(list(random.sample(range(0, particle_type.shape[0]), mesh_size)))
                #points = sorted(list(range(0, particle_type.shape[0], 3)))
                particle_type = particle_type[points]
                position_seq = position_seq[points]
                target_position = target_position[points]
            elif(self.sampling_strategy == 'fps'):
                init_pos = position_seq.permute(1, 0, 2)[0].unsqueeze(0)
                point_idx = farthest_point_sampler(init_pos, mesh_size)[0]
                particle_type = particle_type[point_idx]
                position_seq = position_seq[point_idx]
                target_position = target_position[point_idx]

        
        # construct the graph
        with torch.no_grad():
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std, self.radius, self.graph_type)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph


class RolloutDataset(pyg.data.Dataset):
    def __init__(self, data_path, metadata_dir, window_length=7, sampling=False, sampling_strategy='random', graph_type='radius', mesh_size=90, radius=None):
        super().__init__()
        
        # load data from the disk
        self.data_path = data_path
        with open(metadata_dir) as f:
            self.metadata = json.load(f)

        self.window_length = window_length
        self.sampling = sampling
        self.graph_type = graph_type
        self.sampling_strategy = sampling_strategy
        self.radius = radius
        self.tensorpath = data_path
        dataset = torch.load(self.tensorpath)
        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']
        self.mesh_size = mesh_size

        self.dim = self.position[0].shape[2]

    def len(self):
        return len(self.position)
    
    def get(self, idx):


        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        position_seq = torch.permute(position_seq, dims=(1,0,2))
        
        target_position = torch.from_numpy(self.outputs[idx])
        if(self.sampling):
            if(self.sampling_strategy == 'random'):
                self.points = sorted(random.sample(range(0, particle_type.shape[0]), self.mesh_size))
                particle_type = particle_type[self.points]
                position_seq = position_seq.permute(1,0,2)
                position_seq = position_seq[self.points]
                position_seq = position_seq.permute(1,0,2)
                #target_position = target_position[self.points]
            elif(self.sampling_strategy == 'fps'):
                init_pos = position_seq.permute(1, 0, 2)[0].unsqueeze(0)
                point_idx = farthest_point_sampler(init_pos, self.mesh_size)[0]
                particle_type = particle_type[point_idx]
                position_seq = position_seq[point_idx]
                #target_position = target_position[point_idx]
        data = {"particle_type": particle_type, "position": position_seq}
        return data