import torch
import torch_geometric as pyg
import json
import pickle
import os
import numpy as np
import random
from dgl.geometry import farthest_point_sampler
from scipy.spatial import Delaunay


def generate_noise(position_seq, noise_std):
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise

class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data_path=None, split='valid',noise_std=0.0, return_pos=False, sampling = False, sampling_strategy='random', sampling_percent_lb = 0.18, sampling_percent_ub=0.200, sample_mesh_size=None, train_on_displacement=False):
        super().__init__()

        # load dataset from the disk
        #self.data_path = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
        self.data_path = data_path
        with open(os.path.join(self.data_path, "metadata.json")) as f:
            self.metadata = json.load(f)


        self.noise_std = noise_std
        self.sample_mesh_size = sample_mesh_size
        self.return_pos = return_pos

        self.tensorpath = os.path.join(self.data_path, split)
        if('.pt' in self.tensorpath):
            dataset = torch.load(self.tensorpath)
        else:
            file = open(self.tensorpath, 'rb')
            dataset = pickle.load(file)

        self.train_on_displacement = train_on_displacement
        self.position = dataset['position']
        self.sampling = sampling
        self.sampling_strategy = sampling_strategy

        self.dim = self.position[0].shape[2]

        self.sampling_percent_lb = sampling_percent_lb
        self.sampling_percent_ub = sampling_percent_ub

    def len(self):
        return self.position[0].shape[1]

    def get(self, idx):
        # load corresponding data for this time slice
        #[15, 1000, N, 3] --> [TrajIndex, T, N, D]
        traj_index = random.randint(0, len(self.position)-1) #Sample random traj
        #traj_index = 0
        fom_ic = torch.from_numpy(self.position[traj_index]) #Initial condition (T=0)
        fom_gt = torch.from_numpy(self.position[traj_index]) #Current Position at IDX (0, 1000)
        fom_ic = torch.permute(fom_ic, dims=(1,0,2))
        fom_gt = torch.permute(fom_gt, dims=(1,0, 2))
        
        ic_index = 0
        f_ic = fom_ic[ic_index] #T=0
        f_gt = fom_gt[idx] #T=IDX
        

        if(self.sample_mesh_size is not None):
            mesh_size = self.sample_mesh_size
        else:
            mesh_size =  np.random.randint(int(self.sampling_percent_lb*f_ic.shape[0]), int(self.sampling_percent_ub*f_ic.shape[0]))
        if(self.sampling_strategy == 'random'):
            self.points = sorted(random.sample(range(0, f_ic.shape[0]), mesh_size))
        else:
            self.points = range(0, f_ic.shape[0], int(f_ic.shape[0]/mesh_size))

        rom_ic = f_ic[self.points]

        
        rom_f = f_gt[self.points] #- rom_ic
        
        
        velocity_noise = torch.randn_like(rom_f) * (self.noise_std / (1+idx) ** 0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)
        rom_f = rom_f + velocity_noise
        if(self.train_on_displacement):
            f_gt = (f_gt - f_ic)#/(idx-ic_index+0.1)
            rom_f = rom_f - rom_ic
        return f_ic, rom_ic, rom_f, f_gt