import torch
import torch_geometric as pyg
import json
import pickle
import os
import numpy as np
import random
#from dgl.geometry import farthest_point_sampler
from scipy.spatial import Delaunay

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import ball_query, knn_points

import open3d.ml.torch as ml3d

def voxelize_grid(grid_points, voxel_size = 2, grid_size = 64):
    num_voxels = grid_size//voxel_size
    row_splits = torch.Tensor([0, grid_points.shape[0]]).to(torch.int64)
    voxel_size = (grid_points.max(dim=0)[0] - grid_points.min(dim=0)[0])/num_voxels
    print(f"Voxel size for {grid_size}: {voxel_size}")
    points_range_min = grid_points.min(dim=0)[0]-1e-3
    points_range_max = grid_points.max(dim=0)[0]+1e-3
    vox_coord, pt_ind, ptr_sp, b_sp = ml3d.ops.voxelize(grid_points,
                  row_splits,
                  voxel_size=voxel_size,
                  points_range_min=points_range_min,
                  points_range_max=points_range_max)
    # print(vox_coord)
    # print(pt_ind[:10]) #[262144] / [2097152]
    # print(ptr_sp[:10]) # [35938]
    # print(ptr_sp[:10])
    return pt_ind, ptr_sp #voxel_point_indices, voxel_point_row_splits,

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
    def __init__(self, data_path=None, split='valid',noise_std=0.0, return_pos=False, sampling = False, sampling_strategy='random', sampling_percent_lb = 0.18, 
                 sampling_percent_ub=0.200, sample_mesh_size=None, train_on_displacement=False, load_quad=False):
        super().__init__()

        # load dataset from the disk
        #self.data_path = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
        self.data_path = data_path
        # with open(os.path.join(self.data_path, "metadata.json")) as f:
        #     self.metadata = json.load(f)


        self.noise_std = noise_std
        self.sample_mesh_size = sample_mesh_size
        self.return_pos = return_pos

        self.tensorpath = self.data_path
        if('.pt' in self.tensorpath):
            dataset = torch.load(self.tensorpath)
        else:
            file = open(self.tensorpath, 'rb')
            dataset = pickle.load(file)

        self.train_on_displacement = train_on_displacement
        self.position = dataset
        self.sampling = sampling
        self.sampling_strategy = sampling_strategy

        if(self.sample_mesh_size is None):
            self.sample_mesh_size = 700
        fom_ic = self.position[0]   #Initial condition (T=0)
        self.points = sorted(random.sample(range(0, fom_ic.shape[0]), self.sample_mesh_size))
        rom_ic = fom_ic[self.points]

        vox_pt_fom, vox_split_320 = voxelize_grid(fom_ic, voxel_size=4, grid_size=12)
        vox_pt_rom, vox_split_32 = voxelize_grid(rom_ic, voxel_size=4, grid_size=12)
        print(len(vox_split_32))
        print(len(vox_split_320))
        print(vox_pt_fom)
        print(vox_pt_rom)
        
        voxel_map_32 = {}
        voxel_map_320 = {}

        fom_ic = fom_ic[vox_pt_fom]
        rom_ic = rom_ic[vox_pt_rom]
        split_data = []
        self.data_quads = []
        if(load_quad == False):
            for i in range(self.position.shape[0]):
                voxel_map_32[i] = {}
                voxel_map_320[i] = {}
                print("Time Step: ", i)
                for idx in range(1, vox_split_32.shape[0]):
                    
                    voxel_points = rom_ic[vox_split_32[idx-1]:vox_split_32[idx]].clone()
                    voxel_f_rom = self.position[i][self.points]
                    voxel_f_rom = voxel_f_rom[vox_pt_rom]
                    voxel_f_rom = voxel_f_rom[vox_split_32[idx-1]:vox_split_32[idx]].clone()
                    key = tuple(voxel_points.mean(dim=0).numpy())
                    if key in voxel_map_32[i].keys():
                        print("Key already exists")
                        exit()
                    else:
                        voxel_map_32[i][key] = [voxel_points, voxel_f_rom]
                for idx in range(1, vox_split_320.shape[0]):
                    voxel_points = fom_ic[vox_split_320[idx-1]:vox_split_320[idx]].clone()
                    voxel_f_fom = self.position[i]
                    voxel_f_fom = voxel_f_fom[vox_pt_fom]
                    voxel_f_fom = voxel_f_fom[vox_split_320[idx-1]:vox_split_320[idx]].clone()
                    key = tuple(voxel_points.mean(dim=0).numpy())
                    if key in voxel_map_320[i].keys():
                        print("Key already exists")
                        exit()
                    else:
                        voxel_map_320[i][key] = [voxel_points, voxel_f_fom]
                map_rom_centroids = torch.from_numpy(np.asarray(list(voxel_map_32[i].keys())))
                map_fom_centroids = torch.from_numpy(np.asarray(list(voxel_map_320[i].keys())))
                diff = map_fom_centroids.unsqueeze(1) - map_rom_centroids.unsqueeze(0)
                distances = torch.norm(diff, dim=2).cpu()
                min_distances, min_indices = torch.min(distances, dim=1)
                zero_counter = 0
                for idx in range(map_fom_centroids.shape[0]):
                    map_key = tuple(map_fom_centroids[idx].numpy())
                    if map_key not in voxel_map_320[i].keys():
                        print("Key not found")
                        exit()
                    target_pair = voxel_map_320[i][map_key]
                    map_index = min_indices[idx].item()
                    target_map_key = tuple(map_rom_centroids[map_index].numpy())
                    if target_map_key not in voxel_map_32[i].keys():
                        print("Key not found")
                        exit()
                    input_pair = voxel_map_32[i][target_map_key]
                    train_quad = [input_pair, target_pair, i]
                    self.data_quads.append(train_quad)
            torch.save({'data':self.data_quads}, "data_quads.pt")
            print(len(self.data_quads))
        else:
            data_quads = torch.load("data_quads.pt")
            self.data_quads = data_quads['data']
            print(len(self.data_quads))
        self.dim = 3

        self.sampling_percent_lb = sampling_percent_lb
        self.sampling_percent_ub = sampling_percent_ub

    def len(self):
        return len(self.data_quads)

    def get(self, idx):
        # load corresponding data for this time slice
        #[15, 1000, N, 3] --> [TrajIndex, T, N, D]
        #traj_index = random.randint(0, len(self.position)-1) #Sample random traj
        #traj_index = 0
        train_quad = self.data_quads[idx]
        rom_ic, rom_f = train_quad[0]
        fom_ic, fom_f = train_quad[1]
        time_step = train_quad[2]

        #f_ic = self.position[0] #Initial condition (T=0)
        #f_gt = self.position[idx] #Current Position at IDX (0, 1000)
        #fom_ic = torch.permute(fom_ic, dims=(1,0,2))
        #fom_gt = torch.permute(fom_gt, dims=(1,0, 2))
        
        #ic_index = 0
        #f_ic = fom_ic[ic_index] #T=0
        #f_gt = fom_gt[idx] #T=IDX
        

        #if(self.sample_mesh_size is not None):
        #    mesh_size = self.sample_mesh_size
        #else:
        #    mesh_size =  np.random.randint(int(self.sampling_percent_lb*f_ic.shape[0]), int(self.sampling_percent_ub*f_ic.shape[0]))
        #if(self.sampling_strategy == 'random'):
        #    self.points = sorted(random.sample(range(0, f_ic.shape[0]), mesh_size))
        #else:
        #    self.points = range(0, f_ic.shape[0], int(f_ic.shape[0]/mesh_size))

        #rom_ic = f_ic[self.points]

        
        #rom_f = f_gt[self.points] #- rom_ic
        
        
        #velocity_noise = torch.randn_like(rom_f) * (self.noise_std / (1+idx) ** 0.5)
        #velocity_noise = velocity_noise.cumsum(dim=1)
        #rom_f = rom_f + velocity_noise
        #if(self.train_on_displacement):
        #    f_gt = (f_gt - f_ic)#/(idx-ic_index+0.1)
        #    rom_f = rom_f - rom_ic
        dt = time_step * 7.5e-3
        if(time_step == 0):
            dt = 1e-3
        rom_f = (rom_f - rom_ic)/dt
        return fom_ic, rom_ic, rom_f, fom_f, time_step