import numpy as np


#import meshio
import math

import h5py
import torch
import random


#input_dataset = torch.load('/home/hviswan/Documents/Neural Operator/rollout_ground_truth.pt') #INPUT SHAPE = 482,1000,2
input_dataset = torch.load('/home/csuser/Documents/Neural Operator/human_pred.pt') 
#input_dataset['old_positions'] = torch.permute(input_dataset['old_positions'], dims=(1, 0, 2))
print(input_dataset.shape)
print(len(input_dataset))
counter = 0

for i in range(39):
        current_position = input_dataset[i]
        x = current_position - input_dataset[0]
        x0 = input_dataset[0]
        filename_h5 = '/home/csuser/Documents/h5_outs/human/h5_f_' + str(i).zfill(10) + '.h5'
        with h5py.File(filename_h5, 'w') as h5_file:
                dset = h5_file.create_dataset("x", data=x0.T)
                dset = h5_file.create_dataset("q", data=x.T)




input_dataset = torch.load('/home/hviswan/Documents/Neural Operator/rollout_predicted.pt') #INPUT SHAPE = 482,1000,2
#input_dataset = torch.permute(input_dataset, dims=(1, 0, 2))
print(input_dataset.shape)
#print(len(input_dataset['old_positions']))
counter = 0
for i in range(320):
        current_position = input_dataset[i]
        x = current_position - input_dataset[0]
        x0 = input_dataset[0]
        filename_h5 = '/home/hviswan/Documents/Neural Operator/PCA/Sand/predicted/h5_f_' + str(i).zfill(10) + '.h5'
        with h5py.File(filename_h5, 'w') as h5_file:
                dset = h5_file.create_dataset("x", data=x0.T)
                dset = h5_file.create_dataset("q", data=x.T)
"""
for j in range(100):
        positions = torch.from_numpy(input_dataset['old_positions'][j])
        if(positions.shape[0]<512):
               continue
        points = sorted(list(random.sample(range(0, positions.shape[0]), 512)))
        positions = positions[points]
        temporal_positions = torch.permute(positions, dims=(1, 0, 2)).numpy()
        #print(temporal_positions.shape)

        for i in range(1000):
                print(" i = ", i, " j = ", j, " 1000*j +i = ", 1000*j + i)
                current_position = temporal_positions[i]
                x = current_position - temporal_positions[0]
                x0 = temporal_positions[0]
                filename_h5 = '/home/hviswan/Documents/Neural Operator/PCA/Sand/ground_truth/h5_f_' + str(1000*j + i).zfill(10) + '.h5'
                with h5py.File(filename_h5, 'w') as h5_file:
                        dset = h5_file.create_dataset("x", data=x0.T)
                        dset = h5_file.create_dataset("q", data=x.T)

"""
"""
#ps_vol = ps.register_volume_mesh("test volume mesh", mesh.points, mesh.cells_dict['tetra'])

t = 0.
idx = 0

x0 = None

while(1):
    filename_obj = "path_to_obj" #'/home/changyue/Desktop/LiCROM_cloth/cloth_training_data/snapshot_' + str(idx) + '.obj'
    filename_h5 = "path_to_h5" #'cloth_basis_0920/sim_seq_cloth/h5_f_' + str(idx).zfill(10) + '.h5'
    mesh = meshio.read(filename_obj)
    if(x0 is None):
       x0 = mesh.points.astype(np.float64) # initialize the initial position
    
    x = mesh.points.astype(np.float64) - x0 # calculate displacement
    #print(x.shape, x0.shape)
    #print(x0)
    
    with h5py.File(filename_h5, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data = x0.T)
            dset = h5_file.create_dataset("q", data = x.T)
    idx = idx + 1
"""