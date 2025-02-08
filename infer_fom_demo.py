from models.fom_gno import PhysicsEngine as FOM
import torch
import numpy as np
import os
import json
import random
import torch_geometric as pyg
from tqdm import tqdm
random.seed(42)

import argparse
from argparse import Namespace
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--train_config", help="Train Config File Name")
parser.add_argument("-n", "--noise", help='Noise', default=3e-4, type=float)
parser.add_argument("-m", "--model", help='Model', choices=['FOM_GNO'])
parser.add_argument("-d", "--dataset", help='Dataset')
parser.add_argument("-drt", "--dataset_rootdir", help='Dataset Rootdir')
parser.add_argument("-ckpt", "--ckpt_name", help='Checkpoint Name (Usually model name underscore datasetname)')
parser.add_argument("-mc", "--model_config", help='Model configs (usually modelname.yaml)')


params = parser.parse_args()
if(params.train_config is not None):
    if(params.train_config.endswith('.yaml') == False):
        params.train_config += '.yaml'
    config_path = os.path.join(os.getcwd(), 'configs', params.train_config)
    if(os.path.exists(config_path) == False):
        raise Exception("Invalid Config Name")
    with open(config_path, 'r') as f:
        params = yaml.full_load(f)
    params = Namespace(**params)

if(params.model_config is not None):
    if(params.model_config.endswith('.yaml') == False):
        params.model_config += '.yaml'
    model_config_path = os.path.join(os.getcwd(), 'configs', params.model_config)
    if(os.path.exists(model_config_path) == False):
        raise Exception("Invalid Model config path")
    with open(model_config_path, 'r') as f:
        model_config = yaml.full_load(f)
    

print(model_config)


print(params)

if(params.model is None):
    raise Exception("Model not specified")
elif(params.model == 'fom_gno'):
    from models.fom_gno import PhysicsEngine
else:
    raise Exception("Invalid model name")

if(params.dataset is None):
    raise Exception("Dataset not specified")
if(params.dataset_rootdir is None):
    raise Exception("Dataset rootdir not specified")

if(os.path.exists(params.dataset_rootdir)):
    dataset_dir = os.path.join(params.dataset_rootdir, params.dataset)
else:
    raise Exception("Dataset directory Invalid")

checkpoint_directory = os.path.join(os.getcwd(), 'saved_models')
if(os.path.exists(checkpoint_directory) == False):
    os.mkdir(checkpoint_directory)
if(params.load_checkpoint == True):
    if(params.ckpt_name is None):
        raise Exception("No checkpoint Name specified")
    if(params.ckpt_name.endswith('.pt') == False):
        params.ckpt_name += '.pt'
    checkpoint = os.path.join(checkpoint_directory, params.ckpt_name)
    if(os.path.exists(checkpoint)==False):
        raise Exception("Invalid Checkpoint Directory")

log_directory = os.path.join(os.getcwd(), 'logs')
if(os.path.exists(log_directory) == False):
    os.mkdir(log_directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(params.model_config is not None):
    simulator = FOM(device, **model_config)
else:
    simulator = FOM(device)

ckpt = torch.load(checkpoint)
weights = torch.load(checkpoint)['model']

model_dict = simulator.state_dict()
ckpt_dict = {}

#print(simulator.keys())
model_dict = dict(model_dict)

for k, v in weights.items():
    k2 = k[0:]
    #print(k2)
    if k2 in model_dict:
        #print(k2)
        if model_dict[k2].size() == v.size():
            ckpt_dict[k2] = v
        else:
            print("Size mismatch while loading! %s != %s Skipping %s..."%(str(model_dict[k2].size()), str(v.size()), k2))
            mismatch = True
    else:
        print("Model Dict not in Saved Dict! %s != %s Skipping %s..."%(2, str(v.size()), k2))
        mismatch = True
if len(simulator.state_dict().keys()) > len(ckpt_dict.keys()):
    mismatch = True
model_dict.update(ckpt_dict)
simulator.load_state_dict(model_dict)


#simulator.load_state_dict(weights['model'])
simulator = simulator.to(device)
total_params = sum(p.numel() for p in simulator.parameters())
print(f"Number of parameters: {total_params}")


class RolloutDataset(pyg.data.Dataset):
    def __init__(self, data_path, split, window_length=7, random_sample=False):
        super().__init__()
        
        # load data from the disk
        self.data_path = data_path
        with open(os.path.join(data_path, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.metadata['default_connectivity_radius'] = 0.1050
        self.window_length = window_length
        self.random_sample = random_sample
        self.sampling_strategy = params.sampling_strategy
        dataset = torch.load(data_path+'/'+split)
        self.position = dataset['position']
        self.dim = self.position[0].shape[2]
    def len(self):
        return len(self.position)
    
    def get(self, idx):

        particle_type = []
        position_seq = torch.from_numpy(self.position[idx])
        position_seq = torch.permute(position_seq, dims=(1,0,2))
        

        
        if(self.random_sample):

            self.mesh_size = particle_type.shape[0]//128
            if(self.sampling_strategy == 'random'):
                self.points = sorted(random.sample(range(0, particle_type.shape[0]), self.mesh_size))
                particle_type = particle_type[self.points]
                position_seq = position_seq.permute(1,0,2)
                position_seq = position_seq[self.points]
                position_seq = position_seq.permute(1,0,2)

        data = {"particle_type": particle_type, "position": position_seq}
        return data

sim_id = 0
rollout_full = RolloutDataset(dataset_dir, "rollout_full.pt", random_sample=False)

gt_sim = rollout_full[sim_id]['position']
lossfn = torch.nn.MSELoss()

#rollout_out = torch.load('water_rollouts.pt') --> This line is used to load the outputs from time-stepper model. For the demo, we are using a random discretization of GT.
mesh_size =  3000
points = sorted(random.sample(range(0, gt_sim.shape[1]),mesh_size))
points = range(0, gt_sim.shape[1], 7)
rom_pred = gt_sim.permute(1, 0, 2)
rom_pred = rom_pred[points]
rom_pred = rom_pred.permute(1, 0, 2)

#rom_pred = rollout_out
print(gt_sim.shape)
print(rom_pred.shape)
fom_ic = gt_sim[0]
rom_ic = rom_pred[0]
fom_pred = []
total_loss = 0.0
i = 0
counter = 0
progress_bar = tqdm(range(rom_pred.shape[0]), desc=f"Datapoint {i}")
for i in progress_bar:

    cur_pos = rom_pred[i] #- rom_ic
    if(params.train_on_displacement==True):
        cur_pos = rom_pred[i] - rom_ic

    pred = simulator(rom_ic.to(device), fom_ic.to(device), cur_pos.to(device)).detach().cpu().numpy().squeeze(0)
    if(params.train_on_displacement == True):
        pred = pred + fom_ic.detach().cpu().numpy()
    fom_pred.append(pred)
    loss = lossfn(torch.from_numpy(pred), gt_sim[i])
    total_loss += loss.item()
    progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (i+1e-4)})
    counter+=1
    del cur_pos, pred, loss
avg_loss = total_loss/rom_pred.shape[0]
fom_pred = np.asarray(fom_pred)
print("AVG LOSS: ", avg_loss)
print(fom_pred.shape)
loss = (torch.from_numpy(fom_pred) - gt_sim) ** 2
loss = loss.sum(dim=-1).mean()
print("ROLLOUT LOSS: ", loss)

torch.save(fom_pred, f"fom_pred_{params.dataset}.pt")