"""
Main script for training and evaluating physics-based machine learning models on material simulation datasets.

This script sets up argument parsing, loads configuration files, initializes datasets, and trains models using
a physics-informed machine learning approach. The models are designed for reduced-order modeling (ROM) of
material simulations and leverage graph-based architectures.

### Features:
- Parses command-line arguments for training and model configurations.
- Loads training and model configurations from YAML files.
- Downloads datasets from Hugging Face Hub.
- Supports multiple material simulation datasets in 2D and 3D.
- Constructs graph-based representations of the datasets.
- Initializes different model architectures including GNNs and transformer-based models.
- Implements training with an adaptive optimizer and scheduler.
- Supports model checkpointing and logging.
- Provides visualization options for graph structures.

### Usage:
Run the script with command-line arguments:
"""

import torch
from data import OneStepDataset, RolloutDataset
from huggingface_hub import hf_hub_download, snapshot_download
import os
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from utils.utils import oneStepMSE, rolloutMSE, visualize_graph
import os
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric as pyg
import argparse
from argparse import Namespace
import yaml
from training.trainer import ROMTrainer
#from training.train import train
from transformers import TrainingArguments
from training.train import train

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config File Name")
parser.add_argument("-e", "--epoch", help='Epochs', default=1000, type=int)
parser.add_argument("-b", "--batch_size", help='Batch Size', default=4, type=int)
parser.add_argument("-lr", "--lr", help='Learning Rate', default=1e-4, type=float)
parser.add_argument("-g", "--gamma", help='gamma', default=0.1**(1/5e6), type=float)
parser.add_argument("-n", "--noise", help='Noise', default=3e-4, type=float)
parser.add_argument("-si", "--save_interval", help='Save Interval', default=1000, type=int)
parser.add_argument("-ei", "--eval_interval", help='Eval Interval', default=1500, type=int)
parser.add_argument("-li", "--log_interval", help='Log Interval', default=1500, type=int)
parser.add_argument("-ri", "--rollout_interval", help='Rollout Interval', default=1500, type=int)
parser.add_argument("-rs", "--sampling", help='Sampling', default=False, action='store_true')
parser.add_argument("-ss", "--sampling_strategy", help='Sampling', default='random', choices=['random', 'fps'])
parser.add_argument("-gt", "--graph_type", help='Graph Type', default='radius', choices=['radius', 'delaunay'])
parser.add_argument("-ct", "--connectivity_radius", help='Graph Connectivity Radius', default=0.015, type=float)
parser.add_argument("-wd", "--weight_decay", help='Weight Decay', default=1e-6, type=float)
parser.add_argument("-m", "--model", help='Model', choices=['giorom2d','giorom2d_medium', 'giorom2d_large', 'giorom3d', 'giorom3d_large', 'egat'])
parser.add_argument("-d", "--dataset", help='Dataset')
parser.add_argument("-lc", "--load_checkpoint", help='Load Checkpoint', default=False, action='store_true')
parser.add_argument("-viz", "--visualize_graph", help='Visualize Graph', default=False, action='store_true')
parser.add_argument("-logs", "--store_loss", help='Store Logs', default=True, action='store_true')
parser.add_argument("-drt", "--dataset_rootdir", help='Dataset Rootdir')
parser.add_argument("-ckpt", "--ckpt_name", help='Checkpoint Name (Usually model name underscore datasetname)')
parser.add_argument("-mc", "--model_config", help='Model configs (usually modelname.yaml)')
parser.add_argument("-it" ,"--is_train", help='Is Train', action='store_true', default=True)

params = parser.parse_args()
if(params.config is not None):
    if(params.config.endswith('.yaml') == False):
        params.config += '.yaml'
    config_path = os.path.join(os.getcwd(), 'configs', params.config)
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
else:
    raise Exception("Please provide a Model Config")
    

print(model_config)
from models.config import TimeStepperConfig
time_stepper_config = TimeStepperConfig(**model_config)

print(params)

if(params.model is None):
    raise Exception("Model not specified")
elif(params.model == 'giorom3d_T'):
    from models.giorom3d_T import PhysicsEngine
elif(params.model == 'giorom2d_T'):
    from models.giorom2d_T import PhysicsEngine
else:
    raise Exception("Invalid model name")

if(params.dataset is None):
    raise Exception("Dataset not specified")
if(params.dataset_rootdir is None):
    raise Exception("Dataset rootdir not specified")

print("...Loading Dataset...")
materials = {"Water2D":"WaterDrop2DSmall", "Water3D":"Water3DNCLAWSmall", 
             "Water3D_long":"Water3DNCLAWSmall_longer_duration", "Sand2D":"Sand2DSmall", 
             "Sand3D":"Sand3DNCLAWSmall", "Sand3D_long":"Sand3DNCLAWSmall_longer_duration", 
             "MultiMaterial2D":"MultiMaterial2DSmall", "Plasticine3D":"Plasticine3DNCLAWSmall", 
             "Elasticity3D":"Elasticity3DSmall", "Jelly3D":"Jelly3DNCLAWSmall", "RigidCollision3D":"RigidCollision3DNCLAWSmall", 
             "Melting3D":"Melting3DSampleSeq"}

if(params.dataset in materials.keys()):
    if('2D' in params.dataset):
        files = ['train.pt', 'test.pt', 'rollout.pt', 'metadata.json']
        train_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[0]), cache_dir="./dataset_mpmverse")
        test_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[1]), cache_dir="./dataset_mpmverse")
        rollout_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[2]), cache_dir="./dataset_mpmverse")
        metadata_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[3]), cache_dir="./dataset_mpmverse")
    else:
        files = ['train.obj', 'test.pt', 'rollout.pt', 'metadata.json', 'rollout_full.pt']
        train_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[0]), cache_dir="./dataset_mpmverse")
        test_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[1]), cache_dir="./dataset_mpmverse")
        rollout_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[2]), cache_dir="./dataset_mpmverse")
        metadata_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[3]), cache_dir="./dataset_mpmverse")
        rollout_full_dir = hf_hub_download(repo_id=params.dataset_rootdir, repo_type='dataset', filename=os.path.join(materials[params.dataset], files[4]), cache_dir="./dataset_mpmverse")
else:
    raise Exception("Dataset Name Invalid")
checkpoint_directory = os.path.join(os.getcwd(), 'saved_models')
if(os.path.exists(checkpoint_directory) == False):
    os.mkdir(checkpoint_directory)
if(params.load_checkpoint == True):
    if(params.ckpt_name is None):
        raise Exception("No checkpoint Name specified")
    checkpoint = os.path.join(checkpoint_directory, params.ckpt_name)
    if(os.path.exists(checkpoint)==False):
        raise Exception("Invalid Checkpoint Directory")

log_directory = os.path.join(os.getcwd(), 'logs')
if(os.path.exists(log_directory) == False):
    os.mkdir(log_directory)

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = OneStepDataset(train_dir, metadata_dir, noise_std=params.noise, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=params.connectivity_radius)
    valid_dataset = OneStepDataset(test_dir, metadata_dir, noise_std=params.noise, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=params.connectivity_radius)
    rollout_dataset = RolloutDataset(rollout_dir, metadata_dir, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=params.connectivity_radius, mesh_size=170)[2:]
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

    if(params.visualize_graph):
        sample_dataset = OneStepDataset(train_dir, metadata_dir, noise_std=params.noise, return_pos=True, sampling=params.sampling, sampling_strategy='random', graph_type=params.graph_type, radius=params.connectivity_radius)
        visualize_graph(sample_dataset)

    simulator = PhysicsEngine(time_stepper_config)
    
    optimizer = torch.optim.Adamax(simulator.parameters(), lr=params.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=eval(params.gamma))
    
    total_params = sum(p.numel() for p in simulator.parameters())
    print(f"Number of parameters: {total_params}")
    
    if(params.load_checkpoint):
        if('.pt' in checkpoint):
            ckpt = torch.load(checkpoint)
            weights = torch.load(checkpoint)['model']

            model_dict = simulator.state_dict()
            ckpt_dict = {}
            
            
            model_dict = dict(model_dict)

            for k, v in weights.items():
                k2 = k[0:]
                
                if k2 in model_dict:
                    
                    if model_dict[k2].size() == v.size():
                        ckpt_dict[k2] = v
                    else:
                        print("Size mismatch while loading! %s != %s Skipping %s..."%(str(model_dict[k2].size()), str(v.size()), k2))
                        mismatch = True
                else:
                    print("Model Dict not in Saved Dict! %s != %s Skipping %s..."%(2, str(v.size()), k2))
                    mismatch = True
            if len(simulator.state_dict().keys()) > len(ckpt_dict.keys()):
                print("SIZE MISMATCH")
                mismatch = True
            model_dict.update(ckpt_dict)
            simulator.load_state_dict(model_dict)
            simulator = simulator.to(device)
        else:
            model_config = time_stepper_config.from_pretrained(checkpoint)
            simulator = simulator.from_pretrained(checkpoint, config=model_config)
            simulator = simulator.to(device)
            # optimizer_checkpoint = torch.load(checkpoint+'/optimizer.pt')
            # optimizer.load_state_dict(optimizer_checkpoint)
            # scheduler_checkpoint = torch.load(checkpoint+'/scheduler.pt')
            # scheduler.load_state_dict(scheduler_checkpoint)
        print("Loaded Checkpoint")
        
    else:
        ckpt = {}
        ckpt['epoch'] = 0


    


    simulator = simulator.to(simulator._device)
    if(params.is_train):
        print("...Training...")
        train(params, optimizer, scheduler, ROMTrainer, simulator, train_loader, valid_loader, train_dataset.metadata, 
                            rollout_dataset, oneStepMSE, rolloutMSE)