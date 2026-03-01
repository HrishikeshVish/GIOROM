# %%

from models.config import TimeStepperConfig
import torch
from data import OneStepDataset, RolloutDataset
from huggingface_hub import hf_hub_download, snapshot_download
import random
from random import randint
from models.giorom3d_T import PhysicsEngine
from utils.utils import oneStepMSE, rolloutMSE, visualize_graph
import yaml
from graph_utils import compute_spectral_metrics
#from Baselines.GAT import PhysicsEngine

# %%
from argparse import Namespace
with open('configs/configs_nclaw_Sand_T.yaml', 'r') as f:
        params = yaml.full_load(f)
params = Namespace(**params)

# %%
import os
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
time_stepper_config = TimeStepperConfig(**model_config)

# %%
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

# %%
import torch_geometric as pyg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = OneStepDataset(train_dir, metadata_dir, noise_std=params.noise, sampling_strategy=params.sampling_strategy, graph_type=params.graph_type,radius=params.connectivity_radius)
valid_dataset = OneStepDataset(test_dir, metadata_dir, noise_std=params.noise, sampling_strategy=params.sampling_strategy, graph_type=params.graph_type,radius=params.connectivity_radius)
rollout_dataset = RolloutDataset(rollout_dir, metadata_dir, sampling_strategy=params.sampling_strategy, graph_type=params.graph_type,radius=params.connectivity_radius, mesh_size=170, sampling=False)
rollout_full = RolloutDataset(rollout_full_dir, metadata_dir, sampling_strategy=params.sampling_strategy, graph_type=params.graph_type, radius=0.145, mesh_size=170)
train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

# %%
checkpoint_directory = os.path.join(os.getcwd(), 'saved_models')
if(os.path.exists(checkpoint_directory) == False):
    os.mkdir(checkpoint_directory)
if(params.load_checkpoint == True and params.load_huggingface == False):
    if(params.ckpt_name is None):
        raise Exception("No checkpoint Name specified")
    checkpoint = os.path.join(checkpoint_directory, params.ckpt_name)
    if(os.path.exists(checkpoint)==False and params.load_huggingface == False):
        raise Exception("Invalid Checkpoint Directory")
if(params.load_huggingface == True):
    checkpoint = params.ckpt_name
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
        #optimizer_checkpoint = torch.load(checkpoint+'/optimizer.pt')
        #optimizer.load_state_dict(optimizer_checkpoint)
        #scheduler_checkpoint = torch.load(checkpoint+'/scheduler.pt')
        #scheduler.load_state_dict(scheduler_checkpoint)
    print("Loaded Checkpoint")

# %% [markdown]
# 

# %%
from utils.data_utils import preprocess
def rollout(model, data, metadata, noise_std, radius=None, data_full=None):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.config.window_size + 1
    total_time = data["position"].size(0)
    #total_time = 400
    #print("Total Time = ", total_time)
    
    traj = data["position"][:window_size]
    #print("TRAJ SHAPE = ", traj.shape)
    traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]
    
    traj_full = data_full["position"][:window_size] if data_full is not None else None
    if traj_full is not None:
        traj_full = traj_full.permute(1, 0, 2)
        particle_type_full = data_full["particle_type"]

    radius = radius if radius is not None else metadata["default_connectivity_radius"]
    for time in range(total_time - window_size):
        print(time)
        with torch.no_grad():
            #print("PARTICLE TYPE = ", particle_type.shape)
            #print("TRAJECTORY = ", traj.shape)
            
            graph = preprocess(particle_type, traj[:, -window_size:], 0.045, metadata, 0.0, radius=radius)
            if(time == 0 and data_full is not None):
                graph_full = preprocess(particle_type_full, traj_full[:, -window_size:], 0.045, metadata, 0.0, radius=0.105)
                metrics = compute_spectral_metrics(graph_full, graph)  # Compute spectral metrics for the initial graph
                print("Spectral Metrics: ", metrics)
            
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)
    return traj

# %%
#rollout_dataset = RolloutDataset(rollout_dir, metadata_dir, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=params.connectivity_radius, mesh_size=170)
#rollout_dataset_gt = RolloutDataset(rollout_dir, metadata_dir, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=params.connectivity_radius, mesh_size=170)
#print(len(rollout_dataset))
rollout_dataset_rom = rollout_dataset
rollout_dataset_full = rollout_dataset # rollout_full

simulator.eval()
sim_id = 0
rollout_data = rollout_dataset_rom[sim_id]
if(rollout_data['position'].shape[1] != rollout_data['particle_type'].shape[0]):
    temp = rollout_data['position']
    temp = temp.permute(1, 0, 2)
    temp = temp[:rollout_data['particle_type'].shape[0]]
    temp = temp.permute(1, 0, 2)
    rollout_data['position'] = temp
print(rollout_data['position'].shape)
print(rollout_data['particle_type'].shape)

#rollout_data_gt = rollout_dataset_gt[1]
rollout_data_full = rollout_dataset_full[sim_id]
print("ROLLOUT ROM SHAPE: ", rollout_data['position'].shape)
#rollout_data_full = rollout_full[sim_id]
#print(rollout_data_full['position'].shape)
temp = rollout_data['position'][0]



rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params.noise, radius=0.105, data_full=rollout_data_full)
rollout_out = rollout_out.permute(1, 0, 2)
loss = (rollout_out - rollout_data["position"]) ** 2
loss = loss.sum(dim=-1).mean()
print("Rollout Loss: ", loss)
#torch.save(rollout_out, f'outputs/{params["model"]}_{params["dataset"]}_{sim_id}.pt')
#torch.save(rollout_data_full, f'outputs/{params["model"]}_{params["dataset"]}_{sim_id}_gt.pt')

# %%
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# %%
import numpy as np
TYPE_TO_COLOR = {
    3: "black",
    0: "green",
    7: "magenta",
    6: "gold",
    5: "blue",
}


def visualize_prepare(ax, particle_type, position, metadata):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata):
    print(position_pred.shape)
    print(position_gt.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    old_particle_type = torch.ones(size=(position_gt.shape[1],)) * particle_type[0]
    plot_info = [
        visualize_prepare(axes[0], old_particle_type, position_gt, metadata),
        visualize_prepare(axes[1], particle_type, position_pred, metadata),
    ]
    axes[0].set_title("Ground truth")
    axes[1].set_title("Prediction")

    plt.close()
    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for type_, line in points.items():
                mask = particle_type == type_
                if(position.shape[1] == position_gt.shape[1]):
                    mask = old_particle_type == type_
                    #print(position.shape, mask.shape)
                #print(position.shape, mask.shape)
                line.set_data(position[step_i, mask, 0], position[step_i, mask, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(fig, update, frames=np.arange(0, position_gt.size(0)), interval=20, blit=True)

# %% [markdown]
# 

# %%

#inp = torch.load('out.pt')
#anim = visualize_pair(inp['pt'], inp['rout'], inp['pos'], inp['met'])
#anim = visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data["position"], rollout_dataset.metadata)
# import numpy as np
# anim = visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data_gt['position'], rollout_dataset.metadata)
# HTML(anim.to_html5_video())


