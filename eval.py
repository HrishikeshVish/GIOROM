import math
import torch_scatter
import torch
import torch_geometric as pyg
import torch.nn.functional as F
import os
import numpy as np
import json
from neuralop.models import FNO
from torch import nn
from models.neuraloperator.neuralop.layers.mlp import MLP as NeuralOpMLP
from models.neuraloperator.neuralop.layers.embeddings import PositionalEmbedding
from models.neuraloperator.neuralop.layers.integral_transform import IntegralTransform
from models.neuraloperator.neuralop.layers.neighbor_search import NeighborSearch
import random
from random import randint
from utils import rollout
from dataloader import RolloutDataset
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import argparse
from argparse import Namespace
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--eval_config", help="Eval Config File Name")
parser.add_argument("-n", "--noise", help='Noise', default=3e-4, type=float)
parser.add_argument("-m", "--model", help='Model', choices=['giorom2d','giorom2d_medium', 'giorom2d_large', 'giorom3d', 'giorom3d_large'])
parser.add_argument("-d", "--dataset", help='Dataset')
parser.add_argument("-lc", "--load_checkpoint", help='Load Checkpoint', default=True, action='store_true')
parser.add_argument("-viz", "--visualize_graph", help='Visualize Results', default=False, action='store_true')
parser.add_argument("-drt", "--dataset_rootdir", help='Dataset Rootdir')
parser.add_argument("-ckpt", "--ckpt_name", help='Checkpoint Name (Usually model name underscore datasetname)')
parser.add_argument("-mc", "--model_config", help='Model configs (usually modelname.yaml)')
parser.add_argument("-sr", "--save_rollout", help='Save Results')


params = parser.parse_args()
if(params.eval_config is not None):
    if(params.eval_config.endswith('.yaml') == False):
        params.eval_config += '.yaml'
    config_path = os.path.join(os.getcwd(), 'configs', params.eval_config)
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
elif(params.model == 'giorom2d'):
    from models.giorom2d import PhysicsEngine
elif(params.model == 'giorom2d_medium'):
    from models.giorom2d_medium import PhysicsEngine
elif(params.model == 'giorom2d_large'):
    from models.giorom2d_large import PhysicsEngine
elif(params.model == 'giorom3d'):
    from models.giorom3d import PhysicsEngine
elif(params.model == 'giorom3d_large'):
    from models.giorom3d_large import PhysicsEngine
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

def eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = PhysicsEngine(device, **model_config)
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
    simulator = simulator.cuda()

    rollout_dataset = RolloutDataset(dataset_dir, "rollout.pt", random_sample=False)
    rollout_dataset_gt = RolloutDataset(dataset_dir, "rollout_gt.pt", random_sample=False)
    simulator.eval()

    rollout_data = rollout_dataset[params.dataset_idx]
    rollout_data_gt = rollout_dataset_gt[params.dataset_idx]

    rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params.noise)
    rollout_out = rollout_out.permute(1, 0, 2)
    if(params.save_rollout):
        torch.save(rollout_out, f'{params.dataset}_{params.dataset_idx}_rollout.pt')
    
    if(params.visualize_graph):
        visualize(rollout_data, rollout_out, rollout_data_gt, rollout_dataset)


def visualize(rollout_data, rollout_out, rollout_data_gt, rollout_dataset):
    TYPE_TO_COLOR = {
    3: "black",
    0: "green",
    7: "magenta",
    6: "gold",
    5: "blue",
    }
    anim = visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data_gt['position'], rollout_dataset.metadata, TYPE_TO_COLOR)
    HTML(anim.to_html5_video())
    writer = animation.writers['ffmpeg'](fps=30)
    anim.save('disc_4.mp4',writer=writer,dpi=200)
    HTML(anim.to_html5_video())


def visualize_prepare(ax, particle_type, position, metadata, TYPE_TO_COLOR):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata, TYPE_TO_COLOR):
    print(position_pred.shape)
    print(position_gt.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    old_particle_type = torch.ones(size=(position_gt.shape[1],)) * particle_type[0]
    plot_info = [
        visualize_prepare(axes[0], old_particle_type, position_gt, metadata, TYPE_TO_COLOR),
        visualize_prepare(axes[1], particle_type, position_pred, metadata, TYPE_TO_COLOR),
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


if __name__ == '__main__':
    eval()