import torch
from dataloader_nf import OneStepDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import oneStepMSE, rolloutMSE, visualize_graph
import os
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric as pyg
import argparse
from argparse import Namespace
import yaml
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--train_config", help="Train Config File Name")
parser.add_argument("-e", "--epoch", help='Epochs', default=1000, type=int)
parser.add_argument("-b", "--batch_size", help='Batch Size', default=4, type=int)
parser.add_argument("-lr", "--lr", help='Learning Rate', default=1e-4, type=float)
parser.add_argument("-g", "--gamma", help='gamma', default=0.1**(1/5e6), type=float)
parser.add_argument("-n", "--noise", help='Noise', default=3e-4, type=float)
parser.add_argument("-si", "--save_interval", help='Save Interval', default=1000, type=int)
parser.add_argument("-ei", "--eval_interval", help='Eval Interval', default=1500, type=int)
parser.add_argument("-ri", "--rollout_interval", help='Rollout Interval', default=1500, type=int)
parser.add_argument("-rs", "--sampling", help='Sampling', default=False, action='store_true')
parser.add_argument("-ss", "--sampling_strategy", help='Sampling', default='random', choices=['random', 'fps'])
parser.add_argument("-gt", "--graph_type", help='Graph Type', default='radius', choices=['radius', 'delaunay'])
parser.add_argument("-ct", "--connectivity_radius", help='Graph Connectivity Radius', default=0.015, type=float)
parser.add_argument("-m", "--model", help='Model', choices=['FOM_GNO'])
parser.add_argument("-d", "--dataset", help='Dataset')
parser.add_argument("-lc", "--load_checkpoint", help='Load Checkpoint', default=False, action='store_true')
parser.add_argument("-viz", "--visualize_graph", help='Visualize Graph', default=False, action='store_true')
parser.add_argument("-logs", "--store_loss", help='Store Logs', default=True, action='store_true')
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




def oneStepMSE(simulator, dataloader):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        #scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2).cuda()
        for data in dataloader:
            fom_ic, rom_ic, rom_f, fom_gt = data
            fom_ic = fom_ic.cuda()
            rom_ic = rom_ic.cuda()
            rom_f = rom_f.cuda()
            fom_gt = fom_gt.cuda()
            pred = simulator(rom_ic, fom_ic, rom_f)
            pred_pos = pred #+ fom_ic
            gt_pos = fom_gt #+ fom_ic
            mse = ((pred_pos - gt_pos)) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred_pos - gt_pos) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count

def train(params, optimizer, scheduler, ckpt, simulator, train_loader, dataset_size, save_weights):
    loss_fn = torch.nn.MSELoss()
    train_loss_list = []
    loss_list = []
    total_step = 0
    lowest_loss = 100000
    lowest_rollout_mse = 100000
    lowest_one_stop_eval = 100000
    lowest_avg_loss = 100000
    rollout_mse = 100000
    onestep_mse = 100000
    simulator = simulator.to(simulator.device)
    predictions = {}
    groundtruths = {}
    for i in range(ckpt, params.epoch):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            fom_ic, rom_ic, rom_f, fom_gt, time_step = data
            time_step = int(time_step.detach().cpu().numpy())
            #print(time_step)
            fom_ic = fom_ic.to(simulator.device)
            rom_ic = rom_ic.to(simulator.device)
            rom_f = rom_f.to(simulator.device)
            fom_gt = fom_gt.to(simulator.device)

            #print(rom_ic.shape)
            pred = simulator(rom_ic, fom_ic, rom_f) 
            pred = pred * time_step * 7.5e-3 + fom_ic
            #print(pred)
            #print(fom_gt)
            
            if(i == 3):
                if(time_step not in predictions.keys()):
                    predictions[time_step] = []
                if(time_step not in groundtruths.keys()):
                    groundtruths[time_step] = []
                predictions[time_step].append(pred.detach().cpu().squeeze(0))
                groundtruths[time_step].append(fom_gt.detach().cpu().squeeze(0))
                print(time_step)
            # if(params.train_on_displacement):
            #     pred = pred+ fom_ic
            #     fom_gt = fom_gt + fom_ic
            loss = loss_fn(pred, fom_gt) + loss_fn(pred - fom_ic, fom_gt - fom_ic)
            del fom_gt, pred, fom_ic, rom_f, rom_ic
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            ls = loss.item()
            del loss

            batch_count += 1
            progress_bar.set_postfix({"loss": ls, "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1
            train_loss_list.append((total_step, ls))
            if(batch_count%100 == 0):
                    loss_list.append(total_loss/batch_count)
            checkpoint_name = f'{params.model}_{params.dataset}.pt'
            if(lowest_loss > ls and save_weights and ls >1e-7 and i>1):
                print(f'Loss improved from {lowest_loss} to {ls} saving weights!')
                print("checkpoint Directory: ", checkpoint_directory, checkpoint_name)
                lowest_loss = ls
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":i
                    },
                    os.path.join(checkpoint_directory, checkpoint_name)
                )
            if(lowest_rollout_mse > rollout_mse and save_weights):
                print(f'Rollout Loss improved from {lowest_rollout_mse} to {rollout_mse} saving weights!')
                lowest_rollout_mse = rollout_mse
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":i
                    },
                    os.path.join(checkpoint_directory, checkpoint_name)
                )
            if(lowest_one_stop_eval > onestep_mse and save_weights):
                print(f'One Step Loss improved from {lowest_one_stop_eval} to {onestep_mse} saving weights!')
                lowest_one_stop_eval = onestep_mse
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":i
                    },
                    os.path.join(checkpoint_directory, checkpoint_name)
                )
            if(lowest_avg_loss > total_loss/batch_count and save_weights and total_loss/batch_count > 1e-4 and i>1):
                print(f'Average Train Loss improved from {lowest_avg_loss} to {total_loss/batch_count} saving weights!')
                lowest_avg_loss = total_loss/batch_count
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":i
                    },
                    os.path.join(checkpoint_directory, checkpoint_name)
                )
        if(i == 3):
            for key in predictions.keys():
                print(key)
                predictions[key] = torch.cat(predictions[key], dim=0)
                print(predictions[key].shape)
                groundtruths[key] = torch.cat(groundtruths[key], dim=0)
                print(groundtruths[key].shape)
            torch.save(predictions, 'predictions.pt')
            torch.save(groundtruths, 'groundtruths.pt')
            exit()
    return train_loss_list

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #train_dataset = OneStepDataset(dataset_dir, "rollout_full.pt", noise_std=params.noise, sampling=params.sampling, sampling_strategy=params.sampling_strategy, sample_mesh_size=params.sample_mesh_size, train_on_displacement=params.train_on_displacement)
    train_dataset = OneStepDataset('sequence.pt', noise_std=params.noise, sampling=params.sampling, sampling_strategy=params.sampling_strategy, sample_mesh_size=700, train_on_displacement=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    if(params.model_config is not None):
        simulator = PhysicsEngine(device, **model_config)
    else:
        simulator = PhysicsEngine(device)
    optimizer = torch.optim.Adamax(simulator.parameters(), lr=params.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=eval(params.gamma))
    
    total_params = sum(p.numel() for p in simulator.parameters())
    print(f"Number of parameters: {total_params}")
    if(params.load_checkpoint):
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
            mismatch = True
        model_dict.update(ckpt_dict)
        simulator.load_state_dict(model_dict)
        simulator = simulator.to(simulator.device)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        
    else:
        ckpt = {}
        ckpt['epoch'] = 0
    simulator = simulator.to(simulator.device)
    train_loss_list = train(params, optimizer, scheduler, ckpt['epoch'], simulator, train_loader, len(train_dataset),save_weights=params.save_weights)
