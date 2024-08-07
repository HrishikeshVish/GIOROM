import torch
from dataloader import OneStepDataset, RolloutDataset

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from utils import oneStepMSE, rolloutMSE, visualize_graph
import os
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric as pyg
import argparse
from argparse import Namespace
import yaml

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
parser.add_argument("-m", "--model", help='Model', choices=['giorom2d','giorom2d_medium', 'giorom2d_large', 'giorom3d', 'giorom3d_large', 'egat'])
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
elif(params.model == 'giorom3d_large_030'):
    from models.giorom3d_large_030 import PhysicsEngine
elif(params.model == 'egat'):
    from Baselines.egat import PhysicsEngine
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

def train(params, optimizer, scheduler, ckpt, simulator, train_loader, dataset_size, valid_loader=None, valid_rollout_dataset=None, valid_dataset_metadata=None, plot_loss=True, save_weights=True, radius=None):
    loss_fn = torch.nn.MSELoss()
    # recording loss curve
    train_loss_list = []
    loss_list = []
    eval_loss_list = []
    eval_mse_list = []
    rollout_mses = []
    onestep_mse_list = []
    rollout_mse_list = []
    x_axis = []
    total_step = 0
    lowest_loss = 100000
    lowest_rollout_mse = 100000
    lowest_one_stop_eval = 100000
    lowest_avg_loss = 100000
    rollout_mse = 100000
    onestep_mse = 100000
    for i in range(ckpt, params.epoch):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1
            train_loss_list.append((total_step, loss.item()))
            if(batch_count%100 == 0):
                loss_list.append(total_loss/batch_count)
                #print("DATA Y SGAOE = ", len(train_dataset))
                x_axis.append((i*dataset_size//params.batch_size)+batch_count)


            # evaluation
            if total_step % params.eval_interval == 0:
                simulator.eval()
                
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, valid_dataset_metadata, params.noise)
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                eval_mse_list.append(onestep_mse)
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            # do rollout on valid set
            if total_step % params.rollout_interval == 0:
                simulator.eval()
                #valid_rollout_dataset = train_dataset
                rollout_mse = rolloutMSE(simulator, valid_rollout_dataset, params.noise, radius, graph_type=valid_rollout_dataset.graph_type)
                rollout_mse_list.append((total_step, rollout_mse))
                rollout_mses.append(rollout_mse)
                tqdm.write(f"\nEval: Rollout MSE: {rollout_mse}")
                simulator.train()

            # save model
            #if total_step % params["save_interval"] == 0:
            checkpoint_name = f'{params.model}_{params.dataset}.pt'
            if(params.sampling == True):
                checkpoint_name = f'{params.model}_{params.dataset}_sampled.pt'
            if(lowest_loss > loss.item() and save_weights):
                print(f'Loss improved from {lowest_loss} to {loss.item()} saving weights!')
                lowest_loss = loss.item()
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
            if(lowest_avg_loss > total_loss/batch_count):
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
            if(batch_count%100 == 0 and params.store_loss):
                loss_dict = {'train': loss_list,'eval':eval_mse_list,'rollout':rollout_mses}
                torch.save(loss_dict, f'logs/{params.model}_{params.dataset}_logs.pt')
            if(batch_count%100 == 0 and plot_loss):
                #x_axis = list(range(len(loss_list)))
                if(os.path.exists('logs')==False):
                    os.mkdir('logs')
                y_axis = loss_list
                fig = plt.figure()
                plt.plot(x_axis, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Train MSE Loss')
                fig.savefig(f'logs/{params.model}_{params.dataset}_train.png')
                plt.close(fig)

                y_axis = eval_mse_list
                xa = list(range(len(y_axis)))

                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Eval MSE Loss')
                fig.savefig(f'logs/{params.model}_{params.dataset}_eval.png')
                plt.close(fig)

                y_axis = rollout_mses
                xa = list(range(len(y_axis)))
                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Rollout Loss')
                fig.savefig(f'logs/{params.model}_{params.dataset}_rollout.png')
                plt.close(fig)
    return train_loss_list


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = OneStepDataset(dataset_dir, "train.pt", noise_std=params.noise, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=0.060)
    valid_dataset = OneStepDataset(dataset_dir, "test.pt", noise_std=params.noise, sampling_strategy=params.sampling, graph_type=params.graph_type,radius=0.060)
    rollout_dataset = RolloutDataset(dataset_dir, "rollout.pt", sampling_strategy=params.sampling, graph_type=params.graph_type,radius=0.060, mesh_size=170)[:2]
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

    if(params.visualize_graph):
        sample_dataset = OneStepDataset(dataset_dir, "train.pt", noise_std=params.noise, return_pos=True, sampling=params.sampling, sampling_strategy='random', graph_type=params.graph_type, radius=0.060)
        visualize_graph(sample_dataset)

    if(params.model_config is not None):
        simulator = PhysicsEngine(device, **model_config)
    else:
        simulator = PhysicsEngine(device)
    
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params.lr)
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
        simulator = simulator.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
        #ckpt = {}
        #ckpt['epoch'] = 0
        
    else:
        ckpt = {}
        ckpt['epoch'] = 0


    


    simulator = simulator.to(simulator.device)
    train_loss_list = train(params, optimizer, scheduler, ckpt['epoch'], simulator, train_loader, len(train_dataset), valid_loader, valid_rollout_dataset=rollout_dataset, 
                            valid_dataset_metadata=rollout_dataset.metadata, plot_loss=True,save_weights=True, radius=0.060)