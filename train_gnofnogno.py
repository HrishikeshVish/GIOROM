import torch
from dataloader import OneStepDataset, RolloutDataset
from Baselines.gnofnogno import PhysicsEngine
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from utils import oneStepMSE, rolloutMSE, visualize_graph
import os
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric as pyg
import numpy as np

params = {
    "epoch": 1000,
    "batch_size": 4,
    "lr": 1e-4,
    "noise": 6e-4,
    "save_interval": 1000,
    "eval_interval": 2000,
    "rollout_interval": 4000,
    "model": "GoogleGNN",
    "dataset": "Plasticine",
    "load_checkpoint":False,
    "visualize_graph":True,
    "store_loss": True
}

def train(params, optimizer, scheduler, ckpt, simulator, train_loader, dataset_size, valid_loader=None, valid_rollout_dataset=None, valid_dataset_metadata=None, plot_loss=True, save_weights=True):
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
    highest_loss = 100000
    for i in range(ckpt, params["epoch"]):
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
                x_axis.append((i*dataset_size//params["batch_size"])+batch_count)


            # evaluation
            if total_step % params["eval_interval"] == 0:
                simulator.eval()
                
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, valid_dataset_metadata, params["noise"])
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                eval_mse_list.append(onestep_mse)
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            # do rollout on valid set
            if total_step % params["rollout_interval"] == 0:
                simulator.eval()
                #valid_rollout_dataset = train_dataset
                rollout_mse = rolloutMSE(simulator, valid_rollout_dataset, params['noise'])
                rollout_mse_list.append((total_step, rollout_mse))
                rollout_mses.append(rollout_mse)
                tqdm.write(f"\nEval: Rollout MSE: {rollout_mse}")
                simulator.train()

            # save model
            #if total_step % params["save_interval"] == 0:
            if(highest_loss > loss.item() and save_weights):
                print(f'Loss improved from {highest_loss} to {loss.item()} saving weights!')
                highest_loss = loss.item()
                # torch.save(
                #    {
                #        "model": simulator.state_dict(),
                #        "optimizer": optimizer.state_dict(),
                #        "scheduler": scheduler.state_dict(),
                #        "epoch":i
                #    },
                #    os.path.join('/home/csuser/Documents/Neural Operator', f'{params["model"]}_{params["dataset"]}.pt')
                # )
            if(batch_count%100 == 0 and params['store_loss']):
                loss_dict = {'train': loss_list,'eval':eval_mse_list,'rollout':rollout_mses}
                torch.save(loss_dict, f'logs/{params["model"]}_{params["dataset"]}_logs.pt')
            if(batch_count%100 == 0 and plot_loss):
                #x_axis = list(range(len(loss_list)))
                if(os.path.exists('logs')==False):
                    os.mkdir('logs')
                y_axis = loss_list
                fig = plt.figure()
                plt.plot(x_axis, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Train MSE Loss')
                fig.savefig(f'logs/{params["model"]}_{params["dataset"]}_train.png')
                plt.close(fig)

                y_axis = eval_mse_list
                xa = list(range(len(y_axis)))

                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Eval MSE Loss')
                fig.savefig(f'logs/{params["model"]}_{params["dataset"]}_eval.png')
                plt.close(fig)

                y_axis = rollout_mses
                xa = list(range(len(y_axis)))
                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Rollout Loss')
                fig.savefig(f'logs/{params["model"]}_{params["dataset"]}_rollout.png')
                plt.close(fig)
    return train_loss_list


if __name__ == '__main__':
    

    train_dataset = OneStepDataset('/home/csuser/Documents/new_dataset/nclaw_geometries/', "train.obj", noise_std=params["noise"], random_sampling=False)
    valid_dataset = OneStepDataset('/home/csuser/Documents/new_dataset/nclaw_geometries/', "test.pt", noise_std=params["noise"], random_sampling=False)
    rollout_dataset = RolloutDataset('/home/csuser/Documents/new_dataset/nclaw_geometries/', "rollout.pt", random_sample=False)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False)

    if(params['visualize_graph']):
        sample_dataset = OneStepDataset('/home/csuser/Documents/new_dataset/nclaw_geometries/', "test.pt", noise_std=params["noise"], return_pos=True, random_sampling=False)
        visualize_graph(sample_dataset)
    device = 'cuda'
    noise_std = 3e-4
    normalization_stats = {
    'acceleration': {
        'mean':torch.FloatTensor(train_dataset.metadata['acc_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(train_dataset.metadata['acc_std'])**2 + noise_std**2).to(device),
    }, 
    'velocity': {
        'mean':torch.FloatTensor(train_dataset.metadata['vel_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(train_dataset.metadata['vel_std'])**2 + noise_std**2).to(device),
    }, 
    }
    simulator = PhysicsEngine(particle_dimension=3, node_in=37,
        edge_in=4,
        latent_dim=128,
        num_message_passing_steps=10,
        mlp_num_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=0.015,
        boundaries=np.array(train_dataset.metadata['bounds']),
        normalization_stats=normalization_stats,
        num_particle_types=9,
        particle_type_embedding_size=16,)
    
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))
    
    total_params = sum(p.numel() for p in simulator.parameters())
    print(f"Number of parameters: {total_params}")
    
    if(params["load_checkpoint"]):
        ckpt = torch.load('EGATNO_plasticine.pt')
        weights = torch.load('EGATNO_plasticine.pt')['model']

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

        #optimizer.load_state_dict(ckpt['optimizer'])
        #scheduler.load_state_dict(ckpt['scheduler'])
    
    else:
        ckpt = {}
        ckpt['epoch'] = 0


    


    simulator = simulator.cuda()
    train_loss_list = train(params, optimizer, scheduler, ckpt['epoch'], simulator, train_loader, len(train_dataset), valid_loader, valid_rollout_dataset=rollout_dataset, 
                            valid_dataset_metadata=rollout_dataset.metadata, plot_loss=True,save_weights=False)