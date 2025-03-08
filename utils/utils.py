import torch
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric as pyg
from data import preprocess
from dgl.geometry import farthest_point_sampler

def visualize_graph(dataset_sample):
    graph, position = dataset_sample[0]
    print(f"The first item in the valid set is a graph: {graph}")
    print(f"This graph has {graph.num_nodes} nodes and {graph.num_edges} edges.")
    print(f"Each node is a particle and each edge is the interaction between two particles.")
    print(f"Each node has {graph.num_node_features} categorial feature (Data.x), which represents the type of the node.")
    print(f"Each node has a {graph.pos.size(1)}-dim feature vector (Data.pos), which represents the positions and velocities of the particle (node) in several frames.")
    print(f"Each edge has a {graph.num_edge_features}-dim feature vector (Data.edge_attr), which represents the relative distance and displacement between particles.")
    print(f"The model is expected to predict a {graph.y.size(1)}-dim vector for each node (Data.y), which represents the acceleration of the particle.")

    # remove directions of edges, because it is a symmetric directed graph.
    nx_graph = pyg.utils.to_networkx(graph).to_undirected()
    # remove self loops, because every node has a self loop.
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    plt.figure(figsize=(7, 7))
    print(position.shape)
    if(graph.y.size(1) !=2):
        plt.scatter(position.T[0], position.T[1], position.T[2])
    else:
        nx.draw(nx_graph, pos={i: tuple(v) for i, v in enumerate(position)}, node_size=50)
    plt.savefig('graph.png')



def rollout(model, data, metadata, noise_std, radius=None, graph_type='radius'):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.config.window_size + 1
    total_time = data["position"].size(0)
    #print("Total Time = ", total_time)
    
    traj = data["position"][:window_size]
    #print("TRAJ SHAPE = ", traj.shape)
    traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]


    for time in range(total_time - window_size):
        with torch.no_grad():
            graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0, radius=radius, graph_type=graph_type)
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)
    return traj

def oneStepMSE(simulator, dataloader, metadata, noise):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2).cuda()
        for data in dataloader:
            data = data.cuda()
            pred = simulator(data)
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count

def rolloutMSE(simulator, dataset, noise):
    total_loss = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        for rollout_data in dataset:
            rollout_out = rollout(simulator, rollout_data, dataset.metadata, noise, dataset.radius, dataset.graph_type)
            rollout_out = rollout_out.permute(1, 0, 2)
            loss = (rollout_out - rollout_data["position"]) ** 2
            loss = loss.sum(dim=-1).mean()
            #print("ROLLOUT LOSS = ", loss.item())
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count