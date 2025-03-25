import numpy as np
import torch
import random
from scipy.spatial import Delaunay
import torch_geometric as pyg
def less_first(a, b):
    return [a,b] if a < b else [b,a]

def delaunay2edges(tri):

    list_of_edges = []

    for triangle in tri.simplices:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

    array_of_edges = np.unique(list_of_edges, axis=0) # remove duplicates

    return array_of_edges

def generate_noise(position_seq, noise_std):
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise

def preprocess(particle_type, position_seq, target_position, metadata, noise_std, radius=None, graph_type='radius'):
    """Preprocess a trajectory and construct the graph."""
    # Apply noise to the trajectory
    position_noise = generate_noise(position_seq, noise_std)
    noisy_position_seq = position_seq + position_noise

    # Compute recent positions and velocities
    recent_position = noisy_position_seq[:, -1]
    velocity_seq = noisy_position_seq[:, 1:] - noisy_position_seq[:, :-1]

    # Construct the graph based on particle distances
    n_particle = recent_position.size(0)
    
    if graph_type == 'delaunay':
        triangulation = Delaunay(recent_position.numpy())
        edges = delaunay2edges(triangulation)
        edge_index = torch.from_numpy(edges.astype(np.int64)).T
    elif graph_type == 'radius':
        connectivity_radius = radius if radius is not None else metadata["default_connectivity_radius"]
        edge_index = pyg.nn.radius_graph(recent_position, connectivity_radius, loop=True, max_num_neighbors=n_particle)

    # Compute node features: normalized velocity and distance to boundaries
    vel_mean = torch.tensor(metadata["vel_mean"])
    vel_std = torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    normal_velocity_seq = (velocity_seq - vel_mean) / vel_std

    boundary = torch.tensor(metadata["bounds"])
    distance_to_boundary = torch.cat([
        recent_position - boundary[:, 0], 
        boundary[:, 1] - recent_position
    ], dim=-1)

    connectivity_radius = radius if radius is not None else metadata["default_connectivity_radius"]
    distance_to_boundary = torch.clip(distance_to_boundary / connectivity_radius, -1.0, 1.0)

    # Compute edge features: displacement and distance
    dim = recent_position.size(-1)
    edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
    if(radius is not None):
        edge_displacement /= radius
    else:
        edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # Compute ground truth acceleration (if target_position is provided)
    acceleration = None
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acc_mean = torch.tensor(metadata["acc_mean"])
        acc_std = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
        acceleration = (next_velocity - last_velocity - acc_mean) / acc_std

    # Construct and return the graph
    return pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        node_dist=edge_distance,
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1),
        recent_pos=recent_position,
        target_pos=target_position,
        position_seq=position_seq
    )