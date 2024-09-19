import torch
import torch_geometric as pyg
import json
import pickle
import os
import numpy as np
import random
from dgl.geometry import farthest_point_sampler
from scipy.spatial import Delaunay

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


def preprocess(particle_type, position_seq, target_position, metadata, noise_std, radius = None, graph_type='radius'):
    """Preprocess a trajectory and construct the graph"""
    # apply noise to the trajectory
    position_noise = generate_noise(position_seq, noise_std)
    old_position_seq = position_seq
    position_seq = position_seq + position_noise

    # calculate the velocities of particles
    recent_position = position_seq[:, -1]
    
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    
    
    # construct the graph based on the distances between particles
    n_particle = recent_position.size(0)
    if(graph_type == 'delaunay'):
        triangulation = Delaunay(recent_position.numpy())
        edges = delaunay2edges(triangulation)
        edge_index = torch.from_numpy(edges.astype(np.int64)).permute(1,0)

    elif(graph_type == 'radius'):
        if radius is not None:
            edge_index = pyg.nn.radius_graph(recent_position, radius, loop=True, max_num_neighbors=n_particle)
        else:
            edge_index = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=True, max_num_neighbors=n_particle)

    # node-level features: velocity, distance to the boundary
    normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    if(radius is not None):
        distance_to_boundary = torch.clip(distance_to_boundary / radius, -1.0, 1.0)
    else:
        distance_to_boundary = torch.clip(distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0)

    # edge-level features: displacement, distance
    dim = recent_position.size(-1)
    edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
    if(radius is not None):
        edge_displacement /= radius
    else:
        edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # ground truth for training
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
    else:
        acceleration = None

    # return the graph with features

    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        node_dist=edge_distance,
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1),
        recent_pos = recent_position,
        target_pos = target_position,
        position_seq = old_position_seq
    )
    return graph


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data_path=None, split='valid', window_length=7, noise_std=0.0, return_pos=False, sampling = False, sampling_strategy='random', graph_type='radius', sampling_percent_lb = 0.20, sampling_percent_ub=0.33, radius=None):
        super().__init__()

        # load dataset from the disk
        #self.data_path = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
        self.data_path = data_path
        with open(os.path.join(self.data_path, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.window_length = window_length
        self.noise_std = noise_std
        self.return_pos = return_pos
        self.radius = radius
        self.graph_type = graph_type
        self.tensorpath = os.path.join(self.data_path, split)
        if('.pt' in self.tensorpath):
            dataset = torch.load(self.tensorpath)
        else:
            file = open(self.tensorpath, 'rb')
            dataset = pickle.load(file)

        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']
        self.sampling = sampling
        self.sampling_strategy = sampling_strategy

        self.dim = self.position[0].shape[2]

        self.sampling_percent_lb = sampling_percent_lb
        self.sampling_percent_ub = sampling_percent_ub


    def len(self):
        return len(self.position)

    def get(self, idx):
        # load corresponding data for this time slice


        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        target_position = torch.from_numpy(self.outputs[idx])


        if(self.sampling == True):
            mesh_size =  np.random.randint(int(self.sampling_percent_lb*particle_type.shape[0]), int(self.sampling_percent_ub*particle_type.shape[0]))
            while(mesh_size %10 !=0):
                    mesh_size += 1
            if(self.sampling_strategy == 'random'):
                points = sorted(list(random.sample(range(0, particle_type.shape[0]), mesh_size)))
                #points = sorted(list(range(0, particle_type.shape[0], 3)))
                particle_type = particle_type[points]
                position_seq = position_seq[points]
                target_position = target_position[points]
            elif(self.sampling_strategy == 'fps'):
                init_pos = position_seq.permute(1, 0, 2)[0].unsqueeze(0)
                point_idx = farthest_point_sampler(init_pos, mesh_size)[0]
                particle_type = particle_type[point_idx]
                position_seq = position_seq[point_idx]
                target_position = target_position[point_idx]

        
        # construct the graph
        with torch.no_grad():
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std, self.radius, self.graph_type)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph


class RolloutDataset(pyg.data.Dataset):
    def __init__(self, data_path, split, window_length=7, sampling=False, sampling_strategy='random', graph_type='radius', mesh_size=90, radius=None):
        super().__init__()
        
        # load data from the disk
        self.data_path = data_path
        with open(os.path.join(self.data_path, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.window_length = window_length
        self.sampling = sampling
        self.graph_type = graph_type
        self.sampling_strategy = sampling_strategy
        self.radius = radius
        self.tensorpath = os.path.join(self.data_path, split)
        dataset = torch.load(self.tensorpath)
        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']
        self.mesh_size = mesh_size

        self.dim = self.position[0].shape[2]
    def len(self):
        return len(self.position)
    
    def get(self, idx):


        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        position_seq = torch.permute(position_seq, dims=(1,0,2))
        
        target_position = torch.from_numpy(self.outputs[idx])
        if(self.sampling):
            if(self.sampling_strategy == 'random'):
                self.points = sorted(random.sample(range(0, particle_type.shape[0]), self.mesh_size))
                particle_type = particle_type[self.points]
                position_seq = position_seq.permute(1,0,2)
                position_seq = position_seq[self.points]
                position_seq = position_seq.permute(1,0,2)
                target_position = target_position[self.points]
            elif(self.sampling_strategy == 'fps'):
                init_pos = position_seq.permute(1, 0, 2)[0].unsqueeze(0)
                point_idx = farthest_point_sampler(init_pos, self.mesh_size)[0]
                particle_type = particle_type[point_idx]
                position_seq = position_seq[point_idx]
                target_position = target_position[point_idx]
        data = {"particle_type": particle_type, "position": position_seq}
        return data