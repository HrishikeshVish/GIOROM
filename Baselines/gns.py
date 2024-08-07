import torch
from torch import nn
import torch.nn.functional as F
from models.layers import SchInteractionNetwork, MLP
from models.neuraloperator.neuralop.layers.mlp import MLP as NeuralOpMLP
from models.neuraloperator.neuralop.layers.embeddings import PositionalEmbedding
from models.neuraloperator.neuralop.layers.integral_transform import IntegralTransform
from models.neuraloperator.neuralop.layers.neighbor_search import NeighborSearch
from neuralop.models import FNO
import numpy as np


class PhysicsEngine(torch.nn.Module):

    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10,                                                           # number of GNN layers
        num_particle_types=9,
        particle_type_dim=16,                                                     # embedding dimension of particle types
        dim=3,                                                                    # dimension of the world, typical 2D or 3D
        window_size=5,                                                            # the model looks into W frames before the frame to be predicted
        #window_size=2,
        heads = 3,                                                                 # number of attention heads in GAT and EGAT

    ):
        super().__init__()
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in = MLP(particle_type_dim + dim * (window_size + 2), hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.project2d = torch.nn.Linear(3, 2)
        self.bound2d = torch.nn.Tanh()
        self.dim = dim
        self.hidden_size = hidden_size

        self.n_mp_layers = n_mp_layers

        self.layers = torch.nn.ModuleList([SchInteractionNetwork(
              hidden_size, 3
          ) for _ in range(n_mp_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        # pre-processing
        
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        
        # stack of GNN layers
        for i in range(self.n_mp_layers):

            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        # post-processing
        out = self.node_out(node_feature)

        return out