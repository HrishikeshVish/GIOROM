import torch
from torch import nn
import torch.nn.functional as F
from models.layers import SchInteractionNetwork, MLP
from models.neuraloperator.neuralop.layers.mlp import MLP as NeuralOpMLP
from models.neuraloperator.neuralop.layers.embeddings import PositionalEmbedding
from models.neuraloperator.neuralop.layers.integral_transform import IntegralTransform
from models.neuraloperator.neuralop.layers.neighbor_search import NeighborSearch
from neuralop.models import GINO

from neuralop.models import FNO
import numpy as np

def generate_latent_queries(query_res, pad=0, domain_lims=[[-1.,1.],[-1.,1.]]):
    oneDMeshes = []
    for lower,upper in domain_lims:
        oneDMesh = np.linspace(lower,upper,query_res)
        if pad > 0:
            start = np.linspace(lower - pad/query_res, lower, pad+1)
            stop = np.linspace(upper, upper + pad/query_res, pad+1)
            oneDMesh = np.concatenate([start,oneDMesh,stop])
        oneDMeshes.append(oneDMesh)
    grid = np.stack(np.meshgrid(*oneDMeshes,indexing='xy')) # c, x, y, z(?)
    grid = torch.from_numpy(grid.astype(np.float32))
    latent_queries = grid.permute(*list(range(1,len(domain_lims)+1)), 0)
    return latent_queries

class PhysicsEngine(torch.nn.Module):

    def __init__(
        self,
        device,
        hidden_size=128,
        n_mp_layers=3,                                                           # number of GNN layers
        num_particle_types=9,
        particle_type_dim=16,                                                     # embedding dimension of particle types
        dim=3,                                                                    # dimension of the world, typical 2D or 3D
        window_size=5,                                                            # the model looks into W frames before the frame to be predicted
        #window_size=2,
        heads = 3,                                                                 # number of attention heads in GAT and EGAT

        n_modes = (24,24),                                                        #FNO Hyperparams
        fno_in_channels=3,
        fno_hidden_channels=32,
        fno_lifting_channels=32,
        fno_projection_channels=32,
        fno_layers=2,
        fno_use_mlp = True,
        fno_stabilizers='tanh',
        fno_non_linearity = F.gelu,
        fno_preactivation = True,
        fno_domain_padding=0.,
        fno_rank=0.8,
        fno_ada_in_features=8,
        fno_norm='group_norm',
        fno_mlp_expansion=0.0,

        use_open3d = False,                                                       #GNO Hyperparams
        in_gno_mlp_hidden_layers = [32,64],

        in_gno_transform_type = 'nonlinear_kernelonly',
        out_gno_transform_type='linear',
        out_gno_in_dim = 2,
        out_gno_hidden=128,
        out_gno_mlp_hidden_layers = [32, 64],
        out_gno_tanh=None,
        gno_mlp_non_linearity = F.gelu,

        gno_radius=0.145,
        gno_coord_dim=2,
        gno_coord_embed_dim=32,
        gno_embed_max_positions=600,
        gno_use_torch_scatter=True,

        

        projection_channels=16,
        projection_layers=1,
        projection_n_dim = 1,
        projection_non_linearity = F.gelu,

        latent_grid_dim = 16,
        latent_domain_lims = [[-1.0, 1.0], [-1.0, 1.0]]

    ):
        super().__init__()
        self.device = device
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

        self.in_layers = torch.nn.ModuleList([SchInteractionNetwork(
              hidden_size, 3
        ) for _ in range(n_mp_layers)])

        self.out_layers = torch.nn.ModuleList([SchInteractionNetwork(
              hidden_size, 3
        ) for _ in range(n_mp_layers)])
          
        
        self.fno_in_channels = fno_in_channels
        self.fno_hidden_channels = fno_hidden_channels
        self.fno_n_modes = n_modes
        self.fno_lifting_channels = fno_lifting_channels
        self.fno_projection_channels = fno_projection_channels
        self.fno_layers = fno_layers
        self.fno_use_mlp = fno_use_mlp
        self.fno_stabilizers = fno_stabilizers
        self.fno_non_linearity = fno_non_linearity
        self.fno_preactivation = fno_preactivation

        self.out_gno_hidden = out_gno_hidden

        self.gino = GINO(
            in_channels=self.fno_in_channels,
            out_channels=self.out_gno_hidden,
            projection_channels=projection_channels,
            gno_coord_dim=gno_coord_dim,
            gno_coord_embed_dim=gno_coord_embed_dim,
            gno_embed_max_positions=gno_embed_max_positions,
            gno_radius=gno_radius,
            in_gno_mlp_hidden_layers=in_gno_mlp_hidden_layers,
            out_gno_mlp_hidden_layers=out_gno_mlp_hidden_layers,
            gno_mlp_non_linearity=gno_mlp_non_linearity, 
            in_gno_transform_type=in_gno_transform_type,
            out_gno_transform_type=out_gno_transform_type,
            gno_use_open3d=use_open3d,
            gno_use_torch_scatter=gno_use_torch_scatter,
            out_gno_tanh=out_gno_tanh,
            fno_in_channels=self.fno_in_channels,
            fno_n_modes=self.fno_n_modes, 
            fno_hidden_channels=self.fno_hidden_channels,
            fno_lifting_channels=self.fno_lifting_channels,
            fno_projection_channels=self.fno_projection_channels,
            fno_n_layers=self.fno_layers,
            fno_use_mlp=self.fno_use_mlp,
            fno_mlp_expansion=fno_mlp_expansion,
            fno_norm=fno_norm,
            fno_ada_in_features=fno_ada_in_features,
            fno_rank=fno_rank,
            fno_domain_padding=fno_domain_padding,
        )
        self.reset_parameters()

        self.gno_radius = gno_radius
        self.latent_grid_dim = latent_grid_dim
        self.latent_domain_lims = latent_domain_lims
        
        self.in_gno_mlp_hidden_layers = in_gno_mlp_hidden_layers
        

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

            node_feature, edge_feature = self.in_layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        # post-processing
        out = self.node_out(node_feature)

        pos_2d = self.project2d(data.recent_pos)
        pos_2d = self.bound2d(pos_2d)

        latent_queries = generate_latent_queries(self.latent_grid_dim, domain_lims = self.latent_domain_lims).to(self.device)
        latent_queries = latent_queries.reshape(1, self.latent_grid_dim, self.latent_grid_dim, 2)

        node_feature = self.gino(f=out.unsqueeze(0), input_geom=pos_2d.unsqueeze(0), output_queries=pos_2d.unsqueeze(0), latent_queries=latent_queries)
        node_feature = node_feature.squeeze(0)
        
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.out_layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        out = self.node_out(node_feature)

        return out