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
        n_mp_layers=1,                                                           # number of GNN layers
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

        use_open3d = False,                                                       #GNO Hyperparams
        in_gno_mlp_hidden_layers = [32,64],
        in_gno_mlp_non_linearity = F.gelu,
        in_gno_transform_type = 'nonlinear_kernelonly',
        out_gno_in_dim = 2,
        out_gno_hidden=128,
        out_gno_mlp_hidden_layers = [32, 64],
        out_gno_mlp_non_linearity = F.gelu,

        gno_radius=0.145,
        out_gno_transform_type='linear',

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


        self.fno_mapper = FNO(n_modes = self.fno_n_modes,
                                hidden_channels=self.fno_hidden_channels,
                                in_channels=self.fno_in_channels,
                                out_channels=hidden_size,
                                lifting_channels=self.fno_lifting_channels,
                                projection_channels=self.fno_projection_channels,
                                n_layers=self.fno_layers, 
                                use_mlp=self.fno_use_mlp,
                                stabilizer=self.fno_stabilizers,
                                non_linearity=self.fno_non_linearity,
                                preactivation=self.fno_preactivation
                            )
        self.reset_parameters()

        self.gno_radius = gno_radius
        self.latent_grid_dim = latent_grid_dim
        self.latent_domain_lims = latent_domain_lims

        self.in_gno_mlp_hidden_layers = in_gno_mlp_hidden_layers
        kernel_in_dim = 2 + self.fno_in_channels 
        self.in_gno_mlp_hidden_layers.insert(0, kernel_in_dim)
        self.in_gno_mlp_hidden_layers.append(self.fno_in_channels)
        self.in_gno_mlp_non_linearity = in_gno_mlp_non_linearity
        self.in_gno_transform_type = in_gno_transform_type
        
        
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_in = IntegralTransform(
                    mlp_layers=self.in_gno_mlp_hidden_layers,
                    mlp_non_linearity=self.in_gno_mlp_non_linearity,
                    transform_type=self.in_gno_transform_type 
        )

        self.out_gno_transform_type = out_gno_transform_type
        self.out_gno_mlp_hidden_layers = out_gno_mlp_hidden_layers
        self.out_gno_mlp_non_linearity = out_gno_mlp_non_linearity

        out_kernel_in_dim = out_gno_in_dim
        out_kernel_in_dim += self.fno_hidden_channels if self.out_gno_transform_type != 'linear' else 0
        out_gno_mlp_hidden_layers=self.out_gno_mlp_hidden_layers
        out_gno_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_mlp_hidden_layers.append(self.out_gno_hidden)
        
        self.gno_out = IntegralTransform(
                    mlp_layers=out_gno_mlp_hidden_layers,
                    mlp_non_linearity=out_gno_mlp_non_linearity,
                    transform_type=self.out_gno_transform_type,
        )

        self.projection_channels=projection_channels
        self.projection_layers = projection_layers
        self.projection_n_dim = projection_n_dim
        self.projection_non_linearity = projection_non_linearity
        self.projection = NeuralOpMLP(in_channels=self.out_gno_hidden, 
                        out_channels=self.out_gno_hidden, 
                        #out_channels=2, 
                        hidden_channels=self.projection_channels, 
                        n_layers=self.projection_layers, 
                        n_dim=self.projection_n_dim, 
                        non_linearity=self.projection_non_linearity)

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

        latent_queries = latent_queries.view(-1, latent_queries.shape[-1])
        
        
        neighbor_map = self.nb_search_out(pos_2d, latent_queries, self.gno_radius)
        
        
        in_p = self.gno_in(y=pos_2d, x= latent_queries, f_y = out, neighbors=neighbor_map)
        spatial_res = latent_queries.shape[:-1]
        in_p = in_p.view(*spatial_res, self.fno_in_channels).unsqueeze(0)
       
        in_p = in_p.reshape(self.latent_grid_dim, self.latent_grid_dim, self.dim)
        
        latent_input = torch.unsqueeze(in_p, dim=0)
        latent_input = torch.permute(latent_input, dims=(0, 3, 1, 2))
        
        latent = self.fno_mapper(latent_input)
        latent = torch.reshape(latent, shape=(self.latent_grid_dim*self.latent_grid_dim, self.hidden_size))
        
        neighbor_map = self.nb_search_out(latent_queries, pos_2d, self.gno_radius)

        out = self.gno_out(y = latent_queries, neighbors = neighbor_map, f_y=latent, x = pos_2d)
        out = out.unsqueeze(0).permute(0,2,1)
        node_feature = self.projection(out).squeeze(0).permute(1, 0)

        
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.out_layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        out = self.node_out(node_feature)

        return out