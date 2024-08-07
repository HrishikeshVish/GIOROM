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
        hidden_size=128,
        n_mp_layers=2,                                                           # number of GNN layers
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

        gno_radius=0.045,
        out_gno_transform_type='linear',

        projection_channels=16,
        projection_layers=1,
        projection_n_dim = 1,
        projection_non_linearity = F.gelu,

        latent_grid_dim = 16
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in = MLP(particle_type_dim + dim * (window_size + 2), hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.project2d = torch.nn.Linear(3, 2)
        self.downsample = torch.nn.Linear(hidden_size, fno_in_channels)
        self.bound2d = torch.nn.Tanh()
        self.dim = dim
        self.hidden_size = hidden_size

        self.n_mp_layers = n_mp_layers
          
        
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
                                out_channels=dim,
                                lifting_channels=self.fno_lifting_channels,
                                projection_channels=self.fno_projection_channels,
                                n_layers=self.fno_layers, 
                                use_mlp=self.fno_use_mlp,
                                stabilizer=self.fno_stabilizers,
                                non_linearity=self.fno_non_linearity,
                                preactivation=self.fno_preactivation
                            )
        self.reset_parameters()



        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        # pre-processing
        
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        node_feature = self.downsample(node_feature)
        edge_feature = self.edge_in(data.edge_attr)

        latent_in = torch.unsqueeze(node_feature, dim=1)
        latent_in = latent_in.permute(dims=(2, 0, 1))
        latent_in = latent_in.unsqueeze(dim=0)
        latent = self.fno_mapper(latent_in)
        latent = torch.squeeze(latent, dim=0)
        latent = torch.squeeze(latent, dim=1)
        latent = torch.squeeze(latent, dim=1)
        latent = torch.squeeze(latent, dim =2)
        latent = torch.permute(latent, dims=(1,0))
        #latent = torch.reshape(latent, shape=(self.latent_grid_dim*self.latent_grid_dim, self.hidden_size))

        return latent
