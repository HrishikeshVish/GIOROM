import torch
from torch import nn
import torch.nn.functional as F
from models.layers.layers import SchInteractionNetwork, MLP
from models.neuralop030.neuralop.layers.mlp import MLP as ProjectionMLP
from models.neuralop030.neuralop.layers.embeddings import PositionalEmbedding
from models.neuralop030.neuralop.layers.integral_transform import IntegralTransform
from models.neuralop030.neuralop.layers.neighbor_search import NeighborSearch
from models.layers.mmgpt_base import GPTNOT as GNOT
from models.layers.layer_utils import generate_latent_queries
from neuralop.models import FNO
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from models.config import TimeStepperConfig

class PhysicsEngine(PreTrainedModel):
    
    
    config_class = TimeStepperConfig
    def __init__(self, config: TimeStepperConfig, *args, **kwargs):
        
        super().__init__(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        self.embed_type = torch.nn.Embedding(config.num_particle_types, config.particle_type_dim)
        self.node_in = MLP(config.particle_type_dim + config.dim * (config.window_size + 2), config.hidden_size, config.hidden_size, 3)
        self.edge_in = MLP(config.dim + 1, config.hidden_size, config.hidden_size, 3)
        self.node_out = MLP(config.hidden_size, config.hidden_size, config.dim, 3, layernorm=False)
        self.project2d = torch.nn.Linear(3, 2)
        self.bound2d = torch.nn.Tanh()
        self.dim = config.dim
        self.hidden_size = config.hidden_size

        self.n_mp_layers = config.n_mp_layers

        self.in_layers = torch.nn.ModuleList([SchInteractionNetwork(
              config.hidden_size, 3
        ) for _ in range(config.n_mp_layers)])

        self.out_layers = torch.nn.ModuleList([SchInteractionNetwork(
              config.hidden_size, 3
        ) for _ in range(config.n_mp_layers)])
          
        

        self.out_gno_hidden = config.out_gno_hidden
        self.gnot_layer = GNOT(self._device, trunk_size=config.not_trunk_size, branch_sizes=[config.not_branch_size], space_dim=config.not_space_dim, output_size=config.not_output_size, n_layers=config.not_layers, n_head=config.not_heads)
        self.reset_parameters()

        self.gno_radius = config.gno_radius
        self.latent_grid_dim = config.latent_grid_dim
        
        self.in_gno_mlp_hidden_layers = config.in_gno_mlp_hidden_layers
        self.in_gno_transform_type = config.in_gno_transform_type
        
        
        self.nb_search_out = NeighborSearch(use_open3d=config.use_open3d)
        self.gno_in = IntegralTransform(
                    mlp_layers=self.in_gno_mlp_hidden_layers,
                    mlp_non_linearity=F.gelu,
                    transform_type=self.in_gno_transform_type 
        )

        self.out_gno_transform_type = config.out_gno_transform_type
        self.out_gno_mlp_hidden_layers = config.out_gno_mlp_hidden_layers

        out_kernel_in_dim = config.out_gno_in_dim
        out_kernel_in_dim += self.fno_hidden_channels if self.out_gno_transform_type != 'linear' else 0
        out_gno_mlp_hidden_layers=self.out_gno_mlp_hidden_layers
        
        self.gno_out = IntegralTransform(
                    mlp_layers=out_gno_mlp_hidden_layers,
                    mlp_non_linearity=F.gelu,
                    transform_type=self.out_gno_transform_type,
        )

        self.projection_channels=config.projection_channels
        self.projection_layers = config.projection_layers
        self.projection_n_dim = config.projection_n_dim
        self.projection_non_linearity = F.gelu
        self.projection = ProjectionMLP(in_channels=self.out_gno_hidden, 
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


        latent_queries = generate_latent_queries(self.latent_grid_dim, domain_lims = [[-1.0, 1.0], [-1.0,1.0]]).cuda()

        latent_queries = latent_queries.view(-1, latent_queries.shape[-1])
        
        
        neighbor_map = self.nb_search_out(data.recent_pos, latent_queries, self.config.gno_radius)
        
        
        in_p = self.gno_in(y=data.recent_pos, x= latent_queries, f_y = out, neighbors=neighbor_map)
        spatial_res = latent_queries.shape[:-1]
        in_p = in_p.view(*spatial_res, 32).unsqueeze(0)
       
        #in_p = in_p.reshape(self.latent_grid_dim, self.latent_grid_dim, self.dim)
        
        latent_input = in_p
        recent_pos = latent_queries

        latent = self.gnot_layer(latent_input, recent_pos)

        # latent_input = torch.unsqueeze(in_p, dim=0)
        # latent_input = torch.permute(latent_input, dims=(0, 3, 1, 2))
        
        # latent = self.fno_mapper(latent_input)
        # latent = torch.squeeze(latent, dim=0)
        # latent = torch.squeeze(latent, dim=1)
        # latent = torch.squeeze(latent, dim=1)
        # latent = torch.reshape(latent, shape=(self.latent_grid_dim*self.latent_grid_dim, self.hidden_size))
        
        neighbor_map = self.nb_search_out(latent_queries, data.recent_pos, self.gno_radius)

        out = self.gno_out(y = latent_queries, neighbors = neighbor_map, f_y=latent, x = data.recent_pos)
        out = out.unsqueeze(0).permute(0,2,1)
        node_feature = self.projection(out).squeeze(0).permute(1, 0)

        
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.out_layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        out = self.node_out(node_feature)

        return out