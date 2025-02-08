import torch
from torch import nn
import torch.nn.functional as F
from models.layers import SchInteractionNetwork, MLP
from models.neuraloperator.neuralop.layers.mlp import MLP as NeuralOpMLP
from models.neuraloperator.neuralop.layers.embeddings import PositionalEmbedding
from models.neuraloperator.neuralop.layers.integral_transform import IntegralTransform
from models.neuraloperator.neuralop.layers.neighbor_search import NeighborSearch
from neuralop.models import FNO
from Baselines.mmgpt_base import PhysicsEngine as GNOT
import numpy as np

class PhysicsEngine(torch.nn.Module):

    def __init__(
        self,
        device,
        dim=3,
        use_open3d = False,                                                       #GNO Hyperparams
        #in_gno_mlp_hidden_layers = [ 1024, 1024],
        gno_mlp_hidden_layers = [ 128, 256],
        gno_mlp_non_linearity = F.tanh,
        gno_transform_type = 'nonlinear_kernelonly',
        gno_radius=0.02400,
    ):
        super().__init__()
        self.device = device
        self.gno_radius = gno_radius
        self.dim = dim
        self.gno_mlp_hidden_layers = gno_mlp_hidden_layers
        kernel_in_dim = 6
        self.gno_mlp_hidden_layers.insert(0, kernel_in_dim)
        self.gno_mlp_hidden_layers.append(self.dim)
        self.in_gno_mlp_non_linearity = gno_mlp_non_linearity
        self.in_gno_transform_type = gno_transform_type
        
        
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_in = IntegralTransform(
                    mlp_layers=self.gno_mlp_hidden_layers,
                    mlp_non_linearity=self.in_gno_mlp_non_linearity,
                    transform_type=self.in_gno_transform_type 
        )



    def forward(self, rom_ic, fom_ic, rom_f):
        # pre-processing
        
        # node feature: combine categorial feature data.x and contiguous feature data.pos.

        #latent_queries = generate_latent_queries(self.latent_grid_dim, domain_lims = [[-1.0, 1.0], [-1.0,1.0]]).cuda()

        #latent_queries = latent_queries.view(-1, latent_queries.shape[-1])
        rom_ic = rom_ic.squeeze(0)
        fom_ic = fom_ic.squeeze(0)
        rom_f = rom_f.squeeze(0)
        #rom_f = rom_f - rom_ic
        neighbor_map = self.nb_search_out(rom_ic, fom_ic, self.gno_radius)
        
        out = self.gno_in(y=rom_ic, x= fom_ic, f_y = rom_f, neighbors=neighbor_map)
        #out = out + fom_ic
        out = out.unsqueeze(0)
        return out