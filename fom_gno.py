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
import time
class PhysicsEngine(torch.nn.Module):

    def __init__(
        self,
        device,
        dim=3,
        use_open3d = False,                                                       #GNO Hyperparams
        #in_gno_mlp_hidden_layers = [ 1024, 1024],
        gno_mlp_hidden_layers = [ 128, 256],
        gno_mlp_non_linearity = F.gelu,
        gno_transform_type = 'linear',
        gno_radius=0.025,
    ):
        super().__init__()
        self.device = device
        self.gno_radius = gno_radius
        self.dim = dim
        self.gno_mlp_hidden_layers = gno_mlp_hidden_layers
        kernel_in_dim = 3
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

        self.projection = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )


    def forward(self, rom_ic, fom_ic, rom_f):
        # pre-processing
        
        # node feature: combine categorial feature data.x and contiguous feature data.pos.

        #latent_queries = generate_latent_queries(self.latent_grid_dim, domain_lims = [[-1.0, 1.0], [-1.0,1.0]]).cuda()

        #latent_queries = latent_queries.view(-1, latent_queries.shape[-1])

        neighbor_map = self.nb_search_out(rom_ic, fom_ic, self.gno_radius)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()  # optional but ensures clean measurement
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        start = time.time()
        out = self.gno_in(y=rom_ic, x= fom_ic, f_y = rom_f, neighbors=neighbor_map)
        end = time.time()
        print(f"Forward pass time: {end - start:.4f} seconds")
        print(out.shape)
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        net_usage_bytes = end_mem - start_mem
        net_usage_MB = net_usage_bytes / (1024 ** 2)
        print(f"Memory usage: {net_usage_MB:.2f} MB")
        peak_usage_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memory usage during forward pass: {peak_usage_MB:.2f} MB")
        exit()
        return out