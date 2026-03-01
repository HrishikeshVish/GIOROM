import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing from standard library, fallback to local if needed
try:
    from neuralop.layers import IntegralTransform, NeighborSearch
except ImportError:
    from models.neuraloperator.neuralop.layers.integral_transform import IntegralTransform
    from models.neuraloperator.neuralop.layers.neighbor_search import NeighborSearch

class GNO(nn.Module):
    def __init__(
        self,
        dim=3,
        gno_radius=0.035, 
        hidden_dim=12
    ):
        super().__init__()
        self.dim = dim
        self.gno_radius = gno_radius
        
        self.nb_search = NeighborSearch(use_open3d=False)
        
        self.gno_layer = IntegralTransform(
            mlp_layers=[3, 12, 12, dim], 
            mlp_non_linearity=F.gelu,
            transform_type='linear' 
        )

        self.projection = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_sparse, f_sparse, x_dense):
        """
        x_sparse: (N_s, 3) Source Positions (Reference Config)
        f_sparse: (N_s, 3) Source Features (e.g. Displacement)
        x_dense:  (N_d, 3) Query Positions (Reference Config)
        """
        neighbor_map = self.nb_search(x_sparse, x_dense, self.gno_radius)
        out_features = self.gno_layer(y=x_sparse, x=x_dense, f_y=f_sparse, neighbors=neighbor_map)
        out = self.projection(out_features)
        
        return out