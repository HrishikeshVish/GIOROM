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
    """
    A physics-informed neural network model for particle-based simulations, utilizing
    Graph Neural Networks (GNNs), Neural Operators, and Integral Transforms to learn
dynamic systems.
    
    Attributes:
        config (TimeStepperConfig): Configuration object with hyperparameters.
        embed_type (nn.Embedding): Embedding layer for particle type encoding.
        node_in (MLP): MLP for encoding node features.
        edge_in (MLP): MLP for encoding edge features.
        node_out (MLP): MLP for decoding node features into final output.
        in_layers (nn.ModuleList): GNN layers for feature propagation.
        out_layers (nn.ModuleList): Additional GNN layers for refinement.
        gnot_layer (GNOT): Neural Operator Transformer for latent space modeling.
        nb_search_out (NeighborSearch): Neighbor search module for geometric queries.
        gno_in (IntegralTransform): Integral transform for encoding latent representations.
        gno_out (IntegralTransform): Integral transform for decoding latent representations.
        projection (ProjectionMLP): Projection MLP for feature transformation.
    """
    config_class = TimeStepperConfig

    def __init__(self, config: TimeStepperConfig, *args, **kwargs):
        super().__init__(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Particle type embedding
        self.embed_type = nn.Embedding(config.num_particle_types, config.particle_type_dim)

        # Node and edge encoders
        self.node_in = MLP(
            input_size=config.particle_type_dim + config.dim * (config.window_size + 2),
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            layers=3
        )
        self.edge_in = MLP(
            input_size=config.dim + 1,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            layers=3
        )
        self.node_out = MLP(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.dim,
            layers=3,
            layernorm=False
        )

        # Graph Neural Network layers
        self.in_layers = nn.ModuleList(
            [SchInteractionNetwork(config.hidden_size, 3) for _ in range(config.n_mp_layers)]
        )
        self.out_layers = nn.ModuleList(
            [SchInteractionNetwork(config.hidden_size, 3) for _ in range(config.n_mp_layers)]
        )
        # Neural Operator Transformer
        self.gnot_layer = GNOT(
            self._device,
            trunk_size=config.not_trunk_size,
            branch_sizes=[config.not_branch_size],
            space_dim=config.not_space_dim,
            output_size=config.not_output_size,
            n_layers=config.not_layers,
            n_head=config.not_heads
        )

        
            
        
        # Neighbor search and Integral Transforms
        self.nb_search_out = NeighborSearch(use_open3d=config.use_open3d)
        
        if(config.in_gno_transform_type == 'nonlinear_kernelonly'):
            assert config.in_gno_mlp_hidden_layers[0] == config.hidden_size + config.dim
            
        self.gno_in = IntegralTransform(
            mlp_layers=config.in_gno_mlp_hidden_layers,
            mlp_non_linearity=F.gelu,
            transform_type=config.in_gno_transform_type
        )
        
        if(config.out_gno_transform_type == 'linear'):
            assert config.out_gno_mlp_hidden_layers[0] == config.dim
            
        self.gno_out = IntegralTransform(
            mlp_layers=config.out_gno_mlp_hidden_layers,
            mlp_non_linearity=F.gelu,
            transform_type=config.out_gno_transform_type
        )

        # Projection layer for feature transformation
        self.projection = ProjectionMLP(
            in_channels=config.out_gno_hidden,
            out_channels=config.out_gno_hidden,
            hidden_channels=config.projection_channels,
            n_layers=config.projection_layers,
            n_dim=config.projection_n_dim,
            non_linearity=F.gelu
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes embedding layer weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        """
        Processes input data through GNN layers, neural operators, and integral transforms
        to predict the system's next state.

        Args:
            data: Input object containing particle positions, edge attributes, and connectivity.
        
        Returns:
            torch.Tensor: Predicted positions of particles after the simulation step.
        """
        # Encode node features by combining categorical and positional information
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)

        # GNN processing (initial layers)
        for layer in self.in_layers:
            node_feature, edge_feature = layer(
                node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist
            )
        
        # Generate latent space queries
        latent_queries = generate_latent_queries(
            self.config.latent_grid_dim, domain_lims=self.config.latent_domain_lims
        ).to(self._device)
        latent_queries = latent_queries.view(-1, latent_queries.shape[-1])
        
        # Neighbor search for integral transform input
        neighbor_map = self.nb_search_out(data.recent_pos, latent_queries, self.config.gno_radius)
        in_p = self.gno_in(
            y=data.recent_pos, x=latent_queries, f_y=node_feature, neighbors=neighbor_map
        ).view(*latent_queries.shape[:-1], self.config.not_trunk_size).unsqueeze(0)

        # Neural Operator Transformer Processing
        latent = self.gnot_layer(in_p, latent_queries)
        neighbor_map = self.nb_search_out(latent_queries, data.recent_pos, self.config.gno_radius)

        # Apply integral transform on the latent representation
        out = self.gno_out(
            y=latent_queries, neighbors=neighbor_map, f_y=latent, x=data.recent_pos
        ).unsqueeze(0).permute(0, 2, 1)
        node_feature = self.projection(out).squeeze(0).permute(1, 0) + node_feature

        # GNN processing (output layers)
        for layer in self.out_layers:
            node_feature, edge_feature = layer(
                node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist
            )
        
        return self.node_out(node_feature)