from transformers import PretrainedConfig
import torch
import torch.nn.functional as F

class TimeStepperConfig(PretrainedConfig):
    """
    Configuration for the TimeStepper model. This configuration holds all the hyperparameters
    necessary for defining and initializing a model that uses graph neural networks (GNNs), 
    GAT, EGAT, and other components for particle dynamics prediction in a 2D or 3D environment.

    Args:
        hidden_size (int): Size of the hidden layers in the model.
        n_mp_layers (int): Number of GNN layers (Message Passing layers).
        num_particle_types (int): Number of distinct particle types in the simulation.
        particle_type_dim (int): Embedding dimension for each particle type.
        dim (int): Dimensionality of the world (e.g., 2 for 2D, 3 for 3D).
        window_size (int): Number of frames the model looks at to predict the next frame.
        heads (int): Number of attention heads in GAT and EGAT.
        
        GNO Hyperparameters:
            use_open3d (bool): Whether to use Open3D for geometry processing.
            in_gno_mlp_hidden_layers (list[int]): List of hidden layer sizes for the input GNO MLP.
            in_gno_mlp_non_linearity: Activation function used in the input GNO MLP.
            in_gno_transform_type (str): Type of transformation to apply at the input GNO.
            out_gno_in_dim (int): Input dimension for the output GNO.
            out_gno_hidden (int): Hidden dimension for the output GNO.
            out_gno_mlp_hidden_layers (list[int]): List of hidden layer sizes for the output GNO MLP.
            out_gno_mlp_non_linearity: Activation function used in the output GNO MLP.
            gno_radius (float): Radius used for GNO computation (local neighborhood size).
            out_gno_transform_type (str): Type of transformation applied in the output GNO.
        
        Projection Hyperparameters:
            projection_channels (int): Number of channels in the projection network.
            projection_layers (int): Number of layers in the projection network.
            projection_n_dim (int): The number of dimensions in the projection space.
            projection_non_linearity: Activation function used in the projection network.
        
        Latent Grid Hyperparameters:
            latent_grid_dim (int): Dimensionality of the latent grid space.
            latent_domain_lims (list[list[float]]): List of 2D or 3D limits for the latent space domain.
    """
    
    def __init__(self, 
                hidden_size: int = 128,
                n_mp_layers: int = 2,
                num_particle_types: int = 9,
                particle_type_dim: int = 16,
                dim: int = 3,
                window_size: int = 5,
                heads: int = 3,
                use_open3d: bool = False,
                in_gno_mlp_hidden_layers: list[int] = [131, 32, 64, 64],
                in_gno_transform_type: str = 'nonlinear_kernelonly',
                out_gno_in_dim: int = 3,
                out_gno_hidden: int = 128,
                out_gno_mlp_hidden_layers: list[int] = [3, 32, 64, 128],
                
                gno_radius: float = 0.125,
                out_gno_transform_type: str = 'linear',
                not_heads: int = 4,
                not_layers: int = 1,
                not_output_size: int = 128,
                not_space_dim: int = 64,
                not_branch_size: int = 3,
                not_trunk_size: int = 64,
                projection_channels: int = 256,
                projection_layers: int = 1,
                projection_n_dim: int = 1,
                
                latent_grid_dim: int = 16,
                latent_domain_lims: list[list[float]] = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], *args, **kwargs):
        
        super().__init__( *args, **kwargs)
        
        # Initialize the model configuration
        self.hidden_size = hidden_size
        self.n_mp_layers = n_mp_layers
        self.num_particle_types = num_particle_types
        self.particle_type_dim = particle_type_dim
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        
        # GNO Hyperparameters
        self.use_open3d = use_open3d
        self.in_gno_mlp_hidden_layers = in_gno_mlp_hidden_layers
        self.in_gno_transform_type = in_gno_transform_type
        self.out_gno_in_dim = out_gno_in_dim
        self.out_gno_hidden = out_gno_hidden
        self.out_gno_mlp_hidden_layers = out_gno_mlp_hidden_layers
        self.gno_radius = gno_radius
        self.out_gno_transform_type = out_gno_transform_type
        
        # Neural Operator Transformer Hyperparameters
        self.not_heads = not_heads
        self.not_layers = not_layers
        self.not_output_size = not_output_size
        self.not_space_dim = not_space_dim
        self.not_branch_size = not_branch_size
        self.not_trunk_size = not_trunk_size
        
        # Projection Hyperparameters
        self.projection_channels = projection_channels
        self.projection_layers = projection_layers
        self.projection_n_dim = projection_n_dim
        
        # Latent Grid Hyperparameters
        self.latent_grid_dim = latent_grid_dim
        self.latent_domain_lims = latent_domain_lims