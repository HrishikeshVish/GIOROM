import jax
import jax.numpy as jnp
from flax import linen as nn

class HighFreqMonteCarloLagrangianMLS(nn.Module):
    output_dim: int
    grid_res: int = 64
    dtype: jnp.dtype = jnp.float32 

    @nn.compact
    def __call__(self, x_query_ref, x_source_ref, x_source_curr, return_aux=False):
        # 1. Coordinate Clipping
        x_source_ref = jnp.clip(x_source_ref, 0.0, 1.0)
        x_query_ref = jnp.clip(x_query_ref, 0.0, 1.0)

        # 2. Lift Features (Drift)
        u_source = x_source_curr - x_source_ref
        f_trans = nn.Dense(3, dtype=self.dtype)(u_source)
        f_trans = nn.gelu(f_trans)
        
        density_tag = jnp.ones((f_trans.shape[0], 1), dtype=self.dtype)
        content = jnp.concatenate([f_trans, density_tag, u_source], axis=-1)
        n_channels = content.shape[-1]
        
        # 3. Trilinear Splatting
        grid_shape = (self.grid_res, self.grid_res, self.grid_res, n_channels)
        flat_grid = jnp.zeros((self.grid_res**3, n_channels), dtype=self.dtype)
        
        coords = x_source_ref * (self.grid_res - 1)
        coords = jnp.clip(coords, 0.0, self.grid_res - 1.001)
        base_idx = jnp.floor(coords).astype(jnp.int32)
        d = coords - base_idx
        
        offsets = jnp.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                             [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
        
        for i in range(8):
            off = offsets[i]
            o_x, o_y, o_z = [off[k].astype(self.dtype) for k in range(3)]
            w_x = d[:, 0] * o_x + (1.0 - d[:, 0]) * (1.0 - o_x)
            w_y = d[:, 1] * o_y + (1.0 - d[:, 1]) * (1.0 - o_y)
            w_z = d[:, 2] * o_z + (1.0 - d[:, 2]) * (1.0 - o_z)
            weight = (w_x * w_y * w_z)[:, None]
            
            idx_x = base_idx[:, 0] + off[0]
            idx_y = base_idx[:, 1] + off[1]
            idx_z = base_idx[:, 2] + off[2]
            flat_indices = idx_x * self.grid_res**2 + idx_y * self.grid_res + idx_z
            flat_grid = flat_grid.at[flat_indices].add(content * weight)

        grid_raw = flat_grid.reshape(grid_shape)

        # 4. FEYNMAN-KAC STOCHASTIC SAMPLING
        q_coords = x_query_ref * (self.grid_res - 1)
        try:
            rng = self.make_rng('feynman_kac')
            K_paths = 8
            sigma = 0.8
        except:
            rng = None 
            K_paths = 1
            sigma = 0.0

        def monte_carlo_integrator(key, grid_volume, center_coords):
            if key is not None:
                noise = jax.random.normal(key, (K_paths, center_coords.shape[0], 3), dtype=self.dtype) * sigma
            else:
                noise = jnp.zeros((1, center_coords.shape[0], 3), dtype=self.dtype)

            def look_up(noise_vector):
                pts = center_coords + noise_vector
                return jax.scipy.ndimage.map_coordinates(grid_volume, pts.T, order=1, mode='nearest')
            
            samples = jax.vmap(look_up)(noise)
            return jnp.mean(samples, axis=0)

        raw_sampled = jax.vmap(
            lambda c: monte_carlo_integrator(rng, c, q_coords)
        )(jnp.moveaxis(grid_raw, -1, 0)).T

        # 5. Normalization & Fusion
        grid_f = raw_sampled[..., :-4]
        grid_d = raw_sampled[..., -4:-3]
        grid_u = raw_sampled[..., -3:]

        denom = jnp.maximum(grid_d, 1e-5)
        mask = (grid_d > 1e-5).astype(self.dtype)

        f_norm = (grid_f / denom) * mask
        u_norm = (grid_u / denom) * mask
        
        # Positional Encoding
        def get_positional_encoding(x, L=3):
            out = []
            for i in range(L):
                freq = 2.0**i * jnp.pi
                out.append(jnp.sin(freq * x))
                out.append(jnp.cos(freq * x))
            return jnp.concatenate(out, axis=-1)

        pe = get_positional_encoding(x_query_ref).astype(self.dtype)
        decoder_input = jnp.concatenate([f_norm, u_norm, pe], axis=-1)
        
        x = nn.Dense(64, dtype=self.dtype)(decoder_input)
        x = nn.gelu(x)
        x = nn.Dense(64, dtype=self.dtype)(x)
        x = nn.gelu(x)
        residual_drift = nn.Dense(self.output_dim, dtype=self.dtype)(x)
        
        pred_clamped = jnp.clip(u_norm + residual_drift, 0.001, 0.999)
        if return_aux:
            return pred_clamped, grid_raw
        return pred_clamped