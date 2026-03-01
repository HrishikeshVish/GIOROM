import jax
import jax.numpy as jnp
import numpy as np

def calculate_relative_l2_jax(pred, gt, eps=1e-8):
    error_norm = jnp.linalg.norm((pred - gt).ravel())
    gt_norm = jnp.linalg.norm(gt.ravel())
    return error_norm / (gt_norm + eps)

def chamfer_distance(pred, gt, samples=2048):
    pred, gt = jnp.array(pred), jnp.array(gt)
    N = pred.shape[0]
    
    if samples > 0 and N > samples:
        idx_pred = np.random.choice(N, samples, replace=False)
        idx_gt = np.random.choice(N, samples, replace=False)
        pred_sub, gt_sub = pred[idx_pred], gt[idx_gt]
    else:
        pred_sub, gt_sub = pred, gt

    x = pred_sub[:, None, :] 
    y = gt_sub[None, :, :]   
    dist_sq = jnp.sum((x - y) ** 2, axis=-1)
    
    return jnp.mean(jnp.min(dist_sq, axis=1)) + jnp.mean(jnp.min(dist_sq, axis=0))

def compute_reviewer_metrics(grid_raw, n_particles_total):
    if grid_raw.ndim == 4:
        grid_raw = jnp.expand_dims(grid_raw, axis=0)
        
    B, H, W, D, C = grid_raw.shape
    flat_grid = grid_raw.reshape(B, -1, C)
    
    channel_sums = jnp.sum(flat_grid, axis=1) 
    avg_channel_sums = jnp.mean(channel_sums, axis=0)
    
    diffs = jnp.abs(avg_channel_sums - n_particles_total)
    density_idx = jnp.argmin(diffs)
    found_mass = avg_channel_sums[density_idx] 
    
    density_map = flat_grid[..., density_idx]
    n_map = density_map * 8.0
    
    magnitudes = jnp.linalg.norm(flat_grid, axis=-1)
    is_active = magnitudes > 1e-7
    num_active_voxels = jnp.sum(is_active)
    total_voxels = flat_grid.shape[1]
    
    active_n_values = n_map[is_active]
    local_support_n = jnp.where(active_n_values.size > 0, jnp.mean(active_n_values), 0.0)
    feature_count = C - 1

    return {
        "local_dof_n": local_support_n * feature_count,       
        "global_dof_N": n_particles_total,    
        "sparsity": num_active_voxels / total_voxels,
        "active_voxel_count": num_active_voxels,
        "mass_recovered": found_mass,         
        "max_local_overlap": jnp.max(active_n_values) if active_n_values.size > 0 else 0.0
    }