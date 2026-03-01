import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

# Import modularized codebase
from src.utils.system_utils import setup_environment, get_vram_usage_mb
from src.utils.data_utils import load_and_prep_data
from src.utils.eval_utils import compute_reviewer_metrics
from src.giorom.model import HighFreqMonteCarloLagrangianMLS
from scripts.evaluate_giorom import verify_results_aggregate_stoc, run_dynamic_benchmark_stoc

# Setup JAX and CUDA paths BEFORE large allocations
setup_environment()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--sparsity', type=int, default=20)
    parser.add_argument('--param', type=str, default='Grid', choices=['Grid', 'sampling'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data', type=str, default="/media/csuser/DATA/new_dataset/owl/rollout_full.pt")
    parser.add_argument('--obj_out_dir', type=str, default=None, help="Output folder for .obj files")
    args = parser.parse_args()

    print(f"JAX Devices: {jax.devices()}")
    train_data, test_data, video_traj, dmin, dmax = load_and_prep_data(args.data)
    
    model = HighFreqMonteCarloLagrangianMLS(3, grid_res=args.grid)
    
    x_d_ref = jnp.array(train_data[0, 0])
    x_s_ref = x_d_ref[::int(args.sparsity)]
    
    print("\n--- Initializing Model Parameters ---")
    
    key = jax.random.PRNGKey(0)
    key_params, key_dropout = jax.random.split(key)
    params = model.init({'params': key_params, 'feynman_kac': key_dropout}, x_d_ref, x_s_ref, x_s_ref)
    
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(state, params, x_q_ref, x_s_ref, x_d_curr, x_s_curr, rng_key):
        step_key, _ = jax.random.split(rng_key)
        
        def loss_fn(p):
            pred, grid_raw = model.apply(
                p, x_q_ref, x_s_ref, x_s_curr, 
                rngs={'feynman_kac': step_key}, 
                return_aux=True
            )
            mse = jnp.mean((pred - x_d_curr)**2)
            return mse, grid_raw 

        (loss, grid_raw), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, grid_raw

    print("\n--- Training ---")
    n_traj, n_time = train_data.shape[0], train_data.shape[1]
    
    avg_dof = avg_mem_used = peak_mem_used = n_steps = 0.0
    
    for epoch in range(args.epochs):
        n_samples = 2000
        traj_idxs = np.random.randint(0, n_traj, n_samples)
        time_idxs = np.random.randint(0, n_time, n_samples)
        
        avg_loss = 0.0
        jax.clear_backends()
        
        pbar = tqdm(range(0, n_samples, args.batch_size), desc=f"Epoch {epoch}")
        
        for i in pbar:
            t_idx, tm_idx = traj_idxs[i], time_idxs[i]
            n_steps += 1
            
            x_d_ref = jnp.array(train_data[t_idx, 0])
            x_s_ref = x_d_ref[::int(args.sparsity)]
            x_d_curr = jnp.array(train_data[t_idx, tm_idx])
            x_s_curr = x_d_curr[::int(args.sparsity)]
            
            key, subkey = jax.random.split(key)
            params, opt_state, loss, grid_raw = train_step(
                opt_state, params, x_d_ref, x_s_ref, x_d_curr, x_s_curr, subkey
            )
            
            used_mb, peak_mb = get_vram_usage_mb()
            avg_mem_used += used_mb
            peak_mem_used = max(peak_mem_used, peak_mb)
            
            stats = compute_reviewer_metrics(grid_raw, n_particles_total=x_s_ref.shape[0])
            avg_dof += stats['local_dof_n']
            avg_loss += loss.item()
            
            pbar.set_postfix(
                loss=avg_loss/n_steps, 
                DoF=avg_dof/n_steps, 
                sparsity=f"{stats['sparsity']:.2f}", 
                active_voxels=f"{stats['active_voxel_count']:.2f}"
            )

    # --- EVALUATION ---
    verify_results_aggregate_stoc(params, test_data, dmin, dmax, model, args.sparsity)
    
    DATASET_NAME = args.data.split("/")[-2]
    print(f"\nEvaluating against PT test split...")
    run_dynamic_benchmark_stoc(
        params, test_data, dmin, dmax, model, args.sparsity, 
        effective_avg_dof=avg_dof/n_steps, 
        avg_vram_mb=avg_mem_used/n_steps, 
        peak_vram_mb=peak_mem_used, 
        param_key=args.param, 
        grid_res=args.grid,
        dataset_name=DATASET_NAME,
        obj_out_dir=args.obj_out_dir
    )

if __name__ == "__main__":
    main()