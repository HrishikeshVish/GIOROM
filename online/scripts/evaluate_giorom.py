import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm

from src.utils.eval_utils import chamfer_distance
from src.visualizations.rendering import save_obj_jax

def verify_results_aggregate_stoc(params, test_data, dmin, dmax, model, sparsity_factor):
    print("\n--- AGGREGATE AUDIT (Stochastic) ---")
    audit_key = jax.random.PRNGKey(999)
    traj = test_data[0]
    n_frames = len(traj)
    
    x_d_ref = jnp.array(traj[0])
    x_s_ref = x_d_ref[::int(sparsity_factor)]
    
    @jax.jit
    def eval_step(x_s_c, k): 
        return model.apply(params, x_d_ref, x_s_ref, x_s_c, rngs={'feynman_kac': k})
    
    total_mse = 0.0
    keys = jax.random.split(audit_key, n_frames)

    for i in tqdm(range(n_frames)):
        x_d_curr = jnp.array(traj[i])
        x_s_curr = x_d_curr[::int(sparsity_factor)]
        
        pred = eval_step(x_s_curr, keys[i])
        frame_mse = jnp.mean(((x_d_curr - pred)*(dmax-dmin))**2)
        total_mse += frame_mse
        
    avg_agg_mse = total_mse / n_frames
    print(f"\n>>> AGGREGATE ROLLOUT MSE: {avg_agg_mse:.4e} <<<")

def run_dynamic_benchmark_stoc(params, test_data, dmin, dmax, model, sparsity_factor, effective_avg_dof, avg_vram_mb, peak_vram_mb, param_key, grid_res, dataset_name, obj_out_dir=None):
    key = f'{param_key}_{grid_res}' if param_key == 'Grid' else f'sampling_{sparsity_factor}'
    
    print(f"\n--- BENCHMARKING GIOROM (JAX - Stochastic) on {dataset_name} ---")
    test_key = jax.random.PRNGKey(42)

    @jax.jit
    def inference_step(x_q_ref, x_s_ref, x_s_curr, rng_key): 
        return model.apply(params, x_q_ref, x_s_ref, x_s_curr, rngs={'feynman_kac': rng_key}) 

    results = []
    
    if obj_out_dir:
        os.makedirs(obj_out_dir, exist_ok=True)
        print(f"Exporting .obj files to: {obj_out_dir}")

    preds = []
    num_trajs = test_data.shape[0]

    for idx in range(num_trajs):
        traj = test_data[idx]
        seq_name = f"sim_seq_{idx:06d}"
        
        x_ref_norm = jnp.array(traj[0])
        N = x_ref_norm.shape[0]
        indices = np.arange(0, N, step=int(sparsity_factor))
        x_s_ref_norm = x_ref_norm[indices]
        
        traj_l2, traj_times = [], []
        
        for i in tqdm(range(traj.shape[0]), desc=f"Seq {seq_name}"):
            x_curr_norm = jnp.array(traj[i])
            x_s_curr_norm = x_curr_norm[indices]
            
            # Denormalize GT for metric scale
            x_curr_phys = x_curr_norm * (dmax - dmin) + dmin
            
            test_key, subkey = jax.random.split(test_key)
            
            t0 = time.perf_counter()
            pred_norm = inference_step(x_ref_norm, x_s_ref_norm, x_s_curr_norm, subkey).block_until_ready()
            t1 = time.perf_counter()
            
            # Denormalize Prediction
            pred_phys = pred_norm * (dmax - dmin) + dmin
            preds.append(pred_phys)
            
            # Export 3D Mesh for Blender (only for first trajectory to save disk space)
            if idx == 0 and obj_out_dir:
                save_obj_jax(pred_phys, i, obj_out_dir, prefix="pred")
                save_obj_jax(x_curr_phys, i, obj_out_dir, prefix="gt") # Save GT just in case

            # Compute Metrics
            cd = chamfer_distance(pred_phys, x_curr_phys, samples=8192).item()
            err = jnp.linalg.norm((pred_phys - x_curr_phys).ravel())
            gt_n = jnp.linalg.norm(x_curr_phys.ravel())
            traj_l2.append(err / (gt_n + 1e-8))
            traj_times.append((t1 - t0) * 1000.0)

        results.append({
            'Sequence': seq_name,
            'RelL2_Mean': np.mean(traj_l2),
            'RelL2_Std': np.std(traj_l2),
            'Chamfer_Mean': np.mean(cd),
            'Time_ms': np.mean(traj_times),
            'Chamfer_Std': np.std(cd),
        })

    df = pd.DataFrame(results)
    avg_l2, std_l2 = df['RelL2_Mean'].mean(), df['RelL2_Std'].mean() 
    avg_time = df['Time_ms'].mean()
    avg_chamfer, std_chamfer = df['Chamfer_Mean'].mean(), df['Chamfer_Std'].mean()
    
    df['vram_MB'] = avg_vram_mb
    df['peak_vram_MB'] = peak_vram_mb
    
    csv_path = os.path.join(PROJECT_ROOT, "results", "raw_metrics", f"{dataset_name}_giorom_{key}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"\nSaved benchmark results to {csv_path}")
    print("\nLaTeX Row:")
    print(f"GIOROM & {avg_l2:.2%} $\\pm$ {std_l2:.2%} & {avg_time:.1f} & {avg_vram_mb:.0f} & {avg_chamfer:.5e} $\\pm$ {std_chamfer:.3e} \\\\")