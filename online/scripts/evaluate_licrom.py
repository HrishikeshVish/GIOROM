import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Import from our modular codebase
from src.baselines.licrom import LiCROM_Inference
from src.utils.data_utils import HDF5TrajectoryLoader
from src.utils.system_utils import get_torch_vram_usage_mb

# --- 1. LOCAL METRICS ---
def calculate_relative_l2(pred, gt, eps=1e-8):
    error_norm = torch.norm(pred - gt, p=2)
    gt_norm = torch.norm(gt, p=2)
    return error_norm / (gt_norm + eps)

# --- 2. VISUALIZATION ENGINE ---
def generate_licrom_pure_video(model, x_ref, x_traj, sparsity, output_name):
    print(f"Generating LiCROM (Gauss-Newton) Video for {len(x_traj)} frames...")
    
    N = x_ref.shape[0]
    indices = torch.arange(0, N, step=int(sparsity), device=model.device)
    
    # Initialize z with first frame using Encoder (Warm Start)
    try:
        z_curr = model.encoder(x_traj[0].unsqueeze(0))
    except:
        z_curr = model.encoder(x_traj[0].unsqueeze(0).permute(0,2,1))
    
    # Ensure (1, 1, latent) for the loop
    z_curr = z_curr.view(1, 1, -1) 
        
    flat = x_traj[0].cpu().numpy()
    lim_min, lim_max = flat.min(), flat.max()
    margin = (lim_max - lim_min) * 0.1
    lim_min -= margin; lim_max += margin
    
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    titles = [f"Sparse Input (1/{sparsity})", "Ground Truth", "LiCROM (Gauss-Newton)"]
    colors = ['#00FFFF', '#00FF00', '#FF00FF']
    
    axes, scatters = [], []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('black'); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
        ax.set_title(titles[i], color=colors[i], fontsize=14, fontweight='bold')
        axes.append(ax)
        s = 3.0 if i == 0 else 1.5
        sc = ax.scatter([], [], s=s, alpha=0.7, c=colors[i], edgecolors='none')
        scatters.append(sc)

    text_ax = fig.add_subplot(gs[1, :])
    text_ax.set_facecolor('#0f0f0f'); text_ax.axis('off')
    hud_text = text_ax.text(0.5, 0.5, "Init...", ha='center', va='center', color='white', 
                            fontsize=12, fontfamily='monospace', bbox=dict(facecolor='#1a1a1a', edgecolor='white'))
    
    def update(frame_idx):
        nonlocal z_curr
        x_gt = x_traj[frame_idx]
        x_sparse = x_gt[indices]
        
        # LiCROM Step: Gauss-Newton Optimization
        z_curr = model.solve_gauss_newton(x_ref, x_sparse, indices, z_init=z_curr, steps=3)
        
        # Decode Full
        x_pred = model.decode_full(z_curr, x_ref)
        
        # Metrics
        mse = torch.nn.functional.mse_loss(x_pred, x_gt).item()
        error_norm = torch.norm(x_pred - x_gt, p=2)
        gt_norm = torch.norm(x_gt, p=2)
        rel = (error_norm / (gt_norm + 1e-8)).item()
        vram = get_torch_vram_usage_mb()
        
        scatters[0].set_offsets(x_sparse.cpu().numpy()[:, :2])
        scatters[1].set_offsets(x_gt.cpu().numpy()[:, :2])
        scatters[2].set_offsets(x_pred.detach().cpu().numpy()[:, :2])
        
        hud_text.set_text(f"Frame: {frame_idx:03d} | VRAM: {vram:.0f}MB | MSE: {mse:.2e} | Rel L2: {rel:.2%}")
        return scatters + [hud_text]

    print("Rendering...")
    ani = animation.FuncAnimation(fig, update, frames=len(x_traj), interval=1, blit=True)
    out_path = os.path.join(PROJECT_ROOT, "visualizations", "media", output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, fps=30, dpi=100, writer='ffmpeg')
    print(f"Saved to {out_path}")

# --- 3. MAIN ---
def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Plasticine")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_dec = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "decoder.pt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data, help='Path to dataset')
    parser.add_argument('-enc', type=str, default=default_enc, help='Path to encoder.pt')
    parser.add_argument('-dec', type=str, default=default_dec, help='Path to decoder.pt')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-sparsity', type=float, default=3.5)
    args = parser.parse_args()
    
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=dev)
    
    # 1. LOAD DATA
    loader = HDF5TrajectoryLoader(args.data, dev)
    traj_ref, traj_curr = loader.get_trajectory(sim_idx=0, max_frames=200)
    
    if not traj_curr:
        raise ValueError("Trajectory load failed")
    
    x_ref = traj_curr[0]
    
    # 2. INIT MODEL
    model = LiCROM_Inference(args.enc, args.dec, dev, sample_input=x_ref)
    
    # 3. GENERATE
    generate_licrom_pure_video(model, x_ref, traj_curr, args.sparsity, "licrom_pure_results.mp4")

if __name__ == "__main__":
    main()