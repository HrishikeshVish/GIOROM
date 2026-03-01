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
from src.baselines.pca import GappyPCA
from src.utils.data_utils import HDF5TrajectoryLoader
from src.utils.system_utils import get_torch_vram_usage_mb

# --- 1. LOCAL METRICS ---
def calculate_relative_l2(pred, gt, eps=1e-8):
    error_norm = torch.norm(pred - gt, p=2)
    gt_norm = torch.norm(gt, p=2)
    return error_norm / (gt_norm + eps)

# --- 2. VISUALIZATION ENGINE ---
def generate_fair_pca_video(model, test_traj, sparsity, output_name):
    frames = len(test_traj)
    print(f"Generating PCA Video for {frames} frames...")
    
    flat = test_traj[0].cpu().numpy()
    lim_min, lim_max = flat.min(), flat.max()
    margin = (lim_max - lim_min) * 0.1
    lim_min -= margin; lim_max += margin
    
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    titles = [f"Sparse Input (1/{sparsity})", "Ground Truth", "Fair PCA (Unseen Data)"]
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
    
    N, D = test_traj[0].shape
    indices = torch.arange(0, N, step=int(sparsity), device=test_traj[0].device)

    def update(frame_idx):
        x_gt = test_traj[frame_idx]
        x_sparse = x_gt[indices]
        
        # Solve Gappy POD
        x_pred = model.predict(x_sparse, indices, N, D)
        
        mse = torch.nn.functional.mse_loss(x_pred, x_gt).item()
        rel_l2 = calculate_relative_l2(x_pred, x_gt).item()
        vram = get_torch_vram_usage_mb()
        
        scatters[0].set_offsets(x_sparse.cpu().numpy()[:, :2])
        scatters[1].set_offsets(x_gt.cpu().numpy()[:, :2])
        scatters[2].set_offsets(x_pred.cpu().numpy()[:, :2])
        
        hud_text.set_text(f"Frame: {frame_idx:03d} | VRAM: {vram:.0f}MB | MSE: {mse:.2e} | Rel L2: {rel_l2:.2%}")
        return scatters + [hud_text]

    print("Rendering...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
    out_path = os.path.join(PROJECT_ROOT, "visualizations", "media", output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, fps=30, dpi=100, writer='ffmpeg')
    print(f"Saved to {out_path}")
    plt.close(fig)

# --- 3. MAIN ---
def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Plasticine")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data, help='Path to dataset')
    parser.add_argument('-latent', type=int, default=20, help='Number of PCA components')
    parser.add_argument('-sparsity', type=float, default=3.5)
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()
    
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=dev)
    
    # 1. Load Data
    loader = HDF5TrajectoryLoader(args.data, dev)
    num_trajs = len(loader.sim_folders)
    
    if num_trajs < 2:
        raise ValueError("Need at least 2 trajectories to split train/test for PCA!")
        
    num_train = num_trajs - 1
    
    # 2. Extract Training Matrix (Phase 1)
    print("\n--- Phase 1: Constructing Snapshot Matrix ---")
    all_train_snaps = []
    for i in range(num_train):
        # We don't need the reference trajectories for PCA, just the target states
        _, t_curr = loader.get_trajectory(sim_idx=i, max_frames=200)
        # Stack the list of (N, D) tensors into (Frames, N, D)
        all_train_snaps.append(torch.stack(t_curr))
        
    train_snaps = torch.cat(all_train_snaps, dim=0)
    
    # 3. Fit PCA
    print("\n--- Phase 2: PCA Training ---")
    model = GappyPCA(latent_dim=args.latent, device=dev)
    model.fit(train_snaps)
    
    # Free up memory before testing
    del train_snaps
    del all_train_snaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 4. Evaluate on Hold-Out (Phase 3)
    print("\n--- Phase 3: PCA Evaluation ---")
    _, test_traj = loader.get_trajectory(sim_idx=-1, max_frames=200) # -1 gets the last folder
    
    generate_fair_pca_video(model, test_traj, args.sparsity, "pca_fair_results.mp4")

if __name__ == "__main__":
    main()