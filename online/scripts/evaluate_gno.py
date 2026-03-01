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

# Import from our modular codebase
from src.baselines.gno import GNO
from src.utils.data_utils import HDF5TrajectoryLoader
from src.utils.system_utils import get_torch_vram_usage_mb

# --- 1. LOCAL METRICS ---
def calculate_relative_l2(pred, gt, eps=1e-8):
    error_norm = torch.norm(pred - gt, p=2)
    gt_norm = torch.norm(gt, p=2)
    return error_norm / (gt_norm + eps)

# --- 2. VISUALIZATION ENGINE ---
def generate_gno_video(model, traj_ref, traj_curr, sparsity, output_name):
    print("Generating GNO Evaluation Video...")
    frames = len(traj_curr)
    
    # Get Bounds from the first frame
    q0 = traj_curr[0]
    flat = q0.cpu().numpy()
    lim_min, lim_max = flat.min(), flat.max()
    margin = (lim_max - lim_min) * 0.1
    lim_min -= margin; lim_max += margin
    
    # Setup Figure
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    titles = [f"Sparse Input (1/{sparsity})", "Ground Truth", "GNO Prediction"]
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

    # HUD
    text_ax = fig.add_subplot(gs[1, :])
    text_ax.set_facecolor('#0f0f0f'); text_ax.axis('off')
    hud_text = text_ax.text(0.5, 0.5, "Init...", ha='center', va='center', color='white', 
                            fontsize=12, fontfamily='monospace', bbox=dict(facecolor='#1a1a1a', edgecolor='white'))
    
    def update(frame_idx):
        x_ref = traj_ref[frame_idx]
        q_gt = traj_curr[frame_idx]
        
        # 1. Create Sparse Input
        indices = torch.arange(0, x_ref.shape[0], step=int(sparsity), device=model.device if hasattr(model, 'device') else x_ref.device)
        x_sp = x_ref[indices]
        q_sp = q_gt[indices]
        f_sp = q_sp - x_sp # Feature: Displacement
        
        # 2. Inference
        with torch.no_grad():
            pred_disp = model(x_sp, f_sp, x_ref)
            q_pred = x_ref + pred_disp
        
        # 3. Metrics
        mse = torch.nn.functional.mse_loss(q_pred, q_gt).item()
        rel_l2 = calculate_relative_l2(q_pred, q_gt).item()
        vram = get_torch_vram_usage_mb()
        
        # 4. Viz (Detach to fix error)
        scatters[0].set_offsets(q_sp.cpu().numpy()[:, :2])
        scatters[1].set_offsets(q_gt.cpu().numpy()[:, :2])
        scatters[2].set_offsets(q_pred.detach().cpu().numpy()[:, :2])
        
        hud_text.set_text(f"Frame: {frame_idx:03d} | VRAM: {vram:.0f}MB | MSE: {mse:.2e} | Rel L2: {rel_l2:.2%}")
        return scatters + [hud_text]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
    out_path = os.path.join(PROJECT_ROOT, "visualizations", "media", output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, fps=30, dpi=100, writer='ffmpeg')
    print(f"Saved to {out_path}")

# --- 3. MAIN ---
def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Plasticine")
    default_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "gno", "gno_epoch5.pt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data, help='Path to dataset')
    parser.add_argument('-ckpt', type=str, default=default_ckpt, help="Path to gno checkpoint")
    parser.add_argument('-radius', type=float, default=0.15, help="Must match training radius!") 
    parser.add_argument('-sparsity', type=float, default=3.5)
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
    
    # 1. Load Data (sim_idx=0 targets the hold-out sequence)
    loader = HDF5TrajectoryLoader(args.data, device)
    traj_ref, traj_curr = loader.get_trajectory(sim_idx=0, max_frames=200)
    
    if not traj_curr:
        raise ValueError("Trajectory load failed")
    
    # 2. Load Model
    print("Loading GNO Model...")
    model = GNO(dim=3, gno_radius=args.radius).to(device)
    # Store device on model for easy access in the viz loop
    model.device = device 
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # 3. Generate Video
    generate_gno_video(model, traj_ref, traj_curr, args.sparsity, "gno_results.mp4")

if __name__ == "__main__":
    main()