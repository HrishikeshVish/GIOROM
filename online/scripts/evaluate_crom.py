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
from src.baselines.crom import CROM_Inference
from src.utils.data_utils import HDF5TrajectoryLoader
from src.utils.system_utils import get_torch_vram_usage_mb

# --- 1. LOCAL METRICS ---
def calculate_relative_l2(pred, gt, eps=1e-8):
    error_norm = torch.norm(pred - gt, p=2)
    gt_norm = torch.norm(gt, p=2)
    return error_norm / (gt_norm + eps)

# --- 2. VISUALIZATION ENGINE ---
def generate_video(model, x_ref_list, x_curr_list, output_name="crom_vis.mp4"):
    frames = len(x_curr_list)
    print(f"\n--- Generating Video ({frames} frames) ---")
    
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    
    axes = []
    titles = ["Input (Ground Truth)", "Ground Truth", "CROM Prediction"]
    colors = ['#00FFFF', '#00FF00', '#FF00FF']
    
    flat_data = x_curr_list[0].cpu().numpy()
    dmin, dmax = flat_data.min(), flat_data.max()
    margin = (dmax - dmin) * 0.1
    lim_min, lim_max = dmin - margin, dmax + margin

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(lim_min, lim_max) 
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(titles[i], color=colors[i], fontsize=14, fontweight='bold', pad=10)
        axes.append(ax)

    scatters = []
    for i, col in enumerate(colors):
        sc = axes[i].scatter([], [], s=2.0, alpha=0.7, c=col, edgecolors='none')
        scatters.append(sc)

    text_ax = fig.add_subplot(gs[1, :])
    text_ax.set_facecolor('#0f0f0f')
    text_ax.axis('off')
    
    hud_prop = dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='#FF00FF')
    hud_text = text_ax.text(0.5, 0.5, "Initializing...", ha='center', va='center', 
                            color='white', fontfamily='monospace', fontsize=12, bbox=hud_prop)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def update(frame_idx):
        x_ref = x_ref_list[frame_idx]
        x_gt = x_curr_list[frame_idx]
        
        start_event.record()
        x_pred = model.predict(x_ref, x_gt)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        mse = torch.nn.functional.mse_loss(x_pred, x_gt).item()
        rel_l2 = calculate_relative_l2(x_pred, x_gt).item()
        vram = get_torch_vram_usage_mb()
        
        gt_np = x_gt.cpu().numpy()
        pred_np = x_pred.cpu().numpy()
        
        scatters[0].set_offsets(gt_np[:, :2])
        scatters[1].set_offsets(gt_np[:, :2])
        scatters[2].set_offsets(pred_np[:, :2])
        
        status = (f"Frame: {frame_idx:03d} | Inf Time: {elapsed_ms:5.2f} ms | "
                  f"VRAM: {vram:5.0f} MB | "
                  f"MSE: {mse:.2e} | Rel L2: {rel_l2:.2%}")
        hud_text.set_text(status)
        
        return scatters + [hud_text]

    print("Rendering...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
    out_path = os.path.join(PROJECT_ROOT, "visualizations", "media", output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, fps=30, dpi=100, writer='ffmpeg')
    print(f"Saved to {out_path}")
    plt.close(fig)

# --- 3. MAIN EVALUATION LOOP ---
def main():
    # Set default paths relative to PROJECT_ROOT
    default_data = os.path.join(PROJECT_ROOT, "data", "owl")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_dec = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "decoder.pt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data, help='Path to dataset')
    parser.add_argument('-enc', type=str, default=default_enc, help='Path to encoder.pt')
    parser.add_argument('-dec', type=str, default=default_dec, help='Path to decoder.pt')
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        
    loader = HDF5TrajectoryLoader(args.data, device)
    x_ref_list, x_curr_list = loader.get_trajectory(sim_idx=0, max_frames=200)
    
    if not x_ref_list:
        print("No data loaded. Exiting.")
        return

    SPARSITY_FACTOR = 24
    model = CROM_Inference(args.enc, args.dec, device, sparsity=SPARSITY_FACTOR)
    
    generate_video(model, x_ref_list, x_curr_list, output_name="crom_results_metrics.mp4")

if __name__ == "__main__":
    main()