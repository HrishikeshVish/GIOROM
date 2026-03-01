import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import torch
import numpy as np
import h5py
import glob
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import from modular codebase
from src.utils.system_utils import MemoryTracker
from src.visualizations.rendering import save_obj_jax as save_obj_frame # Fallback if torch isn't supported, otherwise write a quick torch wrapper

# ==========================================
# 1. UTILITIES
# ==========================================
def chamfer_distance(pred, gt, samples=1000):
    N = pred.shape[0]
    if samples > 0 and N > samples:
        idx_pred = torch.randperm(N)[:samples]
        idx_gt = torch.randperm(N)[:samples]
        pred_sub = pred[idx_pred]
        gt_sub = gt[idx_gt]
    else:
        pred_sub = pred
        gt_sub = gt

    x = pred_sub.unsqueeze(1) 
    y = gt_sub.unsqueeze(0)   
    dist_sq = torch.sum((x - y) ** 2, dim=-1)
    return dist_sq.min(dim=1)[0].mean() + dist_sq.min(dim=0)[0].mean()

def calculate_metrics(pred, gt):
    mse = torch.nn.functional.mse_loss(pred, gt).item()
    error_norm = torch.norm(pred - gt, p=2)
    gt_norm = torch.norm(gt, p=2)
    rel_l2 = (error_norm / (gt_norm + 1e-8)).item()
    return mse, rel_l2

def get_calibration_sample(data_root, device, sparsity=1):
    seqs = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
    if not seqs: seqs = [data_root]
    files = sorted(glob.glob(os.path.join(seqs[0], "h5_f_*.h5")))
    if not files: raise ValueError(f"No .h5 files")
    with h5py.File(files[0], 'r') as f:
        dense = torch.from_numpy(f['x'][:].T).float().to(device)
    if int(sparsity) > 1: return dense[::int(sparsity)]
    return dense

# ==========================================
# 2. MODEL WRAPPER
# ==========================================
class ModelWrapper:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model_type = args.model_type
        print(f"Initializing {self.model_type.upper()} on {device}...")
        
        if self.model_type in ['licrom', 'dino']:
            self.calibration_sample = get_calibration_sample(args.data_root, device, args.sparsity)
        
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type == 'pca':
            from src.baselines.pca import GappyPCA
            from src.utils.data_utils import HDF5TrajectoryLoader
            loader = HDF5TrajectoryLoader(self.args.data_root, self.device)
            
            all_train_snaps = []
            for i in range(min(3, max(1, len(loader.sim_folders) - 1))): 
                _, t_curr = loader.get_trajectory(sim_idx=i, max_frames=200)
                all_train_snaps.append(torch.stack(t_curr))
                
            train_snaps = torch.cat(all_train_snaps, dim=0)
            pca_model = GappyPCA(latent_dim=self.args.latent_dim, device=self.device)
            pca_model.fit(train_snaps)
            return pca_model

        elif self.model_type == 'gno':
            from src.baselines.gno import GNO
            model = GNO(dim=3, gno_radius=self.args.radius).to(self.device)
            model.load_state_dict(torch.load(self.args.gno_ckpt, map_location=self.device))
            model.eval()
            return model

        elif self.model_type == 'licrom':
            from src.baselines.licrom import LiCROM_Inference
            return LiCROM_Inference(self.args.enc, self.args.dec, self.device, self.calibration_sample)

        elif self.model_type == 'dino':
            from src.baselines.dino import DINoEvaluator
            return DINoEvaluator(self.args.enc, self.args.dec, self.args.ckpt, self.device, self.calibration_sample)
            
        elif self.model_type == 'crom': 
            from src.baselines.crom import CROM_Inference
            return CROM_Inference(self.args.enc, self.args.dec, self.device, sparsity=self.args.sparsity)
            
        elif self.model_type == 'coral':
            from src.baselines.coral import CORAL_Wrapper
            return CORAL_Wrapper(self.args, self.device)
            
        elif self.model_type == 'colora':
            from src.baselines.colora import CoLoRA_Wrapper
            return CoLoRA_Wrapper(self.args, self.device)

        else: raise ValueError(f"Unknown: {self.model_type}")

    def run_inference(self, x_ref, x_curr, indices):
        x_sparse = x_curr[indices]
        N = x_ref.shape[0]
        
        if self.model_type == 'pca':
            return self.model.predict(x_sparse, indices, N, 3)
        elif self.model_type == 'gno':
            f_sparse = x_sparse - x_ref[indices]
            with torch.no_grad():
                pred_disp = self.model(x_ref[indices], f_sparse, x_ref)
            return x_ref + pred_disp
        elif self.model_type == 'licrom':
            try: z_init = self.model.encoder(x_sparse.unsqueeze(0))
            except: z_init = self.model.encoder(x_sparse.unsqueeze(0).permute(0,2,1))
            if z_init.dim() == 2: z_init = z_init.unsqueeze(1)
            z_opt = self.model.solve_gauss_newton(x_ref, x_sparse, indices, z_init, steps=3)
            return self.model.decode_full(z_opt, x_ref)
        elif self.model_type == 'crom':
            return self.model.predict(x_ref, x_curr)
        return x_curr

# ==========================================
# 3. VIDEO GENERATION UTILS
# ==========================================
def generate_video(wrapper, files, x_ref, indices, output_name="vis.mp4"):
    frames = len(files)
    print(f"\n--- Generating Video ({frames} frames) for {wrapper.model_type} ---")
    
    with h5py.File(files[0], 'r') as f:
        if 'q' in f: flat_data = f['q'][:].T
        else: flat_data = f['x'][:].T
    
    fig = plt.figure(figsize=(18, 7), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.1)
    
    axes = []
    titles = ["Input (Sparsified)", "Ground Truth", f"{wrapper.model_type.upper()} Prediction"]
    colors = ['#00FFFF', '#00FF00', '#FF00FF']
    
    dmin, dmax = flat_data.min(), flat_data.max()
    margin = (dmax - dmin) * 0.1
    lim_min, lim_max = dmin - margin, dmax + margin

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('black'); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
        ax.set_title(titles[i], color=colors[i], fontsize=14, fontweight='bold', pad=10)
        axes.append(ax)

    scatters = []
    for i, col in enumerate(colors):
        sc = axes[i].scatter([], [], s=2.0, alpha=0.7, c=col, edgecolors='none')
        scatters.append(sc)

    text_ax = fig.add_subplot(gs[1, :])
    text_ax.set_facecolor('#0f0f0f'); text_ax.axis('off')
    
    hud_prop = dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='#FF00FF')
    hud_text = text_ax.text(0.5, 0.5, "Initializing...", ha='center', va='center', 
                            color='white', fontfamily='monospace', fontsize=12, bbox=hud_prop)

    class State: z = None
    state = State()

    def update(frame_idx):
        fpath = files[frame_idx]
        with h5py.File(fpath, 'r') as f:
            if 'q' in f: x_curr = torch.from_numpy(f['q'][:].T).float().to(wrapper.device)
            else: x_curr = torch.from_numpy(f['x'][:].T).float().to(wrapper.device)
        
        t0 = time.time()
        
        if wrapper.model_type == 'dino':
            if frame_idx == 0:
                state.z = wrapper.model.get_initial_z(x_ref, x_curr[indices], indices)
                pred = wrapper.model.decode(state.z, x_ref)
            else:
                z_in = state.z.squeeze(1)
                with torch.no_grad():
                    dz = wrapper.model.ode_func(0, z_in)
                    state.z = (z_in + dz * 0.01).unsqueeze(1)
                pred = wrapper.model.decode(state.z, x_ref)
                
        elif wrapper.model_type == 'coral':
            if frame_idx == 0:
                state.z = wrapper.model.get_initial_z(x_ref, x_curr[indices], indices)
                pred = wrapper.model.decode(state.z, x_ref)
            else:
                state.z = wrapper.model.step_ode(frame_idx - 1, state.z)
                pred = wrapper.model.decode(state.z, x_ref)

        elif wrapper.model_type == 'colora':
            pred = wrapper.model.predict(
                frame_idx=frame_idx, total_frames=frames, 
                sim_idx=0, total_sims=1, x_ref=x_ref
            )

        else:
            pred = wrapper.run_inference(x_ref, x_curr, indices)

        elapsed_ms = (time.time() - t0) * 1000
        mse, rel_l2 = calculate_metrics(pred, x_curr)
        
        gt_np = x_curr.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        sparse_np = gt_np[indices.cpu().numpy()] if indices is not None else gt_np
        
        scatters[0].set_offsets(sparse_np[:, :2])
        scatters[1].set_offsets(gt_np[:, :2])
        scatters[2].set_offsets(pred_np[:, :2])
        
        status = (f"Frame: {frame_idx:03d} | Inf Time: {elapsed_ms:5.2f} ms | "
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

# ==========================================
# 4. BENCHMARK ENGINE
# ==========================================
def run_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper = ModelWrapper(args, device)
    
    traj_folders = sorted(glob.glob(os.path.join(args.data_root, "sim_seq_*")))
    if not traj_folders: traj_folders = [args.data_root]
        
    print(f"\n--- BENCHMARKING {args.model_type.upper()} ---")
    results = []
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for sim_idx, folder in enumerate(traj_folders):
        total_sims = len(traj_folders)
        seq_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, "h5_f_*.h5")))[:200]
        num_frames = len(files)
        if len(files) < 10: continue
        print(f"Processing {seq_name}...")
        
        with h5py.File(files[0], 'r') as f:
            x_ref = torch.from_numpy(f['x'][:].T).float().to(device)
            indices = torch.arange(0, x_ref.shape[0], step=int(args.sparsity), device=device)
        
        state_z = None
        traj_metrics = {'mse': [], 'rel_l2': [], 'chamfer': [], 'time': [], 'ram_mb': [], 'peak_ram_mb': []}
        track_device = 'cpu' if args.model_type == 'pca' else 'cuda'
        tracker = MemoryTracker(device_type=track_device)
        avg_mem = peak_mem = 0.0
        
        for i, fpath in enumerate(tqdm(files)):
            with h5py.File(fpath, 'r') as f:
                x_curr = torch.from_numpy(f['q'][:].T).float().to(device)
            
            tracker.start()
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if torch.cuda.is_available(): start_event.record()
            t0 = time.time()
            
            # --- INFERENCE ---
            if args.model_type == 'dino':
                if i == 0:
                    state_z = wrapper.model.get_initial_z(x_ref, x_curr[indices], indices)
                    pred = wrapper.model.decode(state_z, x_ref)
                else:
                    z_in = state_z.squeeze(1)
                    with torch.no_grad():
                        dz = wrapper.model.ode_func(0, z_in)
                        state_z = (z_in + dz * 0.01).unsqueeze(1)
                    pred = wrapper.model.decode(state_z, x_ref)
                    
            elif args.model_type == 'coral':
                if i == 0:
                    state_z = wrapper.model.get_initial_z(x_ref, x_curr[indices], indices)
                    pred = wrapper.model.decode(state_z, x_ref)
                else:
                    state_z = wrapper.model.step_ode(t=i-1, z_prev=state_z)
                    pred = wrapper.model.decode(state_z, x_ref)
                    
            elif args.model_type == 'colora':
                pred = wrapper.model.predict(
                    frame_idx=i, total_frames=num_frames,
                    sim_idx=sim_idx, total_sims=total_sims, x_ref=x_ref
                )
            else:
                pred = wrapper.run_inference(x_ref, x_curr, indices)
                
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                inf_time = start_event.elapsed_time(end_event)
            else:
                inf_time = (time.time() - t0) * 1000.0
                
            mem_stats = tracker.stop_and_report()

            # Save OBJ using our relative visualizations directory
            if args.save_obj and sim_idx == 0:
                # Use the path provided by the --obj_out_dir flag if available
                # Fallback to the default path if the flag is missing
                target_obj_dir = args.obj_out_dir if args.obj_out_dir else os.path.join(PROJECT_ROOT, "visualizations", "media", "obj_output")
                
                # We append the prefix "pred" to match the video stitcher's expectations
                save_obj_frame(pred, i, target_obj_dir, prefix="pred")

            if args.model_type == 'pca':
                avg_mem += mem_stats['peak_ram_mb']
                peak_mem = max(peak_mem, mem_stats['peak_ram_mb'])
            else:
                avg_mem += mem_stats['peak_vram_mb']
                peak_mem = max(peak_mem, mem_stats['peak_vram_mb'])
            
            mse, rel_l2 = calculate_metrics(pred, x_curr)
            cd = chamfer_distance(pred, x_curr, samples=4096).item()
            traj_metrics['mse'].append(mse)
            traj_metrics['rel_l2'].append(rel_l2)
            traj_metrics['chamfer'].append(cd)
            traj_metrics['time'].append(inf_time)
            traj_metrics['ram_mb'].append(avg_mem / (i + 1))
            traj_metrics['peak_ram_mb'].append(peak_mem)

        results.append({
            'Sequence': seq_name,
            'RelL2_Mean': np.mean(traj_metrics['rel_l2']),
            'RelL2_Std': np.std(traj_metrics['rel_l2']),
            'Chamfer_Mean': np.mean(traj_metrics['chamfer']),
            'Time_ms': np.mean(traj_metrics['time']),
            'avg_ram_mb': traj_metrics['ram_mb'][-1],
            'peak_ram_mb': np.max(traj_metrics['peak_ram_mb'])
        })

    df = pd.DataFrame(results)
    print("\n--- FINAL RESULTS ---")
    print(df)
    
    # Save CSV cleanly to results directory
    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    sparsity_str = int(args.latent_dim) if args.latent_dim.is_integer() else args.latent_dim
    csv_name = f"{dataset_name}_{args.model_type}_{sparsity_str}.csv"
    
    out_dir = os.path.join(PROJECT_ROOT, "results", "raw_metrics")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    print(f"Saved metrics to: {os.path.join(out_dir, csv_name)}")
    
    avg_l2 = df['RelL2_Mean'].mean()
    std_l2 = df['RelL2_Std'].mean()
    avg_time = df['Time_ms'].mean()
    avg_mem = df['avg_ram_mb'].mean()
    cd_mean = df['Chamfer_Mean'].mean()

    print("\nLaTeX Row:")
    print(f"{args.model_type.upper()} & {avg_l2:.2%} $\\pm$ {std_l2:.2%} & {avg_time:.1f} & {avg_mem:.0f} & {cd_mean:.5f} \\\\")
    
    if args.save_video and len(traj_folders) > 0:
        first_folder = traj_folders[0]
        files = sorted(glob.glob(os.path.join(first_folder, "h5_f_*.h5")))[:100]
        with h5py.File(files[0], 'r') as f:
            x_ref = torch.from_numpy(f['x'][:].T).float().to(device)
            indices = torch.arange(0, x_ref.shape[0], step=int(args.sparsity), device=device)
        
        vid_name = f"vis_{args.model_type}_{args.sparsity}_{dataset_name}.mp4"
        generate_video(wrapper, files, x_ref, indices, output_name=vid_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=str, required=True, choices=['pca', 'gno', 'licrom', 'dino', 'crom', 'coral', 'colora'])
    
    # Default Paths
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Water")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_dec = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "decoder.pt")
    default_colora = os.path.join(PROJECT_ROOT, "checkpoints", "colora", "colora_online.pt")
    default_coral_on = os.path.join(PROJECT_ROOT, "checkpoints", "coral", "hybrid_dynamics.pt")
    
    parser.add_argument('-data_root', type=str, default=default_data)
    parser.add_argument('-sparsity', type=float, default=30)
    
    # Model Paths
    parser.add_argument('-enc', type=str, default=default_enc)
    parser.add_argument('-dec', type=str, default=default_dec)
    parser.add_argument('-ckpt', type=str, default=default_colora, help="Used for CoLoRA or DINo")
    parser.add_argument('-gno_ckpt', type=str, default='')
    
    # CORAL Specific
    parser.add_argument('-coral_online_ckpt', type=str, default=default_coral_on)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-depth', type=int, default=3)

    parser.add_argument('--save_obj', action='store_true', help="Save .obj mesh files for Blender")
    
    parser.add_argument('-radius', type=float, default=0.055)
    parser.add_argument('-latent_dim', type=int, default=64)
    parser.add_argument('--save_video', action='store_true', default=True, help="Save a visualization video")
    parser.add_argument('--obj_out_dir', type=str, default=None, help="Output directory for .obj files")
    
    args = parser.parse_args()
    run_benchmark(args)