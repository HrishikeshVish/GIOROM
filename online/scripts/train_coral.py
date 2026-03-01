import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import h5py
from tqdm import tqdm

try:
    from torchdiffeq import odeint
except ImportError:
    print("Error: torchdiffeq not installed. Run 'pip install torchdiffeq'")
    exit(1)

# Import from our modular codebase
from src.baselines.coral import Derivative

# ==========================================
# 1. IN-MEMORY DATASET (Auto-Extraction)
# ==========================================
class InMemoryCROMDataset(Dataset):
    def __init__(self, data_root, encoder_path, seq_len=50, sparsity=1, device='cuda'):
        self.seq_len = seq_len
        self.device = device
        self.data = []   
        self.valid_indices = [] 
        
        print(f"\n--- Initializing Hybrid Dataset ---")
        print(f"1. Loading CROM Encoder from: {encoder_path}")
        self.encoder = torch.jit.load(encoder_path, map_location=device).eval()
        
        sim_folders = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
        if not sim_folders:
            if glob.glob(os.path.join(data_root, "h5_f_*.h5")):
                sim_folders = [data_root]
            else:
                raise ValueError(f"No data found in {data_root}")

        print(f"2. Processing {len(sim_folders)} trajectories...")
        total_frames = 0
        
        for traj_idx, folder in enumerate(sim_folders):
            files = sorted(glob.glob(os.path.join(folder, "h5_f_*.h5")))
            if len(files) < seq_len:
                continue
                
            traj_latents = []
            with torch.no_grad():
                for fpath in tqdm(files, desc=f"Encoding {os.path.basename(folder)}", leave=False):
                    with h5py.File(fpath, 'r') as f:
                        if 'q' in f: x = torch.from_numpy(f['q'][:].T).float()
                        elif 'x' in f: x = torch.from_numpy(f['x'][:].T).float()
                        else: continue
                        
                        x = x.to(device)
                        if sparsity > 1:
                            inp = x[::sparsity].unsqueeze(0) 
                        else:
                            inp = x.unsqueeze(0)
                            
                        z = self.encoder(inp) 
                        traj_latents.append(z.view(-1).cpu()) 
            
            traj_tensor = torch.stack(traj_latents)
            self.data.append(traj_tensor)
            
            num_valid_starts = len(traj_tensor) - seq_len
            for t in range(num_valid_starts):
                self.valid_indices.append((traj_idx, t))
                
            total_frames += len(traj_tensor)

        self.latent_dim = self.data[0].shape[-1]
        print(f"--- Dataset Ready ---")
        print(f"Total Frames: {total_frames} | Latent Dim: {self.latent_dim}")
        print(f"Valid Sequences: {len(self.valid_indices)}")
        
        del self.encoder
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, start_frame = self.valid_indices[idx]
        z_seq = self.data[traj_idx][start_frame : start_frame + self.seq_len]
        return z_seq.float() 

# ==========================================
# 2. ODE SOLVER HELPERS
# ==========================================
def scheduling(_int, _f, true_codes, t, epsilon, method='euler'):
    if epsilon < 1e-3:
        epsilon = 0
    if epsilon == 0:
        codes = _int(_f, y0=true_codes[0], t=t, method=method)
    else:
        eval_points = np.random.random(len(t)) < epsilon
        eval_points[-1] = False
        eval_points = eval_points[1:]
        
        start_i, end_i = 0, None
        codes = []
        for i, eval_point in enumerate(eval_points):
            if eval_point == True:
                end_i = i + 1
                t_seg = t[start_i:end_i + 1]
                res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)
                
                if len(codes) == 0:
                    codes.append(res_seg)
                else:
                    codes.append(res_seg[1:])
                start_i = end_i
        
        t_seg = t[start_i:]
        res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)
        if len(codes) == 0:
            codes.append(res_seg)
        else:
            codes.append(res_seg[1:])
        codes = torch.cat(codes, dim=0)
    return codes

def ode_scheduling(model, z_gt, t_span, epsilon=0.0):
    z_gt_time_first = z_gt.permute(1, 0, 2)
    integrator_wrapper = lambda f, y0, t, method: odeint(f, y0, t, method=method, options={'step_size': 0.1})
    
    z_pred_time_first = scheduling(
        integrator_wrapper,
        model,
        true_codes=z_gt_time_first,
        t=t_span,
        epsilon=epsilon,
        method='euler'
    )
    return z_pred_time_first.permute(1, 0, 2)

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_hybrid(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = InMemoryCROMDataset(
        data_root=args.data_root, 
        encoder_path=args.enc, 
        seq_len=args.seq_len,
        sparsity=args.sparsity,
        device=device
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) 
    
    model = Derivative(state_c=1, code_c=dataset.latent_dim, hidden_c=args.hidden_dim, depth=args.depth).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    
    t_span = torch.linspace(0, args.seq_len-1, args.seq_len).to(device)
    
    print(f"\nStarting Dynamics Training ({args.epochs} epochs)...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for z_gt in tqdm(dataloader, leave=False, desc=f"Epoch {epoch}"):
            z_gt = z_gt.to(device) 
            
            optimizer.zero_grad()
            epsilon = max(0.0, 1.0 - (epoch / args.epochs)) 
            z_pred = ode_scheduling(model, z_gt, t_span, epsilon=epsilon)
            
            loss = nn.MSELoss()(z_pred, z_gt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
        
        if epoch % 1 == 0 or epoch == args.epochs - 1:
            dataset_name = os.path.basename(os.path.normpath(args.data_root))
            save_path = os.path.join(args.save_dir, f"hybrid_dynamics_epoch{epoch}_{args.sparsity}_{dataset_name}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'latent_dim': dataset.latent_dim,
                'args': args
            }, save_path)

    print(f"Training Complete. Final model saved to {args.save_dir}")

if __name__ == "__main__":
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Water")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_save = os.path.join(PROJECT_ROOT, "checkpoints", "coral_dynamics")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=default_data)
    parser.add_argument('--enc', type=str, default=default_enc)
    parser.add_argument('--save_dir', type=str, default=default_save)
    parser.add_argument('--sparsity', type=int, default=30)
    
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=50) 
    
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_hybrid(args)