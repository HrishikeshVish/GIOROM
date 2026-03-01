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
import glob
import h5py
from tqdm import tqdm

from src.baselines.colora import CoLoRA_Dynamics, HyperNetwork

# ==========================================
# 1. DATASET: Auto-Extraction (In-Memory)
# ==========================================
class LatentCoLoRADataset(Dataset):
    def __init__(self, data_root, encoder_path, sparsity=1, device='cuda'):
        self.data = [] 
        
        print(f"\n--- Initializing CoLoRA Dataset ---")
        print(f"Loading Encoder: {encoder_path}")
        encoder = torch.jit.load(encoder_path, map_location=device).eval()
        
        sim_folders = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
        if not sim_folders:
            if glob.glob(os.path.join(data_root, "h5_f_*.h5")):
                sim_folders = [data_root]
            else:
                raise ValueError(f"No data found in {data_root}")

        print(f"Processing {len(sim_folders)} trajectories...")
        num_sims = len(sim_folders)
        
        for sim_idx, folder in enumerate(sim_folders):
            files = sorted(glob.glob(os.path.join(folder, "h5_f_*.h5")))
            num_frames = len(files)
            if num_frames == 0: continue
            
            mu_val = sim_idx / max(1, num_sims - 1) 
            mu_vec = torch.tensor([mu_val], dtype=torch.float32)
            
            traj_z = []
            with torch.no_grad():
                for fpath in tqdm(files, desc=f"Encoding Sim {sim_idx}", leave=False):
                    with h5py.File(fpath, 'r') as f:
                        if 'q' in f: x = torch.from_numpy(f['q'][:].T).float()
                        else: x = torch.from_numpy(f['x'][:].T).float()
                        
                        if sparsity > 1: x = x[::sparsity]
                        
                        inp = x.to(device).unsqueeze(0) 
                        z = encoder(inp).cpu() 
                        traj_z.append(z.view(-1))
            
            for t_idx, z in enumerate(traj_z):
                t_norm = t_idx / max(1, num_frames - 1)
                self.data.append({
                    't': torch.tensor([t_norm], dtype=torch.float32),
                    'mu': mu_vec,
                    'z': z
                })
        
        self.latent_dim = self.data[0]['z'].shape[0]
        self.param_dim = 1 
        
        print(f"--- Dataset Ready ---")
        print(f"Total Samples: {len(self.data)} | Latent Dim: {self.latent_dim}")
        
        del encoder
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 2. TRAINING ENGINE
# ==========================================
def train_colora(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = LatentCoLoRADataset(args.data_root, args.enc, args.sparsity, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    depth = args.depth
    num_layers_dynamics = depth + 1 
    
    dynamics = CoLoRA_Dynamics(
        latent_dim_crom=dataset.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=depth,
        rank=args.rank
    ).to(device)
    
    hypernet = HyperNetwork(
        param_dim=dataset.param_dim,
        num_target_layers=num_layers_dynamics,
        rank=args.rank
    ).to(device)
    
    optimizer = optim.Adam(list(dynamics.parameters()) + list(hypernet.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    
    print(f"\nStarting CoLoRA Training ({args.epochs} epochs)...")
    
    save_path = ""
    for epoch in range(args.epochs):
        dynamics.train(); hypernet.train()
        total_loss = 0
        
        for batch in dataloader:
            t = batch['t'].to(device)   
            mu = batch['mu'].to(device) 
            z_gt = batch['z'].to(device)
            
            optimizer.zero_grad()
            
            alphas = hypernet(mu)
            z_pred = dynamics(t, alphas)
            
            loss = nn.MSELoss()(z_pred, z_gt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            dataset_name = os.path.basename(os.path.normpath(args.data_root))
            save_path = os.path.join(args.save_dir, f"colora_online_epoch{epoch}_{args.sparsity}_{dataset_name}.pt")
            torch.save({
                'dynamics_state': dynamics.state_dict(),
                'hypernet_state': hypernet.state_dict(),
                'args': args,
                'latent_dim': dataset.latent_dim,
                'param_dim': dataset.param_dim
            }, save_path)
            
    print(f"Training Done. Saved to {save_path}")

if __name__ == "__main__":
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Water")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_save = os.path.join(PROJECT_ROOT, "checkpoints", "colora")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=default_data)
    parser.add_argument('--enc', type=str, default=default_enc)
    parser.add_argument('--save_dir', type=str, default=default_save)
    parser.add_argument('--sparsity', type=int, default=30)
    
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--rank', type=int, default=32)
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_colora(args)