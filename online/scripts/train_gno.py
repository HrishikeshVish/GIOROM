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

# Import the architecture from our modular codebase
from src.baselines.gno import GNO

# ==========================================
# 1. DATASET WITH TRAIN/TEST SPLIT
# ==========================================
class GNODataset(Dataset):
    def __init__(self, data_root, sparsity=3.5, holdout_idx=0):
        """
        Loads all trajectories EXCEPT the holdout_idx.
        This ensures we train on N-1 sequences and test on 1.
        """
        self.sparsity = sparsity
        
        # 1. Find all sequence folders
        all_seqs = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
        if not all_seqs:
            raise ValueError(f"No sim_seq folders found in {data_root}")
            
        # 2. Exclude the Hold-Out Sequence
        self.train_seqs = [s for i, s in enumerate(all_seqs) if i != holdout_idx]
        
        print(f"Total Trajectories: {len(all_seqs)}")
        print(f"Holding out: {os.path.basename(all_seqs[holdout_idx])}")
        print(f"Training on: {len(self.train_seqs)} trajectories")
        
        # 3. Collect all files from training sequences
        self.files = []
        for seq in self.train_seqs:
            self.files.extend(sorted(glob.glob(os.path.join(seq, "h5_f_*.h5"))))
            
        print(f"Total Training Frames: {len(self.files)}")
        print(f"Sparsity: 1/{sparsity:.1f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as f:
            x_dense = torch.from_numpy(f['x'][:].T).float() # Material Coords (Reference)
            q_dense = torch.from_numpy(f['q'][:].T).float() # Spatial Coords (Current)
            
        N = x_dense.shape[0]
        indices = torch.arange(0, N, step=int(self.sparsity))
        
        x_sparse = x_dense[indices]
        q_sparse = q_dense[indices]
        f_sparse = q_sparse - x_sparse # Feature: Displacement
        target_disp = q_dense - x_dense # Target: Dense Displacement
        
        return x_sparse, f_sparse, x_dense, target_disp

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Water")
    default_out = os.path.join(PROJECT_ROOT, "checkpoints", "gno")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data)
    parser.add_argument('-out', type=str, default=default_out, help="Output folder")
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-radius', type=float, default=0.015, help="Radius must be > sparsity gap")
    parser.add_argument('-sparsity', type=float, default=24.0, help="Stride for particle sampling")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    # Initialize Dataset (Excluding Sequence 0)
    dataset = GNODataset(args.data, sparsity=args.sparsity, holdout_idx=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize Model
    model = GNO(dim=3, gno_radius=args.radius).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    print(f"Starting GNO Training on {device} (Radius={args.radius})...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for i, (x_sp, f_sp, x_de, target) in enumerate(pbar):
            # Move to GPU & Squeeze Batch Dim (1, N, 3) -> (N, 3)
            x_sp = x_sp.to(device).squeeze(0)
            f_sp = f_sp.to(device).squeeze(0)
            x_de = x_de.to(device).squeeze(0)
            target = target.to(device).squeeze(0)
            
            optimizer.zero_grad()
            
            # Forward
            pred_disp = model(x_sp, f_sp, x_de)
            
            loss = loss_fn(pred_disp, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # i+1 prevents division by zero on the first iteration
            pbar.set_postfix({'loss': total_loss / (i + 1)})
            
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.4e} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.out, f"gno_epoch{epoch}.pt"))

    # Save Final
    dataset_name = os.path.basename(os.path.normpath(args.data))
    final_path = os.path.join(args.out, f"gno_final_{args.sparsity}_{dataset_name}.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete. Weights saved to {final_path}")

if __name__ == "__main__":
    main()