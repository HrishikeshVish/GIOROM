import sys
import os

# Ensure the root directory is in sys.path so we can import 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import h5py
from tqdm import tqdm

try:
    from torchdiffeq import odeint
except ImportError:
    print("Error: torchdiffeq not installed. Run 'pip install torchdiffeq'")
    exit(1)

# Import the architecture from our modular codebase
from src.baselines.dino import Derivative

# ==========================================
# 1. DATA & NORMALIZATION WRAPPER
# ==========================================
class DINoTrajectoryLoader:
    def __init__(self, data_root, device):
        self.data_root = data_root
        self.device = device
        self.sim_folders = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
        if not self.sim_folders: 
            # Fallback for flat directory
            if glob.glob(os.path.join(data_root, "h5_f_*.h5")):
                self.sim_folders = [data_root]
            else:
                raise ValueError(f"No Data in {data_root}")
        print(f"Found {len(self.sim_folders)} trajectories.")

    def get_batch(self, batch_size=4, seq_len=50):
        sim_indices = np.random.choice(len(self.sim_folders), batch_size)
        batch_x = []
        for sim_idx in sim_indices:
            folder = self.sim_folders[sim_idx]
            files = sorted(glob.glob(os.path.join(folder, "h5_f_*.h5")))
            if len(files) < seq_len: continue
            
            start_t = np.random.randint(0, len(files) - seq_len)
            seq_files = files[start_t : start_t + seq_len]
            
            traj = []
            for fpath in seq_files:
                with h5py.File(fpath, 'r') as f:
                    traj.append(torch.from_numpy(f['q'][:].T).float())
            batch_x.append(torch.stack(traj))

        if not batch_x: return None
        return torch.stack(batch_x).to(self.device)

# ==========================================
# 2. ODE SOLVER HELPER (Scheduled Sampling)
# ==========================================
def scheduling(_int, _f, true_codes, t, epsilon, method='rk4'):
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

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def main():
    default_data = os.path.join(PROJECT_ROOT, "data", "nclaw_Water")
    default_enc = os.path.join(PROJECT_ROOT, "checkpoints", "crom", "encoder.pt")
    default_out = os.path.join(PROJECT_ROOT, "checkpoints", "dino")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default=default_data)
    parser.add_argument('-enc', type=str, default=default_enc)
    parser.add_argument('-out', type=str, default=default_out, help="Output folder")
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-seq_len', type=int, default=20)
    parser.add_argument('-hidden_dim', type=int, default=128, help="Hidden dim for ODE MLP")
    parser.add_argument('-sparsity', type=int, default=64, help="Particle sampling stride")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    # 1. Init Loader
    print("Initializing Data Loader...")
    loader = DINoTrajectoryLoader(args.data, device)
    
    # 2. Load Encoder
    print("Loading CROM Encoder...")
    encoder = torch.jit.load(args.enc, map_location=device).eval()
    
    # 3. Robust Auto-detect Latent Dim
    print("Auto-detecting Latent Dim...")
    sample_batch = loader.get_batch(batch_size=1, seq_len=1) # (1, 1, N, 3)
    if sample_batch is None: raise ValueError("Data load failed")
    
    sample_input = sample_batch[0, 0]
    if args.sparsity > 1:
        sample_input = sample_input[::int(args.sparsity)]
    
    with torch.no_grad():
        try:
            z_sample = encoder(sample_input.unsqueeze(0)) 
        except:
            z_sample = encoder(sample_input.unsqueeze(0).permute(0, 2, 1)) 

    latent_dim = z_sample.numel() 
    print(f"Detected Latent Dim: {latent_dim} (Shape: {z_sample.shape})")

    # 4. Init DINo
    ode_func = Derivative(state_c=1, code_c=latent_dim, hidden_c=args.hidden_dim).to(device)
    optimizer = optim.Adam(ode_func.parameters(), lr=1e-3)
    
    print(f"Starting Training ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(range(50), leave=False)
        
        for _ in pbar:
            x_seq = loader.get_batch(args.batch_size, args.seq_len)
            if x_seq is None: continue
            
            B, T, N, D = x_seq.shape
            
            with torch.no_grad():
                x_flat = x_seq.view(B*T, N, D)
                if args.sparsity > 1:
                    x_flat = x_flat[:, ::int(args.sparsity)]
                try:
                    z_gt_flat = encoder(x_flat)
                except:
                    z_gt_flat = encoder(x_flat.permute(0, 2, 1))
                
                z_gt = z_gt_flat.view(B, T, latent_dim).permute(1, 0, 2)

            optimizer.zero_grad()
            integrator_wrapper = lambda f, y0, t, method: odeint(f, y0, t, method=method, options={'step_size': 0.01})

            epsilon = max(0.0, 1.0 - (epoch / args.epochs))
            t_span = torch.linspace(0, (T-1)*0.01, T).to(device)

            z_pred = scheduling(
                integrator_wrapper, 
                ode_func, 
                true_codes=z_gt, 
                t=t_span, 
                epsilon=epsilon, 
                method='rk4'
            )

            loss = nn.MSELoss()(z_pred, z_gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / 50
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4e} | Epsilon: {epsilon:.2f}")
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            dataset_name = os.path.basename(os.path.normpath(args.data))
            ckpt_path = os.path.join(args.out, f"dino_exact_epoch{epoch}_{args.sparsity}_{dataset_name}.ckpt")
            torch.save({
                'epoch': epoch,
                'state_dict': ode_func.state_dict(),
                'latent_dim': latent_dim,
                'hidden_dim': args.hidden_dim
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()