import torch
import numpy as np
import os
import glob
import h5py
from tqdm import tqdm

def load_and_prep_data(path):
    print(f"Loading {path}...")
    data_raw = torch.load(path, map_location='cpu', weights_only=False)['position']
    
    if isinstance(data_raw, list):
        if len(data_raw) > 0 and isinstance(data_raw[0], np.ndarray):
            data_raw = [torch.from_numpy(arr) for arr in data_raw]
        sizes = [d.shape[0] for d in data_raw]
        if len(set(sizes)) > 1:
            min_p = min(sizes)
            data_raw = [d[:min_p] for d in data_raw]
        data_pt = torch.stack(data_raw)
    else:
        data_pt = data_raw
    
    # Standardize dimensions
    dim1, dim2 = data_pt.shape[1], data_pt.shape[2]
    if dim1 > 5000 and dim1 > dim2:
        data_pt = data_pt.permute(0, 2, 1, 3)
    
    raw_np = data_pt.numpy()
    dmin, dmax = raw_np.min(), raw_np.max()
    norm = (raw_np - dmin) / (dmax - dmin)
    
    # Fixed split for fair evaluation
    test_data = norm[0:1]    
    train_data = norm[0:]   
    video_traj = test_data[0]
    
    print(f"Dataset Split: {train_data.shape[0]} train sequences, {test_data.shape[0]} test sequences.")
    return train_data, test_data, video_traj, dmin, dmax



class HDF5TrajectoryLoader:
    def __init__(self, data_root, device):
        self.data_root = data_root
        self.device = device
        self.sim_folders = sorted(glob.glob(os.path.join(data_root, "sim_seq_*")))
        if len(self.sim_folders) == 0:
            # Fallback if data_root points directly to a trajectory folder
            self.sim_folders = [data_root]
        print(f"Found {len(self.sim_folders)} trajectories at {data_root}.")

    def get_trajectory(self, sim_idx, max_frames=200):
        folder = self.sim_folders[sim_idx]
        files = sorted(glob.glob(os.path.join(folder, "h5_f_*.h5")))
        if not files:
            return [], []
            
        files = files[:max_frames]
        traj_ref = []
        traj_curr = []
        
        print(f"Loading {len(files)} frames from {os.path.basename(folder)}...")
        for fpath in tqdm(files):
            with h5py.File(fpath, 'r') as f:
                # Transpose .T to match (N, 3)
                x_ref = torch.from_numpy(f['x'][:].T).float().to(self.device)
                x_curr = torch.from_numpy(f['q'][:].T).float().to(self.device)
                traj_ref.append(x_ref)
                traj_curr.append(x_curr)
        
        return traj_ref, traj_curr