import os
import torch
import torch.nn as nn
from torchdiffeq import odeint

# ---> PASTE YOUR Swish, MLP2, AND Derivative CLASSES HERE <---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import glob
import h5py
from functools import partial
from tqdm import tqdm

# Check for torchdiffeq
try:
    from torchdiffeq import odeint
except ImportError:
    print("Error: torchdiffeq not installed. Run 'pip install torchdiffeq'")
    exit(1)

# ==========================================
# 1. DYNAMICS ARCHITECTURE (Derivative MLP)
# ==========================================

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

ACTIVATIONS = {
    "relu": partial(nn.ReLU),
    "sigmoid": partial(nn.Sigmoid),
    "tanh": partial(nn.Tanh),
    "selu": partial(nn.SELU),
    "softplus": partial(nn.Softplus),
    "gelu": partial(nn.GELU),
    "swish": partial(Swish),
    "elu": partial(nn.ELU),
    "leakyrelu": partial(nn.LeakyReLU),
}

class MLP2(nn.Module):
    def __init__(self, code_size, hidden_size, depth=1, nl="swish"):
        super().__init__()
        if nl not in ACTIVATIONS:
            raise ValueError(f"Activation {nl} not found.")
        net = [nn.Linear(code_size, hidden_size), ACTIVATIONS[nl]()]
        for j in range(depth - 1):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(ACTIVATIONS[nl]())
        net.append(nn.Linear(hidden_size, code_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, depth=2, **kwargs):
        super().__init__()
        # Input dim = latent_dim (since state_c=1 means 1 latent vector)
        input_dim = code_c * state_c
        self.net = MLP2(input_dim, hidden_c, depth=depth, nl="swish")

    def forward(self, t, u):
        return self.net(u)

class CORAL_Wrapper:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.sparsity = args.sparsity
        
        print(f"\n--- Initializing HYBRID CORAL ---")
        print(f"Loading Encoder: {os.path.basename(args.enc)}")
        self.encoder = torch.jit.load(args.enc, map_location=device).eval()
        
        print(f"Loading Decoder: {os.path.basename(args.dec)}")
        self.decoder = torch.jit.load(args.dec, map_location=device).eval()
        
        print(f"Loading Dynamics: {os.path.basename(args.coral_online_ckpt)}")
        ckpt = torch.load(args.coral_online_ckpt, map_location=device)
        
        d_args = ckpt['args']
        self.latent_dim = ckpt['latent_dim']
        
        # IMPORTANT: Ensure Derivative is defined above or imported
        self.ode = Derivative(state_c=1, code_c=self.latent_dim, 
                              hidden_c=d_args.hidden_dim, depth=d_args.depth).to(device)
        self.ode.load_state_dict(ckpt['model_state'])
        self.ode.eval()

    def get_initial_z(self, x_ref, x_curr_sparse, indices):
        with torch.no_grad():
            inp = x_curr_sparse.unsqueeze(0) 
            z0 = self.encoder(inp)           
            if z0.dim() == 3: z0 = z0.squeeze(1)
        return z0

    def step_ode(self, t, z_prev):
        with torch.no_grad():
            t_span = torch.tensor([float(t), float(t+1)]).to(self.device)
            z_next = odeint(self.ode, z_prev, t_span, method='euler', options={'step_size': 0.1})
            return z_next[-1]

    def decode(self, z, x_ref):
        with torch.no_grad():
            N_full = x_ref.size(0)
            z_expanded = z.unsqueeze(1).expand(-1, N_full, -1)
            ref_expanded = x_ref.unsqueeze(0)
            
            dec_in = torch.cat([z_expanded, ref_expanded], dim=2)
            dec_in_flat = dec_in.view(-1, dec_in.size(2))
            
            pred_flat = self.decoder(dec_in_flat)
            return pred_flat.view(N_full, 3)