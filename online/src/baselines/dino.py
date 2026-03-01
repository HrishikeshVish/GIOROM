import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x):
        return (x * torch.sigmoid(x * F.softplus(self.beta))).div(1.1)

nls = {'swish': partial(Swish), 'relu': partial(nn.ReLU)}

class MLP(nn.Module):
    def __init__(self, code_size, hidden_size, out_size=None, nl='swish'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size),
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, code_size if out_size is None else out_size),
        )
    def forward(self, x):
        return self.net(x)

class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP(input_dim, hidden_c, nl='swish')
    def forward(self, t, u):
        return self.net(u)

class DINoEvaluator:
    def __init__(self, enc_path, dec_path, ckpt_path, device, sample_input):
        print("Loading DINo Models...")
        self.device = device
        
        # 1. Load Decoder
        self.decoder = torch.jit.load(dec_path, map_location=device).eval()
        
        # 2. Robust Latent Dim Detection
        print("Detecting Latent Dim using real frame...")
        encoder = torch.jit.load(enc_path, map_location=device).eval()
        
        inp = sample_input.unsqueeze(0)
        
        with torch.no_grad():
            try:
                z = encoder(inp) 
            except:
                z = encoder(inp.permute(0, 2, 1)) 
        
        self.latent_dim = z.numel() 
        print(f"Latent Dim: {self.latent_dim} (Shape: {z.shape})")
        del encoder
        
        # 3. Load DINo ODE Checkpoint
        print(f"Loading ODE from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        self.ode_func = Derivative(state_c=1, code_c=self.latent_dim, hidden_c=ckpt['hidden_dim']).to(device)
        self.ode_func.load_state_dict(ckpt['state_dict'])
        self.ode_func.eval()

    def get_initial_z(self, x_ref, x_sparse, indices, steps=100):
        z = torch.zeros(1, 1, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([z], lr=0.02)
        ref_expanded = x_ref.unsqueeze(0)
        
        for _ in range(steps):
            optimizer.zero_grad()
            z_exp = z.expand(1, x_ref.size(0), self.latent_dim)
            dec_in = torch.cat([z_exp, ref_expanded], dim=2).view(-1, self.latent_dim+3)
            pred = self.decoder(dec_in).view(1, x_ref.size(0), 3)
            loss = F.mse_loss(pred[:, indices, :], x_sparse.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
        return z.detach()

    def decode(self, z, x_ref):
        z_exp = z.expand(1, x_ref.size(0), self.latent_dim)
        ref_exp = x_ref.unsqueeze(0)
        dec_in = torch.cat([z_exp, ref_exp], dim=2).view(-1, self.latent_dim+3)
        return self.decoder(dec_in).view(x_ref.size(0), 3)