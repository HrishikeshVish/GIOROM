import os
import torch
import torch.nn as nn
from torch.autograd import grad

class Decoder(object):
    def __init__(self, network, md=None, netfuncgrad=None):
        self.network = network
        self.md = md
        self.netfuncgrad = netfuncgrad
        
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False
            
        if self.netfuncgrad:
            self.netfuncgrad.eval()
            for param in self.netfuncgrad.parameters():
                param.requires_grad = False

    def getPartGradx(self, x, part_dim, which):
        x = x.detach()
        x_first = x[:, 0:part_dim]
        x_second = x[:, part_dim:x.size(1)]
        
        if which == 'fir':
            x_grad = x_first
            x_grad.requires_grad_(True)
            x_combined = torch.cat((x_grad, x_second), 1)
        elif which == 'sec':
            x_grad = x_second
            x_grad.requires_grad_(True)
            x_combined = torch.cat((x_first, x_grad), 1)
            
        return x_grad, x_combined

    def jacobianPartAndFunc(self, x, part_dim, which):
        if self.netfuncgrad:
            with torch.inference_mode():
                grad_val, y = self.netfuncgrad(x)
                if which == 'fir':
                    grad_val = grad_val[:, :, 0:part_dim]
                elif which == 'sec':
                    grad_val = grad_val[:, :, part_dim:x.size(1)]
                jacobian = grad_val.view(-1, 1, grad_val.size(2))
            y = y.view(1, y.size(0), y.size(1))
            return jacobian, y
        else:
            x_grad, x_in = self.getPartGradx(x, part_dim, which)
            outputs = self.network(x_in)
            output_dim = outputs.size(1) 
            jacobian = []
            for i in range(output_dim):
                grad_i = grad(outputs[:, i], x_grad, 
                              grad_outputs=torch.ones_like(outputs[:, i]),
                              create_graph=False, retain_graph=True)[0]
                jacobian.append(grad_i.unsqueeze(1)) 
            jacobian = torch.cat(jacobian, dim=1) 
            jacobian = jacobian.view(-1, 1, jacobian.size(2)).detach()
            return jacobian, outputs.unsqueeze(0)


class LiCROM_Inference:
    def __init__(self, enc_path, dec_path, device, sample_input):
        print("Loading CROM Models for LiCROM...")
        self.device = device
        
        net_enc = torch.jit.load(enc_path, map_location=device).eval()
        net_dec = torch.jit.load(dec_path, map_location=device).eval()
        
        grad_path = dec_path.replace("_dec", "_dec_func_grad")
        if os.path.exists(grad_path):
            print(f"Found Gradient Network: {os.path.basename(grad_path)}")
            net_grad = torch.jit.load(grad_path, map_location=device).eval()
        else:
            print("No Gradient Network found. Using Autograd Fallback.")
            net_grad = None

        self.encoder = net_enc 
        self.decoder = Decoder(net_dec, dec_path, net_grad)
        
        print("Detecting Latent Dim using Real Data...")
        inp = sample_input.unsqueeze(0)
        
        with torch.no_grad():
            try:
                z = self.encoder(inp) 
            except:
                z = self.encoder(inp.permute(0, 2, 1))
        
        self.latent_dim = z.numel() 
        print(f"Latent Dim: {self.latent_dim} (Shape: {z.shape})")

    def solve_gauss_newton(self, x_ref, x_sparse, indices, z_init, steps=5):
        z = z_init.clone() 
        x_ref_sparse = x_ref[indices] 
        M = x_ref_sparse.shape[0]
        
        for k in range(steps):
            z_exp = z.expand(1, M, self.latent_dim)
            ref_exp = x_ref_sparse.unsqueeze(0) 
            dec_in = torch.cat([z_exp, ref_exp], dim=2).view(-1, self.latent_dim + 3)
            
            Jac, y = self.decoder.jacobianPartAndFunc(dec_in, self.latent_dim, 'fir')
            
            pred = y.view(M, 3)
            residual = (x_sparse - pred).view(-1, 1) 
            
            J_mat = Jac.squeeze(1) 
            A = torch.matmul(J_mat.T, J_mat)
            b = torch.matmul(J_mat.T, residual)
            
            A = A + torch.eye(self.latent_dim, device=self.device) * 1e-4
            delta_z = torch.linalg.solve(A, b)
            
            z = z + delta_z.view(1, 1, self.latent_dim)
            
        return z

    def decode_full(self, z, x_ref):
        N = x_ref.shape[0]
        z_exp = z.expand(1, N, self.latent_dim)
        ref_exp = x_ref.unsqueeze(0)
        dec_in = torch.cat([z_exp, ref_exp], dim=2).view(-1, self.latent_dim + 3)
        out = self.decoder.network(dec_in)
        return out.view(N, 3)