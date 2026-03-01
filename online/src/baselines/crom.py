import torch

class CROM_Inference:
    def __init__(self, enc_path, dec_path, device, sparsity=1):
        print(f"Loading CROM Models (Sparsity: {sparsity})...")
        self.encoder = torch.jit.load(enc_path, map_location=device).eval()
        self.decoder = torch.jit.load(dec_path, map_location=device).eval()
        self.device = device
        self.sparsity = int(sparsity)

    def predict(self, x_ref, x_curr):
        """
        Runs CROM Autoencoder Inference in 'Sparse Sensor' mode.
        """
        with torch.no_grad():
            # 1. ENCODE (Sparse Input)
            if self.sparsity > 1:
                x_curr_sparse = x_curr[::self.sparsity]
                inp = x_curr_sparse.unsqueeze(0) 
            else:
                inp = x_curr.unsqueeze(0)

            z = self.encoder(inp) 
            if z.dim() == 2: 
                z = z.unsqueeze(1)

            # 2. DECODE (Dense Query)
            N_full = x_ref.size(0)
            z_expanded = z.expand(1, N_full, z.size(2))
            ref_expanded = x_ref.unsqueeze(0)
            
            dec_in = torch.cat([z_expanded, ref_expanded], dim=2)
            dec_in_flat = dec_in.view(-1, dec_in.size(2))
            
            pred_flat = self.decoder(dec_in_flat)
            pred = pred_flat.view(N_full, 3)
            
        return pred