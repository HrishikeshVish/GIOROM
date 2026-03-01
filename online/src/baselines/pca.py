import torch

class GappyPCA:
    def __init__(self, latent_dim=20, device='cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.U = None

    def fit(self, snapshots):
        """
        Computes the PCA spatial basis U from training snapshots.
        Args:
            snapshots: Tensor of shape (Total_Frames, N, D)
        """
        B, N, D = snapshots.shape
        # Flatten spatial dims: (Total_Frames, N*D)
        flattened = snapshots.view(B, -1)
        
        # Transpose to (Features, Samples) for PCA
        X = flattened.T 
        
        print(f"Computing PCA on snapshot matrix {X.shape} (Features, Samples)...")
        
        # Torch's pca_lowrank is fast for large snapshot matrices
        _, _, V = torch.pca_lowrank(X.T, q=self.latent_dim, center=False) 
        
        # Spatial Basis (Features, k)
        self.U = V 
        print(f"Basis computed. Shape: {self.U.shape}")

    def predict(self, x_sparse, indices, N, D):
        """
        Solves the Gappy POD least-squares problem to reconstruct the full state.
        """
        if self.U is None:
            raise ValueError("PCA basis not computed. Call fit() first.")
            
        k = self.U.shape[1]
        
        # Vectorized index creation to extract the rows of U corresponding to active sensors
        idx_flat = (indices.unsqueeze(1) * D + torch.arange(D, device=self.device).unsqueeze(0)).view(-1)
        
        U_sparse = self.U[idx_flat, :] # (M*D, k)
        y = x_sparse.view(-1, 1)       # (M*D, 1)
        
        # Setup Least Squares: (U^T U) a = U^T y
        XtX = torch.matmul(U_sparse.T, U_sparse)
        Xty = torch.matmul(U_sparse.T, y)
        
        # Add tiny jitter to diagonal for numerical stability (Tikhonov regularization)
        XtX = XtX + torch.eye(k, device=self.device) * 1e-6
        coeffs = torch.linalg.solve(XtX, Xty)
        
        # Reconstruct full state: x = U * a
        x_recon_flat = torch.matmul(self.U, coeffs)
        return x_recon_flat.view(N, D)