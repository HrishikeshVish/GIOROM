import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

def compute_spectral_metrics(fine_graph, coarse_graph, kmax=30):
    """
    Computes spectral metrics for graph coarsening.

    Parameters:
    - fine_graph (torch_geometric.data.Data): Original fine-resolution graph.
    - coarse_graph (torch_geometric.data.Data): Coarsened graph.
    - kmax (int): Number of eigenvalues/vectors to consider.

    Returns:
    - metrics (dict): Dictionary containing spectral quality metrics.
    """
    # Convert edge indices to adjacency matrices
    W_fine = to_scipy_sparse_matrix(fine_graph.edge_index, num_nodes=fine_graph.num_nodes).tocsc()
    W_coarse = to_scipy_sparse_matrix(coarse_graph.edge_index, num_nodes=coarse_graph.num_nodes).tocsc()

    # Ensure adjacency matrices are not empty
    if W_fine.nnz == 0 or W_coarse.nnz == 0:
        raise ValueError("Adjacency matrix is empty. Ensure the graphs have edges.")

    # Compute normalized Laplacians
    L_fine = sp.csgraph.laplacian(W_fine, normed=True)
    L_coarse = sp.csgraph.laplacian(W_coarse, normed=True)

    # # Ensure Laplacians are not zero
    # if np.all(L_fine == 0) or np.all(L_coarse == 0):
    #     raise ValueError("Laplacian matrix is all zeros. Check the input graphs.")

    # Compute eigenvalues and eigenvectors with a random initial vector
    k_fine = min(kmax, L_fine.shape[0] - 1)
    k_coarse = min(kmax, L_coarse.shape[0] - 1)

    l_fine, U_fine = sp.linalg.eigsh(L_fine, k=k_fine, which='SM', v0=np.random.rand(L_fine.shape[0]))
    l_coarse, U_coarse = sp.linalg.eigsh(L_coarse, k=k_coarse, which='SM', v0=np.random.rand(L_coarse.shape[0]))

    # Compute eigenvalue error
    error_eigenvalue = np.abs(l_fine[:k_coarse] - l_coarse[:k_coarse]) / (l_fine[:k_coarse] + 1e-6)
    error_eigenvalue[0] = 0  # Ignore first eigenvalue (close to zero)

    # Compute angles between eigenspaces
    #angle_matrix = U_fine.T @ U_coarse

    # Compute subspace errors
    #error_sintheta = np.linalg.norm(angle_matrix[:, k_coarse:], ord='fro') ** 2

    metrics = {
        "error_eigenvalue": error_eigenvalue.mean(),
        "r": 1 - (coarse_graph.num_nodes / fine_graph.num_nodes),
        "m": coarse_graph.num_edges
    }

    return metrics
