"""
Python implementation of neighbor-search algorithm for use on CPU to avoid
breaking torch_cluster's CPU version.
"""

import torch

def simple_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """

    Parameters
    ----------
    Density-Based Spatial Clustering of Applications with Noise
    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    dists = torch.cdist(queries, data).to(queries.device) # shaped num query points x num data points
    in_nbr = torch.where(dists <= radius, 1., 0.) # i,j is one if j is i's neighbor
    #zeroes = torch.zeros_like(dists).cuda()
    #bottomk, bottomkindices = torch.topk(dists, k=20, largest=False)
    #zeroes[torch.arange(zeroes.size(0)).unsqueeze(1),bottomkindices] = 1.
    #for i in range(zeroes.shape[0]):
    #    zeroes[i][bottomkindices[i]] = 1.
    #in_nbr = zeroes
    #in_nbr = torch.where(dists==radius, 1., 0.)

    nbr_indices = in_nbr.nonzero()[:,1:].reshape(-1,) # only keep the column indices
    nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0) # num points in each neighborhood, summed cumulatively
    #print(nbrhd_sizes, nbrhd_sizes.shape)
    #print(nbr_indices)
    #print(in_nbr.shape)
    #print(nbr_indices.shape)

    splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
    nbr_dict['neighbors_row_splits'] = splits.long().to(queries.device)
    
    return nbr_dict