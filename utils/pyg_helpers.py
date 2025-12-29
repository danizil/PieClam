import os
import scipy.sparse as sp
import scipy.io
import torch

from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric import utils

from utils.printing_utils import printd

#* pyg supplement functions. 



def load_data(data_source):
    data = scipy.io.loadmat("gae/data/{}.mat".format(data_source))
    # labels = data["gnd"]
    return data

def sparse_matrix_to_edge_index(sparse_matrix):
    
    # Ensure the matrix is in COO format
    sparse_matrix = sparse_matrix.tocoo()

    # Extract row and column indices
    row = torch.tensor(sparse_matrix.row, dtype=torch.long)
    col = torch.tensor(sparse_matrix.col, dtype=torch.long)

    # Stack indices to form edge_index
    edge_index = torch.stack([row, col], dim=0)

    return edge_index

def has_repeating_edges(edge_index):
    
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()

    # Check for duplicates
    for edge in edges:
        if edge in seen_edges:
            return True
        seen_edges.add(edge)

    return False

def keep_bidirectional_edges(edge_index):
    """
    Throw out directed edges
    
    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only bidirectional edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()
    bidirectional_edges = []

    for edge in edges:
        if (edge[1], edge[0]) in seen_edges:
            bidirectional_edges.append(edge)
            bidirectional_edges.append((edge[1], edge[0]))
        seen_edges.add(edge)

    # Convert back to tensor
    bidirectional_edges = torch.tensor(bidirectional_edges, dtype=torch.long).t()

    return bidirectional_edges

def remove_duplicate_edges(edge_index):
    """
    Remove duplicate edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with duplicate edges removed.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track unique edges
    unique_edges = list(set(edges))

    # Convert back to tensor
    unique_edges = torch.tensor(unique_edges, dtype=torch.long).t()

    return unique_edges


def get_undirected_edges(edge_index):
    """
    Identify and return undirected edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only undirected edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()
    undirected_edges = []

    for edge in edges:
        if (edge[1], edge[0]) in seen_edges:
            undirected_edges.append(edge)
            undirected_edges.append((edge[1], edge[0]))
        seen_edges.add(edge)

    # Convert back to tensor
    undirected_edges = torch.tensor(undirected_edges, dtype=torch.long).t()

    return undirected_edges

def get_directed_edges(edge_index):
    """
    Identify and return directed edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only directed edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set(edges)
    directed_edges = []

    for edge in edges:
        if (edge[1], edge[0]) not in seen_edges:
            directed_edges.append(edge)

    # Convert back to tensor
    directed_edges = torch.tensor(directed_edges, dtype=torch.long).t()

    return directed_edges

def check_indices_in_edge_index(indices, edge_index):
    """
    Check if the edge_index array contains the indices in the list.

    Parameters:
    indices (list of tuples): List of index tuples to check.
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    list of bool: Boolean array indicating the presence of each index in the edge_index.
    """
    # Convert edge_index to a set of tuples for efficient lookup
    edges_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Check if each index is in the edges_set
    result = [(i, j) in edges_set for i, j in indices]

    return result

def check_anomalies_in_edge_index(edge_index, anomaly_indices):
    """
    Check if the edge_index contains indices from the anomaly_indices array.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].
    anomaly_indices (torch.Tensor): The tensor containing anomaly indices.

    Returns:
    torch.Tensor: Boolean tensor indicating the presence of anomaly indices in the edge_index.
    """
    # Convert anomaly_indices to a set for efficient lookup
    anomaly_set = set(anomaly_indices.tolist())

    # Check if each edge contains an anomaly index
    result = [(i in anomaly_set and j in anomaly_set) for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist())]

    # Convert result to a tensor
    result_tensor = torch.tensor(result, dtype=torch.bool)

    return result_tensor

def compare_datas(data1, data2):
    att1_notin2 = set()
    att2_notin1 = set()
    diffs = set()

    # Get the attributes of the datasets
    attrs1 = data1.__dict__['_store']
    attrs2 = data2.__dict__['_store']
    
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    # first compare if they have the same attributes
    for key in keys1:
        if key not in keys2:
            att1_notin2.add(key)
        else:
            try:
                if isinstance(attrs1[key], torch.Tensor) and isinstance(attrs2[key], torch.Tensor): 
                    if not torch.equal(attrs1[key], attrs2[key]):
                        diffs.add(key)
                else:
                    if (torch.tensor(attrs1[key]) != torch.tensor(attrs2[key])).any():
                        diffs.add(key)
            except Exception as e:
                printd(f'Error when comparing data objects: {e}')

    for key in keys2:
        if key not in keys1:
            att2_notin1.add(key)
    
    return att1_notin2, att2_notin1, diffs


def check_single_device(data, verbose=False):
    device = data.edge_index.device
    diff_device_list = []
    tbr = True
    for key, value in data:
        if hasattr(value, 'device') and value.device != device:
            tbr = False
            diff_device_list.append(key)
    if verbose:
        printd(f"\ndevice of edge_index: {device}")
        if not tbr:
            printd(f"\nthe following attributes are not on the same device as the edge_index:\n {diff_device_list}")
    
    return tbr



def dropout_edge_undirected(edge_index, p=0.5):
    
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index
    edge_index_directed = edge_index[:, row < col]

    row_directed, col_directed = edge_index_directed

    edge_mask_directed = torch.rand(row_directed.size(0), device=edge_index.device) >= p
    edge_index_directed_retained = edge_index_directed[:, edge_mask_directed]

    edge_index_retained = torch.cat([edge_index_directed_retained, edge_index_directed_retained.flip(0)], dim=1)
    edge_mask_retained = torch.cat([edge_mask_directed, edge_mask_directed])
    edge_index_orig_rearange = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)

    return edge_index_retained, edge_mask_retained, edge_index_orig_rearange

# i just want an edge_attr mask for the edges that were dropped

# def edge_mask_drop_and_rearange(edge_index, p):
#     '''
#     drops edges with probability p.
#     returns the edge index rearaged and the mask for the dropped edges
#     it's important to get the rearanged edge_index to get the correct mask because it's hard to get it for undirected'''
    
#     # assert utils.is_undirected(edge_index), 'edge_index is directed'
#     row, col = edge_index
#     edge_index_directed = edge_index[:, row < col] # the edge_index is assumed to be directed

#     row_directed, col_directed = edge_index_directed
    
#     edge_mask_directed_retain = torch.rand(row_directed.size(0), device=edge_index.device) >= p
    
#     edge_mask_retain = torch.cat([edge_mask_directed_retain, edge_mask_directed_retain])
#     edge_index_orig_rearange = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)
#     # very important to use the new edge index otherwise the positions of the dropped edges is not correct!
#     return edge_index_orig_rearange, edge_mask_retain

def edge_mask_drop_and_rearange(edge_index, p, directed):
    '''
    drops edges with probability p.
    returns the edge index rearaged and the mask for the dropped edges
    it's important to get the rearanged edge_index to get the correct mask because it's hard to get it for undirected'''
    
    # assert utils.is_undirected(edge_index), 'edge_index is directed'
    num_edges = edge_index.size(1)
    num_edges_to_omit = int(p * num_edges)
    
    if directed:
        # For directed graphs, sample edges directly
        indices_omitted = torch.randperm(num_edges, device=edge_index.device)[:num_edges_to_omit]
        edge_mask_retain = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        edge_mask_retain[indices_omitted] = False
        return edge_index, edge_mask_retain
    else:
        # For undirected graphs, work with unique edges
        row, col = edge_index
        edge_index_directed = edge_index[:, row < col]  # Get unique edges
        
        num_unique_edges = edge_index_directed.size(1)
        num_unique_to_omit = int(p * num_unique_edges)
        
        # Sample unique edges
        unique_indices_omitted = torch.randperm(num_unique_edges, device=edge_index.device)[:num_unique_to_omit]
        edge_mask_directed_retain = torch.ones(num_unique_edges, dtype=torch.bool, device=edge_index.device)
        edge_mask_directed_retain[unique_indices_omitted] = False
        
        # Create mask for both directions
        edge_mask_retain = torch.cat([edge_mask_directed_retain, edge_mask_directed_retain])
        
        # Create the rearranged edge index
        edge_index_orig_rearrange = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)
        
        return edge_index_orig_rearrange, edge_mask_retain



# def two_hop_link(data):
#     '''densify the edges with with attr 1. if one of the edges with attr 0 is produced, set it's attr to 1'''
#     assert data.edge_index is not None
#     edge_index, edge_attr = data.edge_index, data.edge_attr
#     N = data.num_nodes

#     # Sort edge_index by the attribute values
#     # sorted_indices = edge_attr.argsort(dim=0, descending=True)
#     # edge_index = edge_index[:, sorted_indices]
#     # edge_attr = edge_attr[sorted_indices]
    
#     #densify the edge with attribute 1
#     edges_to_densify = edge_index[:, edge_attr]

#     edges_to_densify = EdgeIndex(edges_to_densify, sparse_size=(N, N))
#     edges_to_densify = edges_to_densify.sort_by('row')[0]
#     # all of the 2hop edges V:
#     edges_densified = edges_to_densify.matmul(edges_to_densify)[0].as_tensor()
#     # edges_densified, _ = remove_self_loops(edges_densified)
#     edge_index = torch.cat([edge_index, edges_densified], dim=1)

#     # We treat newly added edge features as "zero-features":
#     attr_densified = torch.ones(edges_densified.size(1)).bool()
#     edge_attr = torch.cat([edge_attr, attr_densified], dim=0)

#     edge_index, edge_attr = coalesce(edge_index, edge_attr, N, reduce="max")

#     return edge_index, edge_attr

import torch

def two_hop_link(data):
    '''densify the edges with with attr 1. if one of the edges with attr 0 is produced, set it's attr to 1'''
    assert data.edge_index is not None
    edge_index, edge_attr = data.edge_index, data.edge_attr
    N = data.num_nodes

    #densify the edge with attribute 1
    edges_to_densify = edge_index[:, edge_attr]
    
    # Convert to sparse COO tensor
    indices = edges_to_densify
    values = torch.ones(edges_to_densify.shape[1], device=edges_to_densify.device, dtype=torch.float32)
    #! todo: the edges created by densification should have a smaller weight in the calculation if the iteration
    sparse_adj = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(N, N)
    )
    
    # Compute 2-hop connections: sparse matrix multiplication A²
    sparse_adj_2hop = torch.sparse.mm(sparse_adj, sparse_adj)
    
    # Convert back to edge_index (COO format)
    sparse_adj_2hop = sparse_adj_2hop.coalesce()
    edges_densified = sparse_adj_2hop.indices()
    
    # Remove self-loops if desired
    # edges_densified, _ = remove_self_loops(edges_densified)
    edge_index = torch.cat([edge_index, edges_densified], dim=1)

    # We treat newly added edge features as "zero-features":
    attr_densified = torch.ones(edges_densified.size(1), device=edges_densified.device).bool()
    edge_attr = torch.cat([edge_attr, attr_densified], dim=0)

    edge_index, edge_attr = coalesce(edge_index, edge_attr, N, reduce="max")

    return edge_index, edge_attr

# 88b 88 888888  dP""b8 
# 88Yb88 88__   dP   `" 
# 88 Y88 88""   Yb  "88 
# 88  Y8 888888  YboodP 

# .dP"Y8    db    8b    d8 88""Yb 88     88 88b 88  dP""b8 
# `Ybo."   dPYb   88b  d88 88__dP 88     88 88Yb88 dP   `" 
# o.`Y8b  dP__Yb  88YbdP88 88"""  88  .o 88 88 Y88 Yb  "88 
# 8bodP' dP""""Yb 88 YY 88 88     88ood8 88 88  Y8  YboodP 

# -*- coding: utf-8 -*-
# Negative sampling that CAN include self-loops.
# This mirrors torch_geometric.utils.negative_sampling but changes the
# encoding/decoding helpers to include the diagonal (self-loops).
# All CHANGED lines are annotated.

import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# (Optional) used by batched version below (unchanged logic there)
from torch_geometric.utils import coalesce, cumsum, degree, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.no_grad()
def negative_sampling_with_self_loops(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges, **including self-loops** if they do not
    already exist in :attr:`edge_index`. API mirrors PyG.

    CHANGED: self-loops are now part of the candidate population.
    """

    assert method in ['sparse', 'dense']

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    # CHANGED: use *_with_diag helpers (include diagonal)
    idx, population = edge_index_to_vector_with_diag(edge_index, size, bipartite,
                                                     force_undirected)  # CHANGED

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2  # will mirror later (same as PyG)

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(max(1, 1.1 * num_neg_samples / max(prob, 1e-12)))  # (Over)-sample size.  # CHANGED: clamp

    neg_idx = None
    if method == 'dense':
        # The dense version creates a mask of shape `population` to check for invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx_cpu = idx.to('cpu')
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device='cpu')
            mask = np.isin(rnd, idx_cpu.numpy())
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to('cpu').numpy())
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = torch.as_tensor(rnd, device=edge_index.device)[~mask]  # CHANGED: ensure tensor on device
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    # CHANGED: use decoder that understands diagonal (and mirroring)
    return vector_to_edge_index_with_diag(neg_idx, size, bipartite, force_undirected)  # CHANGED


# ------------------ helpers (CHANGED vs. PyG) ------------------

def edge_index_to_vector_with_diag(
    edge_index: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[Tensor, int]:
    """
    CHANGED: Unlike PyG's edge_index_to_vector, this includes the diagonal (self-loops)
    in the population for non-bipartite graphs.
    """
    row, col = edge_index

    if bipartite:  # No concept of self-loops; unchanged.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population

    assert size[0] == size[1]
    n = size[0]

    if force_undirected:
        # CHANGED: operate on upper triangle INCLUDING diagonal: row <= col
        mask = row <= col
        row, col = row[mask], col[mask]
        # Map (r,c), r<=c to a linear index in [0, n(n+1)/2)
        # count before row r in upper-tri (incl diag): t_r = r*n - r*(r-1)/2
        t_r = row * n - (row * (row - 1)) // 2
        idx = t_r + (col - row)
        population = (n * (n + 1)) // 2  # CHANGED: include diagonal
        return idx, population

    else:
        # CHANGED: directed mapping includes diagonal: idx = r*n + c in [0, n*n)
        idx = row * n + col
        population = n * n
        return idx, population


def vector_to_edge_index_with_diag(
    idx: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tensor:
    """
    CHANGED: Inverse of the diag-aware encoding above.
    """
    if bipartite:
        row = idx.div(size[1], rounding_mode='floor')
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    assert size[0] == size[1]
    n = size[0]

    if force_undirected:
        # Invert idx = t_r + (c - r), t_r = r*n - r*(r-1)//2, with r<=c
        r_vals = torch.arange(n, device=idx.device)
        t = r_vals * n - (r_vals * (r_vals - 1)) // 2  # 0, n, 2n-1, 3n-3, ...
        r = torch.bucketize(idx, t[1:], right=False)
        t_r = r * n - (r * (r - 1)) // 2
        c = r + (idx - t_r)

        # Mirror to both directions; keep self-loops once.
        is_self = (r == c)
        row = torch.cat([r[~is_self], c[~is_self], r[is_self]])
        col = torch.cat([c[~is_self], r[~is_self], c[is_self]])
        return torch.stack([row, col], dim=0)

    else:
        # Directed: simple base-n decode
        row = idx.div(n, rounding_mode='floor')
        col = idx % n
        return torch.stack([row, col], dim=0)


# ------------------ utilities (unchanged) ------------------

def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


# ---------- Optional: batched variant that calls the new sampler ----------

@torch.no_grad()
def batched_negative_sampling_with_self_loops(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    """
    Same behavior as PyG's batched_negative_sampling, but using the
    self-loop–aware sampler above.
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        ptr = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, ei in enumerate(edge_indices):
        ei = ei - ptr[i]
        neg_ei = negative_sampling_with_self_loops(
            ei, num_nodes[i], num_neg_samples, method, force_undirected
        )
        neg_ei += ptr[i]
        neg_edge_indices.append(neg_ei)

    return torch.cat(neg_edge_indices, dim=1)