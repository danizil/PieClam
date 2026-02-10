import torch
import normflows as nf
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

import matplotlib.pyplot as plt

from utils.utils import clam_edges_from_feats
from torch.nn.functional import relu, one_hot


from datasets.data_utils import intersecting_tensor_from_non_intersecting_vec
from transformation import relu_lightcone_pts
import math

def create_sbm_directed(num_samples_per_comm, interaction_probs):
    num_comms = math.sqrt(len(interaction_probs))
    assert num_comms == int(num_comms), 'num_comms is not an integer'
    num_comms = int(num_comms)
    probs = torch.zeros([num_comms, num_comms])
    for i in range(num_comms):
        for j in range(num_comms):
            probs[i,j] = interaction_probs.pop(0)
    
    sbm = probs.repeat_interleave(num_samples_per_comm, dim=0).repeat_interleave(num_samples_per_comm, dim=1)
    # Create community labels y
    y = torch.zeros(num_samples_per_comm * num_comms)
    for i in range(num_comms):
        y[i * num_samples_per_comm: (i + 1) * num_samples_per_comm] = i

    # turn y into an intersecting community tensor 
    y = intersecting_tensor_from_non_intersecting_vec(y)
    return sbm, y

    
def create_sbm_undirected(num_samples_per_comm, p_comm, p_bipart):
    
    num_comms = len(p_comm)
    assert len(p_bipart) == (num_comms**2 - num_comms)/2, 'p_bipart != num_parts^2 - num_parts'
    
    y = torch.tensor([])
    #* fill probs tensor
    probs = torch.zeros([num_comms, num_comms])
    for i in range(num_comms - 1):
        for j in range(i+1, num_comms):
            probs[i,j] = p_bipart.pop(0)
    #* make symmetric
    probs = probs + probs.T
    #* fill diagonal
    probs.diagonal().copy_(torch.tensor(p_comm))
    
    #!  this isn't exactly an sbm yet. we can maybe give differnt parts, make the part list a dictionary

    sbm = probs.repeat_interleave(num_samples_per_comm, dim=0).repeat_interleave(num_samples_per_comm, dim=1)
    # make sbm symmetric with 0 on the diagonal
    sbm.fill_diagonal_(0)
    sbm.tril_()
    sbm = sbm + sbm.T
    
    y = torch.zeros(num_samples_per_comm * num_comms)
    for i in range(num_comms):
        y[i * num_samples_per_comm: (i + 1) * num_samples_per_comm] = i

    assert(len(y) == num_samples_per_comm*num_comms), 'y has wrong length'
    # turn y into an intersecting community tensor
    y = intersecting_tensor_from_non_intersecting_vec(y)
    return sbm, y

def sample_from_adj_directed(prob_adj):
    '''sample edges from probability adjacency'''
    assert torch.all((prob_adj >= 0) & (prob_adj <= 1)), "All elements in the tensor must be between 0 and 1"
    adj_mat = torch.bernoulli(prob_adj)
    return adj_mat.int()

def sample_from_adj_undirected(prob_adj):
    '''sample edges from probability adjacency'''
    
    assert torch.all((prob_adj >= 0) & (prob_adj <= 1)), "All elements in the tensor must be between 0 and 1"
    adj_mat = torch.bernoulli(prob_adj)
    adj_mat = torch.triu(adj_mat, diagonal=1)
    adj_mat = adj_mat + adj_mat.transpose(0,1)
    return adj_mat.int()

# add all graph creation with prior models here

def sample_normflows_dist(num_samples, name_shape, lorenz=False, device='cpu', directed=True):
    ''''sample nodes from one of the normflows packge distributions'''
    assert name_shape in ['Circ', 'TwoMoons', 'ChubGaus'], 'sample_normflow_dist: name_shape should be one of Circ, TwoMoons, ChubGaus'

    if name_shape=='Circ':
        dist = nf.distributions.CircularGaussianMixture()
    if name_shape=='TwoMoons':
        dist = nf.distributions.TwoMoons()
    if name_shape == 'ChubGaus':
        if lorenz:
            raise NotImplementedError('ChubGaus not implemented for lorenz')
        dist = nf.distributions.GaussianMixture(n_modes=2, dim=2, loc=[[0.2, -2],[-1.8, 0]], scale=[[0.3,0.05],[0.08, 0.3]])
        
    node_feats = dist.sample(num_samples).to(device)
    #todo: different shift and scale for lorenz
    if lorenz:
        rotation_matrix = torch.tensor([[1, -1], [1, 1]]).float()
        node_feats = torch.matmul(node_feats, rotation_matrix)
        node_feats = relu_lightcone_pts(node_feats/3 + torch.tensor([1.5, 0]))
    else:
        node_feats = relu(node_feats/5 + torch.tensor([0.5,0.5]))
        


    graph = clam_edges_from_feats(node_feats, lorenz, directed=directed)
    graph.name = name_shape
    graph.edge_attr = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)
    graph.edge_attr = graph.edge_attr.to(device)

    if name_shape == 'TwoMoons':
        mean_x_2moons = graph.x[:, 0].mean(dim=0)
        graph.y = (graph.x[:, 0] > mean_x_2moons).int()
        
        graph.y = one_hot(graph.y.long().squeeze(), num_classes=2)

    if name_shape == 'ChubGaus':
        graph.y = graph.x[:, 0] > graph.x[:, 1]
        #?^^ the sender nodes are true (sender node is bigger than receiver node)
        graph.y = one_hot(graph.y.long().squeeze(), num_classes=2)
        #? ^^ one hot encoding sends true to [0,1] and false to [1,0]!!
    return graph, dist

#todo: need to have an option for directed graphs. the import dataset function makes them undirected and we check for that everywhere.
def simulate_dataset(name, verbose=False):
    figsize = (2, 1)
    if name == '2UP':
        num_samples_per_comm = 20
        # prob_adj_bipart, y = create_sbm_directed(num_samples_per_comm, interaction_probs=[1,1,0,0])
        prob_adj_bipart, y = create_sbm_directed(num_samples_per_comm, interaction_probs=[0.9,0.9,0.1,0.1])
        adj_bipart = sample_from_adj_directed(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')
        return data

# 8888b.  88 88""Yb 
#  8I  Yb 88 88__dP 
#  8I  dY 88 88"Yb  
# 8888Y"  88 88  Yb 
    elif name == 'BipartDir':
        num_samples_per_comm = 20
        # prob_adj_bipart, y = create_sbm_directed(num_samples_per_comm, interaction_probs=[0.1,0.9,0.1,0.1])
        prob_adj_bipart, y = create_sbm_directed(num_samples_per_comm, interaction_probs=[0.1,0.9,0.1,0.1])
        adj_bipart = sample_from_adj_directed(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')

    elif name == 'BipartDirSmallFull':
        num_samples_per_comm = 5
        prob_adj_bipart, y = create_sbm_directed(num_samples_per_comm, interaction_probs=[0,1,0,0])
        adj_bipart = sample_from_adj_directed(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')

    elif name == 'SmallDirected':
        '''small directed graph, every dyad is an edge with probability 0.5'''
        prob_adj_bipart, y = create_sbm_directed(10, interaction_probs=[0.2])
        adj = sample_from_adj_directed(prob_adj_bipart)
        edge_index = dense_to_sparse(adj)[0]
        data = Data(edge_index=edge_index)
        if verbose == True:
            _, ax = plt.subplots(1,1, figsize=figsize)
            ax.imshow(adj)

            
# 88   88 88b 88 8888b.  88 88""Yb 
# 88   88 88Yb88  8I  Yb 88 88__dP 
# Y8   8P 88 Y88  8I  dY 88 88"Yb  
# `YbodP' 88  Y8 8888Y"  88 88  Yb 
    elif name == 'smallBipart':
        num_samples_per_comm = 3
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0], p_bipart=[1])
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')
    
    
    elif name == 'bipartite':
        num_samples_per_comm = 30
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.1, 0.1], p_bipart=[0.9])
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')

    elif name == 'tripartite':
        num_samples_per_comm = 100
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0, 0.0], p_bipart=[0.5, 0.0, 0.5]) 
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)

    elif name == 'largeBipartFull':
        num_samples_per_comm = 100
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0], p_bipart=[1.0])
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)

    elif name == 'smallBipartFull':
        num_samples_per_comm = 10
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0], p_bipart=[1.0])
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
            
    elif name == 'bipartiteHalf':
        num_samples_per_comm = 100
        prob_adj_bipart, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0], p_bipart=[0.5])
        adj_bipart = sample_from_adj_undirected(prob_adj_bipart)
        edge_index = dense_to_sparse(adj_bipart)[0]
        data = Data(edge_index=edge_index, y=y)
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_bipart)
            axes[1].imshow(adj_bipart)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')


    elif name=='sbm3x3HalfCenter':
        num_samples_per_comm = 70
        prob_adj_3X3, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.5, 0.0, 0.5], p_bipart=[0.5, 0.5, 0.5]) 
        adj_3X3 = sample_from_adj_undirected(prob_adj_3X3)
        edge_index = dense_to_sparse(adj_3X3)[0]
        data = Data(edge_index=edge_index, y=y)
        data.sbm = prob_adj_3X3
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_3X3)
            axes[1].imshow(adj_3X3)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')



    elif name=='sbm3x3HalfDiag':
        num_samples_per_comm = 70
        prob_adj_3X3, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.0, 0.0, 0.0], p_bipart=[0.5, 0.5, 0.5]) 
        adj_3X3 = sample_from_adj_undirected(prob_adj_3X3)
        edge_index = dense_to_sparse(adj_3X3)[0]
        data = Data(edge_index=edge_index, y=y)
        data.sbm = prob_adj_3X3
        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_3X3)
            axes[1].imshow(adj_3X3)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')
        


    elif name=='sbm3x3':
        num_samples_per_comm = 70
        prob_adj_3X3, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.9, 0.1, 0.8], p_bipart=[0.8, 0.1, 0.2]) 
        adj_3X3 = sample_from_adj_undirected(prob_adj_3X3)
        edge_index = dense_to_sparse(adj_3X3)[0]
        data = Data(edge_index=edge_index, y=y)

        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_3X3)
            axes[1].imshow(adj_3X3)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')

    elif name == 'sbm4X4':
        num_samples_per_comm = 70
        prob_adj_4X4, y = create_sbm_undirected(num_samples_per_comm, p_comm=[0.9, 0.1, 0.2, 0.5], p_bipart=[0.8, 0.1, 0.7, 0.3, 0.4, 0.6]) 
        adj_4X4 = sample_from_adj_undirected(prob_adj_4X4)
        edge_index = dense_to_sparse(adj_4X4)[0]
        data = Data(edge_index=edge_index, y=y)

        if verbose == True:
            _, axes = plt.subplots(1,2, figsize=figsize)
            axes[0].imshow(prob_adj_4X4)
            axes[1].imshow(adj_4X4)
            axes[0].set_title('sbm')
            axes[1].set_title('sampled')


    else: 
        raise NotImplementedError(f'in simulate_dataset, dataset {name} not implemented yet')
    
    return data
