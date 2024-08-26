
import math
import torch

import numpy as np

from torch_geometric.utils import to_networkx, degree, to_dense_adj
from sklearn.decomposition import PCA

import seaborn as sns
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils.utils import get_prob_graph
from utils.printing_utils import printd


def plot_feats(graph):
    '''plot the individual features per node one by one'''
    num_feats = graph.x.shape[1]
    printd(f'plotting {num_feats} features')
    fig, axes = plt.subplots(2, math.ceil(num_feats/2))
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    
    if num_feats == 2:
        axes[0].plot(graph.x[:,0].detach().numpy())
        axes[1].plot(graph.x[:,1].detach().numpy())
        
    if num_feats > 2:
        for i in range(math.ceil(num_feats/2)):
            axes[0,i].plot(graph.x[:,i].detach().numpy())
            if i+math.ceil(num_feats/2) < num_feats:
                axes[1,i].plot(graph.x[:,i+math.ceil(num_feats/2)].detach().numpy())
        

def plot_relu_lines(lorenz, ax, line_range=1):
    if lorenz:
        plot_uv_axes(ax, line_range)
    else:
        plot_xy_axes(ax, line_range)

def plot_uv_axes(ax, line_range=1.5):
    # Create a range of x values
    x = np.linspace(0, line_range, 50)

    # Create y values for each line
    y1 = x  # 45 degree line
    y2 = -x  # -45 degree line
    line_color = plt.rcParams['lines.color']

    # Add the lines in segments
    for i in range(0, len(x), 150):
        ax.plot(x[i:i+50], y1[i:i+50], color=line_color, linestyle='--')
        ax.plot(x[i:i+50], y2[i:i+50], color=line_color, linestyle='--')

def plot_xy_axes(ax, line_range=2):
    # Create a range of x values
    x = np.linspace(0, line_range, 50)

    # Create y values for each line
    y = np.zeros_like(x)
    
    # in order to make datk plots in dark mode
    line_color = plt.rcParams['lines.color']

    # Add the lines in segments
    for i in range(0, len(x), 100):
        ax.plot(x[i:i+50], y[i:i+50], color=line_color, linestyle='-', linewidth=0.2)
        ax.plot(y[i:i+50], x[i:i+50], color=line_color, linestyle='-', linewidth=0.2)

def plot_relu_lines(lorenz, ax, line_range=1.7):
     if lorenz:
          plot_uv_axes(ax, line_range)
     else:
          plot_xy_axes(ax, line_range)

def plot_2dgraph(graph,lorenz_fig_lims, proj_dims=[0,1],community_affiliation=None, test_mask=None, x_fig_lim=None, y_fig_lim=None, ax=None, figsize=(2,2)):
        
        if x_fig_lim is None:
            if lorenz_fig_lims:
                x_fig_lim = [-0.01, 2.7]
                y_fig_lim = [-1.7, 1.7]
            else:
                x_fig_lim = [-0.1, 2]
                y_fig_lim = [-0.1, 2]
        graph_cpu = graph.clone().to('cpu')
        if y_fig_lim is None:
            y_fig_lim = x_fig_lim
        node_feats = graph_cpu.x.detach().numpy()[:, proj_dims]
        num_nodes = node_feats.shape[0]
        node_positions_dict = {i: feat for i, feat in enumerate(node_feats)}
        color_list = sns.color_palette("Paired", n_colors=graph_cpu.x.shape[0])
        node_sizes = degree(graph_cpu.edge_index[0]).detach().numpy()
        G = to_networkx(graph_cpu)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        # COMMUNITY AFFILIATION
        if (community_affiliation is not None) and (community_affiliation.numel() != 0):
            node_colors = community_affiliation_to_colors(community_affiliation)

        else:
            node_colors = sns.color_palette("Paired", n_colors=graph_cpu.x.shape[0])
        
        edge_color = 'black' 
        nx.draw(G, pos=node_positions_dict, node_color=node_colors, node_size=5*node_sizes*figsize[0]/(num_nodes), arrows=False, edge_color=edge_color, width=0.01, ax=ax)
     
        #* add the axes (nx doesn't use them ever)
        ax.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        
        ax.set_title(f'feat {proj_dims[0]} vs feat {proj_dims[1]}')
        ax.set_aspect('equal')
        ax.set_xlim(x_fig_lim[0], x_fig_lim[1])
        ax.set_ylim(y_fig_lim[0], y_fig_lim[1])
        ax.set_aspect('equal')

        plot_relu_lines(lorenz=lorenz_fig_lims, ax=ax)

        if test_mask is not None:
            test_mask = test_mask.to(graph_cpu.x.device)
            test_points = graph_cpu.x[test_mask == 1]
            for point in test_points:
                ax.add_patch(plt.Circle((point[proj_dims[0]], point[proj_dims[1]]), radius=0.1, color='red', fill=False, zorder=10))
        return ax

def community_affiliation_to_colors(community_affiliation):
    '''converts the community affiliation to colors'''
    num_nodes = community_affiliation.shape[0]
    community_affiliation = community_affiliation.cpu().detach()

    # Separate anomalous and non-anomalous rows
    anomalous_indices = torch.all(community_affiliation == 1, dim=1)
    non_anomalous_indices = ~anomalous_indices
    anomalous_affiliation = community_affiliation[anomalous_indices]
    non_anomalous_affiliation = community_affiliation[non_anomalous_indices]

    # Convert non-anomalous affiliation to numpy array for PCA
    non_anomalous_affiliation_np = non_anomalous_affiliation.numpy()

    # Run PCA on non-anomalous rows
    n_components = min(3, non_anomalous_affiliation_np.shape[1])
    pca = PCA(n_components=n_components)
    colors = pca.fit_transform(non_anomalous_affiliation_np)

    # Normalize colors
    new_min = 0.3
    new_max = 1.0
    global_min = colors.min()
    global_max = colors.max()
    colors = new_min + (colors - global_min) * (new_max - new_min) / (global_max - global_min)

    # Assign bright red to anomalous rows
    anomalous_colors = np.zeros((anomalous_affiliation.shape[0], 3))

    # Combine colors
    node_colors = np.zeros((num_nodes, 3))
    node_colors[anomalous_indices.cpu().numpy()] = anomalous_colors
    node_colors[non_anomalous_indices.cpu().numpy()] = colors

    return node_colors


def plot_prob(log_prob_fn, device, zz=None, ax=None, figsize=(3,3), x_fig_lim=[0, 2], y_fig_lim=None, title=''):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if zz is None:
        grid_size = 200
        xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2).to(device)
    
    if y_fig_lim is None:
        y_fig_lim = x_fig_lim

    else:
        grid_size = int(np.sqrt(zz.shape[0]))
        xx = zz[:,0].view(grid_size, grid_size).cpu().detach().numpy()
        yy = zz[:,1].view(grid_size, grid_size).cpu().detach().numpy()
    # Calculate the values of the distribution function on zz
    #! i don't understand, the output changes depending on what i put there?
    log_prob = log_prob_fn(zz, sum=False)
    
    # log_prob = log_prob_fn(zz, sum=False)
    prob = torch.exp(log_prob.to('cpu').view(*xx.shape)).detach().numpy()
    # prob[torch.isnan(prob)] = 0

    im = ax.pcolormesh(xx, yy, prob, cmap='viridis')
    ax.set_xlim(x_fig_lim[0], x_fig_lim[1])
    ax.set_ylim(y_fig_lim[0], y_fig_lim[1])
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_aspect('equal')
    return im

def plot_normflows_dist(dist, device,shift, scale, zz=None, ax=None, figsize=(2,2), x_fig_lim=[0, 2], y_fig_lim=None, title=''):
    '''since we shift the distribution for sampling, we need to shift the plot as well'''
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if zz is None:
        grid_size = 200
        xx, yy = torch.meshgrid(torch.linspace(-2, 2, grid_size), torch.linspace(-2, 2, grid_size))
        xx = xx - 0.5
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2).to(device)
    
    if y_fig_lim is None:
        y_fig_lim = x_fig_lim

    else:
        grid_size = int(np.sqrt(zz.shape[0]))
        xx = zz[:,0].view(grid_size, grid_size).cpu().detach().numpy()
        yy = zz[:,1].view(grid_size, grid_size).cpu().detach().numpy()
    # Calculate the values of the distribution function on zz
    #! i don't understand, the output changes depending on what i put there?
    rotation_matrix = torch.tensor([[-1, 1], [-1, -1]]).float().to(zz.device)

    zz = torch.matmul(zz, rotation_matrix)
    log_prob = dist.log_prob(((zz)))
    
    # log_prob = log_prob_fn(zz, sum=False)
    prob = torch.exp(log_prob.to('cpu').view(*xx.shape)).detach().numpy()
    # prob[torch.isnan(prob)] = 0

    im = ax.pcolormesh(xx, yy, prob, cmap='viridis')
    ax.set_xlim(x_fig_lim[0], x_fig_lim[1])
    ax.set_ylim(y_fig_lim[0], y_fig_lim[1])
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_aspect('equal')
    return im


def plot_auc_dict(auc_dict):
    '''plot the auc_dict'''
    plt.figure(figsize=(4, 3))
    for key, value in auc_dict.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC vs Epoch')
    plt.show()

#! maybe make the first flow a hyperbolic transformation
def plot_losses(losses_first, losses_second=None, figsize=(2,4)):
    '''plotting the losses feats and losses prior after the fit function'''
    # losses should be on cpu as they are lists.
    if losses_second is not None:
        _, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(losses_first)
        axes[0].set_title('losses first')
        axes[1].plot(losses_second)
        axes[1].set_title('losses second')
    else:
        plt.figure(figsize=figsize)
        plt.plot(losses_first)
        plt.title('losses first')

def plot_trans_gridlines(transformation, bounds=[-1,1], dx=0.1, bounds_y=None, dy=None, plot_codes=False, scale_positives=True, figsize=(5,5),bounds_fig=[-0.2, 2], bounds_fig_y=None, title='', ax=None):
    '''plots the gridlines under a transformation
    #*bounds is the figure bounds for a square figure. for a rectangle, use bounds_y
    
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if bounds_fig_y is None:
        bounds_fig_y = bounds_fig

    if bounds_y is None:
        bounds_y = bounds
    if dy is None:
        dy = dx
    xs, ys = torch.meshgrid(torch.arange(bounds[0], bounds[1], dx), torch.arange(bounds_y[0], bounds_y[1], dy))
    grid = torch.stack([xs, ys], dim=2)
    lines_constant_x = [grid[:,i,:] for i in range(grid.shape[1])]
    lines_constant_y = [grid[i,:,:] for i in range(grid.shape[0])]

    trans_lines_x = [transformation(lines_constant_x[i]) for i in range(len(lines_constant_x))]
    trans_lines_y = [transformation(lines_constant_y[i]) for i in range(len(lines_constant_y))]

    cmap = get_cmap('Blues')  # Colormap for blue
    for i in range(len(trans_lines_x)):
        color = cmap(i / len(trans_lines_x), alpha=2)  # Calculate color based on index and adjust alpha value
        ax.plot(trans_lines_x[i][:,0].detach().numpy(), trans_lines_x[i][:,1].detach().numpy(), color=color, linewidth=2)
    
    cmap = get_cmap('Reds')  # Colormap for red
    for i in range(len(trans_lines_y)):
        color = cmap(i / len(trans_lines_y), alpha=2)  # Calculate color based on index and adjust alpha value
        ax.plot(trans_lines_y[i][:,0].detach().numpy(), trans_lines_y[i][:,1].detach().numpy(), color=color, linewidth=2)

    ax.set_aspect('equal')
    ax.set_xlim([bounds_fig[0], bounds_fig[1]])
    ax.set_ylim([bounds_fig_y[0], bounds_fig_y[1]])

def plot_feature_prediction_adj(graph, lorenz, prior=None, ax=None, figsize=(3,3), x_fig_lim=[-0.1, 2], y_fig_lim=None, title=''):
    '''plot the probability graph of the graph'''
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if y_fig_lim is None:
        y_fig_lim = x_fig_lim
    if hasattr(graph.x, 'device'):
        if graph.x.device != 'cpu':
            graph_cpu = graph.clone().to('cpu')
    else:
        graph_cpu = graph.clone().to('cpu')
    w_opt = get_prob_graph(graph_cpu.x, lorenz=lorenz, prior=prior)
    w_cut = w_opt.clone()
    w_cut[w_cut<0] = 0
    im = ax.imshow(w_cut.detach().numpy())
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')
    # ax.set_xlim(x_fig_lim[0], x_fig_lim[1])
    # ax.set_ylim(y_fig_lim[0], y_fig_lim[1])
    ax.set_aspect('equal')

def plot_optimization_stage(
                prior, 
                graph, 
                lorenz, 
                things_to_plot=['adj','feats', '2dgraphs', 'losses'], 
                community_affiliation=None, 
                dyads_to_omit=None, 
                losses=None, 
                i=0, 
                n_iter=0,  
                calling_function_name=''):
    
    '''plot various figures of the situation of the graph in the optimization'''
    if graph.x.shape[1] > 6 and 'feats' in things_to_plot:
        things_to_plot.remove('feats')
        printd('too many features to plot')

    graph_cpu = graph.clone().to('cpu')
    if community_affiliation is not None:
        community_affiliation_cpu = community_affiliation.clone().to('cpu')
    else:
        community_affiliation_cpu = None

    printd(f'\n{calling_function_name= } : iter {i+1}/{n_iter}')
    if prior:
        prior.model.eval()
    
    num_feats = graph_cpu.x.shape[1]   

    # plot SBM and original adj
    if 'adj' in things_to_plot:
        w_opt = get_prob_graph(graph_cpu.x, lorenz=lorenz, prior=prior)
        w_gt = to_dense_adj(graph_cpu.edge_index)[0]
        w_cut = w_opt.clone()

        w_cut[w_cut<0] = 0
        fig1, axes1 = plt.subplots(1,2)
        im_opt = plot_adj(w_cut, dyads_to_omit, ax=axes1[0])
        im_gt = plot_adj(w_gt, ax=axes1[1])

        axes1[0].set_title('optimized adj')
        axes1[1].set_title('ground truth adj')

        my_colorbar(im_opt, ax=axes1[0])
        my_colorbar(im_gt, ax=axes1[1])

        plt.subplots_adjust(wspace=0.5)
        
    # plot the features
    if 'feats' in things_to_plot:
        plot_feats(graph_cpu)
    # plot 2d graphs
    
    if '2dgraphs' in things_to_plot:
        if graph_cpu.x.shape[1] > 2:

            num_rows = math.ceil(num_feats / 6)
            num_cols = 3
            fig3, axes3 = plt.subplots(num_rows, num_cols)

            if type(axes3) != np.ndarray:
                axes3 = np.array([axes3])
            
            if axes3.ndim == 1:
                axes3 = np.expand_dims(axes3, axis=0)

            if graph_cpu.x.shape[1] % 2 == 0:
                # even number of features
                for j in range(num_feats // 2):
                    row = j // num_cols
                    col = j % num_cols
                    plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, proj_dims=[j, j + num_feats // 2], lorenz_fig_lims=lorenz, ax=axes3[row, col]) 
                    # plot_relu_lines(lorenz=lorenz, ax=axes3[row, col])

            else:
                # odd number of features
                # (this is only for clam)
                for j in range(0, num_feats - 1, 2):
                    row = (j //2) //num_cols
                    col = (j// 2) % num_cols
                    plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, proj_dims=[j, j + 1], lorenz_fig_lims=lorenz, ax=axes3[row, col]) 
                    plot_relu_lines(lorenz=lorenz, ax=axes3[row, col])
                    #todo: plot conditional probability on the plane. can test this when have prior say now
                
                # the last two features
                row = (num_feats // 2) // num_cols
                col = (num_feats // 2) % num_cols
                plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, proj_dims=[num_feats - 2, num_feats - 1], lorenz_fig_lims=lorenz, ax=axes3[row, col])
                plot_relu_lines(lorenz=lorenz, ax=axes3[row, col])
            

        else: #* 2 features
            
            if prior: 
                if lorenz:
                    x_fig_lim = [-0.01, 2.7]
                    y_fig_lim = [-1.5, 2]
                else:
                    x_fig_lim = [-0.1, 2]
                    y_fig_lim = [-0.1, 2]   
                _, axes = plt.subplots(1, 2, figsize=(7, 3))
                

                # plot prior with the 2d graph
                # plot_prob(prior.model.log_prob, device=next(prior.model.parameters()).device, ax=axes[0], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
                plot_prob(prior.forward_ll, device=next(prior.model.parameters()).device, ax=axes[0], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
                
                plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, lorenz_fig_lims=lorenz, ax=axes[0], figsize=(3,3))
                
                plot_xy_axes(axes[0], line_range=2)
                
                # plot just the prior
                # im_prob = plot_prob(prior.model.log_prob, device=next(prior.model.parameters()).device, ax=axes[1], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
                
                im_prob = plot_prob(prior.forward_ll, device=next(prior.model.parameters()).device, ax=axes[1], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
                
                my_colorbar(im_opt, ax=axes[1])
                plot_xy_axes(axes[1], line_range=2)

            else:
                plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, lorenz_fig_lims=lorenz, figsize=(3,3))
                plot_relu_lines(lorenz=lorenz, ax=plt.gca())
                
    
    #* plot losses
    if 'losses' in things_to_plot:
        if i > 0 and losses is not None:    
            plt.figure(figsize=[3,3])
            plt.figure(figsize=[3,3])
            plt.plot(losses)
            plt.ylim(bottom=-100)
            plt.plot(losses)
            plt.title(f'losses for function: {calling_function_name}')
        
    plt.show()
    if prior:
        prior.model.train()   

# def plot_test_accuracies(acc_test, n_iter_feats, n_iter_prior, n_back_forth, figsize=(9,3)):
#     n_iter_total = n_iter_feats + n_iter_prior
#     feat_stops = [n_iter_feats - 1 + i*n_iter_total for i in range(n_back_forth)]
#     prior_stops = [n_iter_total*i - 1 for i in range(1, n_back_forth+1)]

#     _, axes = plt.subplots(1, 3, figsize=figsize)
    
#     # Plot for 'vanilla_star'
#     axes[0].plot(acc_test['vanilla_star'])
#     for stop in feat_stops:
#         axes[0].axvline(x=stop, color='blue', linestyle='--', linewidth=1, alpha=0.5)
#     for stop in prior_stops:
#         axes[0].axvline(x=stop, color='red', linestyle='--', linewidth=1, alpha=0.5)
#     axes[0].set_title('vanilla')

#     # Plot for 'prior'
#     axes[1].plot(acc_test['prior'])
#     for stop in feat_stops:
#         axes[1].axvline(x=stop, color='blue', linestyle='--', linewidth=1, alpha=0.5)
#     for stop in prior_stops:
#         axes[1].axvline(x=stop, color='red', linestyle='--', linewidth=1, alpha=0.5)
#     axes[1].set_title('prior')

#     # Plot for 'prior_star'
#     axes[2].plot(acc_test['prior_star'], label='test')
#     for stop in feat_stops:
#         axes[2].axvline(x=stop, color='blue', linestyle='--', linewidth=1, alpha=0.5)
#     for stop in prior_stops:
#         axes[2].axvline(x=stop, color='red', linestyle='--', linewidth=1, alpha=0.5)
#     axes[2].set_title('prior_star')

#     plt.show()



def plot_test_accuracies(
                acc_test, 
                n_iter_1st, 
                n_iter_2nd, 
                n_back_forth, 
                figsize=3):
    
    n_lists = len(acc_test)
    figsize = (figsize * n_lists, figsize)  # Make the figure wider if there are more lists
    n_iter_total = n_iter_1st + n_iter_2nd
    feat_stops = [n_iter_1st - 1 + i*n_iter_total for i in range(n_back_forth)]
    prior_stops = [n_iter_total*i - 1 for i in range(n_back_forth+1)]


    _, axes = plt.subplots(1, n_lists, figsize=figsize)
    for i, (key, value) in enumerate(acc_test.items()):
        axes[i].plot(value)
        for stop in feat_stops:
            axes[i].axvline(x=stop, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            # axes[i].plot(stop, acc_test[key][stop], 'bo', markersize=7)  
        for stop in prior_stops:
            axes[i].axvline(x=stop, color='red', linestyle='--', linewidth=1, alpha=0.5)
            # axes[i].plot(stop, acc_test[key][stop], 'ro', markersize=7)  

        axes[i].set_title(f'{key}')
    
    plt.tight_layout()
    plt.show()




# def plot_test_accuracies(acc_test, n_iter_feats, n_iter_prior, n_back_forth, figsize=(9,3)):
#     n_iter_total = n_iter_feats + n_iter_prior
#     feat_stops = [n_iter_feats - 1 + i*n_iter_total for i in range(n_back_forth)]
#     prior_stops = [n_iter_total*i - 1 for i in range(1, n_back_forth+1)]

#     _, axes = plt.subplots(1, 3, figsize=figsize)
#     axes[0].plot(acc_test['vanilla_star'])
#     for stop in feat_stops:
#         axes[0].plot(stop, acc_test['vanilla_star'][stop], 'bo', markersize=7)

#         # Plot large red points at prior_stops
#     for stop in prior_stops:
#         axes[0].plot(stop, acc_test['vanilla_star'][stop],'ro', markersize=7)

#     axes[0].set_title('vanilla')

#     axes[1].plot(acc_test['prior'])
#     for stop in feat_stops:
#         axes[1].plot(stop, acc_test['prior'][stop], 'bo', markersize=7)

#         # Plot large red points at prior_stops
#     for stop in prior_stops:
#         axes[1].plot(stop, acc_test['prior'][stop],'ro', markersize=7)

#     axes[1].set_title('prior')


#     axes[2].plot(acc_test['prior_star'], label='test')
#     for stop in feat_stops:
#         axes[2].plot(stop, acc_test['prior_star'][stop], 'bo', markersize=7)

#         # Plot large red points at prior_stops
#     for stop in prior_stops:
#         axes[2].plot(stop, acc_test['prior_star'][stop],'ro', markersize=7)

#     axes[2].set_title('prior_star')
#     plt.show()

def plot_sparse_adj(edge_index, dyads_to_omit=None, test_index=None, test_mask=None, ax=None, figsize=(3,3), title=''):
    W = to_dense_adj(edge_index)[0]
    return plot_adj(W, dyads_to_omit, test_index, test_mask, ax, figsize, title)


def plot_adj(w, dyads_to_omit=None, test_index=None, test_mask=None, ax=None, figsize=(3,3), title=''):
    
    if w.shape[0] == 2 and w.shape[1] != 2:
        w = to_dense_adj(w)[0]

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(w.detach().cpu().numpy())



    if dyads_to_omit is not None:
        dyads_to_omit_np = (dyads_to_omit[0].detach().cpu().numpy(), dyads_to_omit[1].detach().cpu().numpy())
        ax.scatter(dyads_to_omit_np[0][0], dyads_to_omit_np[0][1], color='red', s=5/w.shape[1])
        ax.scatter(dyads_to_omit_np[1][0], dyads_to_omit_np[1][1], color='red', s=5/w.shape[1])

    if test_mask is not None:
        test_index = torch.where(test_mask)[0]

    if test_index is not None:
            test_nodes_np = test_index.detach().cpu().numpy()
            for node in test_nodes_np:
                ax.scatter(node, 0, color='red', s=10000/w.shape[1])  # Mark at the beginning of the row
                ax.scatter(0, node, color='red', s=10000/w.shape[0])  # Mark at the top of the column
    
    ax.set_title(title)
    
    return im
 
def my_colorbar(mappable, ax=None, **kwargs):
    if ax is None:
        ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)

def plot_feature_pairs(graph, lorenz, prior=None, test_mask=None, axes=None):
    '''plot one feature from the first half against one feature from the second half. if there is an odd number of features there will be something like 15 vs. 16 and 16 vs. 17'''
    community_affiliation_cpu = graph.y if hasattr(graph, 'y') else None
    graph_cpu = graph.clone().to('cpu')
    num_feats = graph_cpu.x.shape[1]

    if num_feats > 2:

        num_rows = math.ceil(num_feats / 6)
        num_cols = 3
        fig3, axes = plt.subplots(num_rows, num_cols)

        if type(axes) != np.ndarray:
            axes = np.array([axes])
        
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=0)

        if graph_cpu.x.shape[1] % 2 == 0:
            # even number of features
            for j in range(num_feats // 2):
                row = j // num_cols
                col = j % num_cols
                plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, test_mask=test_mask, proj_dims=[j, j + num_feats // 2], lorenz_fig_lims=lorenz, ax=axes[row, col]) 
                # plot_relu_lines(lorenz=lorenz, ax=axes3[row, col])

        else:
            # odd number of features
            # (this is only for clam)
            for j in range(0, num_feats - 1, 2):
                row = (j //2) //num_cols
                col = (j// 2) % num_cols
                plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, test_mask=test_mask, proj_dims=[j, j + 1], lorenz_fig_lims=lorenz, ax=axes[row, col]) 
                plot_relu_lines(lorenz=lorenz, ax=axes[row, col])
            
            # the last two features
            row = (num_feats // 2) // num_cols
            col = (num_feats // 2) % num_cols
            plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, test_mask=test_mask, proj_dims=[num_feats - 2, num_feats - 1], lorenz_fig_lims=lorenz, ax=axes[row, col])
            plot_relu_lines(lorenz=lorenz, ax=axes[row, col])
        

    else: #* 2 features
        
        if prior: 
            if lorenz:
                x_fig_lim = [-0.01, 2]
                y_fig_lim = [-1.5, 2]
            else:
                x_fig_lim = [-0.1, 2]
                y_fig_lim = [-0.1, 2]   
            _, axes = plt.subplots(1, 2, figsize=(7, 3))
            

            # plot prior with the 2d graph
            # plot_prob(prior.model.log_prob, device=next(prior.model.parameters()).device, ax=axes[0], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
            plot_prob(prior.forward_ll, device=next(prior.model.parameters()).device, ax=axes[0], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
            
            plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, lorenz_fig_lims=lorenz, ax=axes[0], test_mask=test_mask, figsize=(3,3))
            
            plot_xy_axes(axes[0], line_range=2)
            
            # plot just the prior
            # im_prob = plot_prob(prior.model.log_prob, device=next(prior.model.parameters()).device, ax=axes[1], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
            
            im_prob = plot_prob(prior.forward_ll, device=next(prior.model.parameters()).device, ax=axes[1], title='prior', x_fig_lim=x_fig_lim, y_fig_lim=y_fig_lim)
            
            plot_xy_axes(axes[1], line_range=2)

        else:
            plot_2dgraph(graph_cpu, community_affiliation=community_affiliation_cpu, lorenz_fig_lims=lorenz, test_mask=test_mask, figsize=(3,3))
            plot_relu_lines(lorenz=lorenz, ax=plt.gca())
    

def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_edge_index_nx(edge_index, ax=None):
    G = nx.Graph()
    G.add_edges_from(edge_index.T.numpy())
    if ax is None:
        _, ax = plt.subplots(figsize=(3,3))
    nx.draw(G, ax=ax)
    ax.set_aspect('equal')