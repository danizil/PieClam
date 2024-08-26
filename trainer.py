import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, sort_edge_index, is_undirected, to_dense_adj
import torch_geometric
from torch_geometric.transforms import RemoveDuplicatedEdges, TwoHop
import numpy as np
from collections import OrderedDict
import os
import random
import json
from itertools import product
import tqdm
import matplotlib.pyplot as plt
import time
from transformation import train_prior
from datasets.import_dataset import import_dataset, transform_attributes
import clamiter as ci
import community_allocation as ca
from utils.plotting import plot_optimization_stage, plot_2dgraph
from utils.printing_utils import printd
from utils import utils
# from tests import tests
import json
import yaml
from copy import deepcopy




#  dP""b8 888888 88b 88 888888 88""Yb    db    888888  dP"Yb  88""Yb .dP"Y8 
# dP   `" 88__   88Yb88 88__   88__dP   dPYb     88   dP   Yb 88__dP `Ybo." 
# Yb  "88 88""   88 Y88 88""   88"Yb   dP__Yb    88   Yb   dP 88"Yb  o.`Y8b 
#  YboodP 888888 88  Y8 888888 88  Yb dP""""Yb   88    YbodP  88  Yb 8bodP'

#* clamiter parameters
#! we should make a different parameter set for iegam and bigclam, also, if the iegam dim feats is odd, add 1 to it so it's even. or multiply by 2 to compare them... 
dim_feats_list = [6, 8, 10, 12, 14, 16, 18, 20]
s_reg_list = [0.0, 0.5, 1, 10, 50]
l1_regs_list = [0.0, 0.01, 0.05, 0.1]
num_coupling_blocks_list = [8, 16, 32, 64]
hidden_dims_list = [8, 16, 32, 64] 

#TODO: check to see how many 

#* feat optimization params
lr_feats_list = [0.00001, 0.00002, 0.00005]
n_iter_feats_list = [20000]
dropout_edge_list = [0.0]
dropout_node_list = [0.0,0.0]


#* prior params
#TODO: change network parameters
weight_decay_list = [1e-2, 1e-3, 1e-4, 1e-5]
n_iters_prior_list = [500, 700, 1000, 2000]
lr_prior_list = [0.000001, 0.000002, 0.000005]
noise_amps_list = [0.01, 0.05, 0.1, 0.2]

#* fit params
n_back_forths = [40, 50, 60]
scheduler_step_sizes = [8, 16, 32]
scheduler_gammas = [0.5, 0.2, 0.1, 0.05]

clamiter_params_od = OrderedDict([('dim_feat', dim_feats_list), 
                                  ('s_reg', s_reg_list), 
                                  ('l1_reg', l1_regs_list), 
                                  ('num_coupling_blocks', num_coupling_blocks_list), 
                                  ('hidden_dim', hidden_dims_list)])
feat_opt_params_od = OrderedDict([('lr', lr_feats_list), 
                              ('n_iter', n_iter_feats_list), 
                              ('dropout_edge', dropout_edge_list), 
                              ('dropout_node', dropout_node_list)])

prior_opt_params_od = OrderedDict([('weight_decay', weight_decay_list), 
                                   ('noise_amp', noise_amps_list),
                                    ('n_iter', n_iters_prior_list), 
                                    ('lr', lr_prior_list)])

fit_params_od = OrderedDict([('n_back_forth', n_back_forths), 
                             ('scheduler_step_size', scheduler_step_sizes),
                             ('scheduler_gamma', scheduler_gammas)])



param_od_list = [clamiter_params_od, feat_opt_params_od, prior_opt_params_od, fit_params_od]


#* VANILLA param ordered dict
clamiter_params_od_vanilla = OrderedDict([('dim_feat', dim_feats_list), 
                                  ('s_reg', s_reg_list), 
                                  ('l1_reg', l1_regs_list)])
feat_opt_params_od_vanilla = OrderedDict([('lr', lr_feats_list), 
                              ('n_iter', n_iter_feats_list), 
                              ('dropout_edge', dropout_edge_list), 
                              ('dropout_node', dropout_node_list)])

param_od_list_vanilla = [clamiter_params_od_vanilla, feat_opt_params_od_vanilla]
                                 




def coarse_grid_search_generator(param_od_list, use_extremes):
    
    if use_extremes:
        # Modify each list to only use extremes
        param_od_list = [OrderedDict((k, utils.extremes(v)) for k, v in od.items()) for od in param_od_list]

    all_params_combinations = [product(*params_od.values()) for params_od in param_od_list]
    for combination in product(*all_params_combinations):
        config_list = [dict(zip(params_od.keys(), values)) for params_od, values in zip(param_od_list, combination)]
        yield config_list
#todo: i want the config dict to be initialized somehow but for the values to be changed: set [parent][child][value]
def random_config_generator(use_extremes, vanilla):
    if vanilla:
        param_od_list = [clamiter_params_od, feat_opt_params_od]
    else:
        param_od_list = [clamiter_params_od, feat_opt_params_od, prior_opt_params_od, fit_params_od]
    if use_extremes:
        # Modify each list to only use extremes
        param_od_list = [OrderedDict((k, utils.extremes(v)) for k, v in od.items()) for od in param_od_list]

    while True:
        config_list = []
        for params_od in param_od_list:
            config = {key: random.choice(value) for key, value in params_od.items()}
            config_list.append(config)
        yield config_list




#  dP""b8  dP"Yb  88b 88 888888 88  dP""b8                                  
# dP   `" dP   Yb 88Yb88 88__   88 dP   `"                                  
# Yb      Yb   dP 88 Y88 88""   88 Yb  "88                                  
#  YboodP  YbodP  88  Y8 88     88  YboodP 

def set_config(configs_dict, parent, child, value):
    '''set the value of a parameter in the configs dict.'''
    #todo: if it's clamiter init that you change, initialize a new clamiter.

    if parent in configs_dict:
        if child in configs_dict[parent]:
            configs_dict[parent][child] = value
    return configs_dict


def set_multiple_configs(configs_dict, config_triplets):
    '''set multiple values in the configs dict.'''
    for config_triplet in config_triplets:
        configs_dict = set_config(configs_dict, config_triplet[0], config_triplet[1], config_triplet[2])
    return configs_dict


# 888888 88""Yb    db    88 88b 88 888888 88""Yb 
#   88   88__dP   dPYb   88 88Yb88 88__   88__dP 
#   88   88"Yb   dP__Yb  88 88 Y88 88""   88"Yb  
#   88   88  Yb dP""""Yb 88 88  Y8 888888 88  Yb 


class Trainer():
    ''' trainer trains a model on a dataset with different parameters.'''
    #* attribute transform and n_componens is given here.
    #*attr opt is given at init because the prior is different for the different optimizations.
    def __init__(self, 
                 model_name, 
                 task,
                 device, 
                 dataset_name=None, 
                 configs_dict=None,
                 mighty_configs_dict=False,
                 config_triplets_to_change=[], 
                 dataset=None, 
                 clamiter=None, 
                 prior=None,
                 attr_opt=False, # move to clamiter init configs
                 attr_transform='auto',
                 inflation_flow_name=None,
                 optimizer=None, 
                 scheduler=None):
        
        self.device = device
        self.task = task
        #! should i make "task" a member of clamiter?
        # SAFEGUARDS
        if not config_triplets_to_change and configs_dict is not None:
            printd('\n\nWARNING\nWarning: both config_triplets and configs_dict are given. configs_dict will be used.\nWARNING\n\n')

        # =================================
        # HYPERS
        # need to load the hypers from the yaml file. 
        # ================================
        # GET DATA
        if dataset is not None:
            self.data = dataset
            self.dataset_name = self.data.name
            if hasattr(self.data, 'x'):
                if self.data.x is not None:
                    # Check if there is a triplet with 'dim_feat' as the second element
                    for i, triplet in enumerate(config_triplets_to_change):
                        if triplet[1] == 'dim_feat':
                            # Remove the existing triplet
                            config_triplets_to_change.pop(i)
                            break

    # Append the new triplet
                    config_triplets_to_change.append(['clamiter_init', 'dim_feat', self.data.x.size(1)])
        else:
            self.dataset_name = dataset_name #should be a string
            self.data = import_dataset(self.dataset_name)
        self.data.communities_found = torch.tensor([]).to(self.device) # GPU 400 mib (maybe an empty tensor allocates more...)
        if not hasattr(self.data, 'edge_index_original'):
            
            self.data.edge_index_original = self.data.edge_index.clone() # still on cpu
            if self.data.edge_attr is None:
                self.data.edge_attr = torch.ones(self.data.edge_index.shape[1]).bool()

        # =========================================================
        
        # OPTIMIZER, SCHEDULER, VANILLA, LORENZ
        self.optimizer = optimizer
        self.scheduler = scheduler
        

        if model_name == 'iegam':
            self.vanilla = True
            self.lorenz = True
        elif model_name == 'bigclam':
            self.vanilla = True
            self.lorenz = False
        elif model_name == 'piegam':
            self.vanilla = False
            self.lorenz = True
        elif model_name == 'pclam':
            self.vanilla = False
            self.lorenz = False
        else:
            raise NotImplementedError('model name not implemented')
        
        
        self.model_name = model_name
        self.params_name = self.dataset_name + '_' + model_name
        # =====================================================
        
        # CONFIGS DICT
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.task is None:
            hypers_file_name = 'distance'
        else:
            hypers_file_name = self.task
        self.configs_path = os.path.join(dir_path, 'hypers', 'hypers_'+ hypers_file_name + '.yaml')

        if configs_dict is None:
            if mighty_configs_dict:
                self.get_mighty_configs_dict(config_triplets=config_triplets_to_change)
            else:
                '''if not mighty config then the individual dataset configs provide config triplets'''
                self.configs_dict_from_top_list(config_triplets=config_triplets_to_change)
        else:
            self.configs_dict = configs_dict
        # ====================================
        #todo: put attribute transform here?
        self.attr_opt = attr_opt
        if hasattr(self.data, 'raw_attr'):
            if 'dim_attr' in self.configs_dict['clamiter_init']:
                if self.configs_dict['clamiter_init']['dim_attr'] is not None:
                    #* if the given attr dim is smaller than the attr dim of the data
                    self.configs_dict['clamiter_init']['dim_attr'] = min(self.configs_dict['clamiter_init']['dim_attr'], self.data.raw_attr.shape[1])
        # CLAMITER
        if clamiter is not None:
            self.clamiter=clamiter

        else:
            self.clamiter = ci.PCLAMIter(
                    vanilla=self.vanilla, 
                    lorenz=self.lorenz, 
                    attr_opt=self.attr_opt,
                    device=self.device, 
                    inflation_flow_name=inflation_flow_name,
                    **self.configs_dict['clamiter_init'])
        if prior is not None:
            self.add_prior(prior)
            # add the prior config into the clamiter init
        # =====================================================
        
        # ATTRIBUTES
        #* vanilla doesn't need attrs so save the attr raw to the trainer that will use them in the future
        if self.attr_opt and not self.vanilla: # dim_attr is in the clamiter dict and transformation type is an input parameter 
            self.attr_transform = attr_transform
            if not hasattr(self.data, 'attr'):  
                #! put transform attributes before clamiter and then add the attribute dimension
                self.data.attr = transform_attributes(self.data.raw_attr, self.attr_transform, self.configs_dict['clamiter_init']['dim_attr'])
            #* delete raw_attr if it's not vanilla 
            if hasattr(self.data, 'raw_attr'):
                delattr(self.data, 'raw_attr')

        # =====================================================

        return
    
    @classmethod
    def from_path(cls, path_in_checkpoints, device=torch.device('cpu'), verbose=False):
        ''' create trainer from saved model. load model'''
        self = cls.__new__(cls)
        # Load the model into the Trainer object
        self.clamiter, self.data, self.configs_dict, self.model_name, self.dataset_name = ci.load_model(path_in_checkpoints, device, verbose)
        #! here they are on the same device
        self.set_device(device)
        return self
    
    @classmethod
    def copy_with_prior(cls,another_trainer, prior=None, config_triplets_to_change=[]):
        '''create a copy of a vanilla trainer and adds a prior.'''
        self = cls.__new__(cls)
        self = deepcopy(another_trainer)
        self.add_prior(prior) 
        return self
    


    def configs_dict_from_top_list(self, config_triplets=None):
        
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(dir_path, 'hypers.yaml'), 'r') as file:
        with open(self.configs_path, 'r') as file:
            params_dict = yaml.safe_load(file)
        
        
        configs_ds = deepcopy(params_dict[self.dataset_name+'_'+self.model_name])
        config_triplets_ds = [[outer_key, inner_key, inner_value] for outer_key, outer_value in configs_ds.items() for inner_key, inner_value in outer_value.items()]
        self.get_mighty_configs_dict(config_triplets=config_triplets_ds)
        
        if config_triplets:
            self.set_multiple_configs(config_triplets)


        # if self.vanilla:
        #     clamiter_config = deepcopy(params_dict['clamiter_params_dict'][self.params_name])
        #     feat_opt_config = deepcopy(params_dict['feat_opt_params_dict'][self.params_name])
        #     self.configs_dict = {'clamiter_init': clamiter_config, 
        #                     'feat_opt': feat_opt_config}
        # else:
        #     feat_opt_config = deepcopy(params_dict['feat_opt_params_dict'][self.params_name])
        #     clamiter_config = deepcopy(params_dict['clamiter_params_dict'][self.params_name])
        #     prior_config = deepcopy(params_dict['prior_opt_params_dict'][self.params_name])
        #     back_forth_config = deepcopy(params_dict['back_forth_params_dict'][self.params_name])


        #     self.configs_dict = {'clamiter_init': clamiter_config, 
        #                     'feat_opt': feat_opt_config, 
        #                     'prior_opt': prior_config, 
        #                     'back_forth': back_forth_config}
        # if config_triplets:
        #     self.set_multiple_configs(config_triplets)     
    
    def get_mighty_configs_dict(self, config_triplets=None):
        '''in hypers.yaml there are four mighty config dictionaries for each model in unsupervised learning.'''
        
        with open(self.configs_path, 'r') as file:
            params_dict = yaml.safe_load(file)
        
        self.configs_dict = deepcopy(params_dict['MightyConfigs'+ '_' + self.model_name])

        if config_triplets:
            self.set_multiple_configs(config_triplets)   
        return   
    
    def set_config(self, parent, child, value):
        '''set the value of a parameter in the configs dict.'''
        #todo: if changing clamiter init stuff need to set more things, i think that make these things protected....
        self.configs_dict = set_config(self.configs_dict, parent, child, value)
        return


    def set_multiple_configs(self, config_triplets):
        '''set multiple values in the configs dict.'''
        self.configs_dict = set_multiple_configs(self.configs_dict, config_triplets)
        return

    def train(self,
            init_type='small_gaus', 
            dyads_to_omit = None, 
            task_params=None,
            init_feats=False, 
            acc_every=20, 
            performance_metric=None,
            prior_fit_mask=None, 
            plot_every=1000, 
            verbose=True, 
            verbose_in_funcs=False,
            node_feats_for_init=None):
        
        '''train one of the 4 models (bool vanilla, bool lorenz) on the given parameters. 
        You can chose to omit dyads from the calculation, dyads are a tuple (edges_to_omit, non_edges_to_omit).
        If only params_name is given, train on the optimal parameters as saved.
        If both params_name and params_dicts are given, train on the given parameters.
        
        Args:  
        :prior_fit_mask: a subset of the nodes on which to train the prior. for GGAD setting.
        
        
        ===============
        returns: losses_feats, losses_prior, auc_scores, cutnorms
        '''
        #todo: print the classification score every few back and forth? in the fit functions? it does take some time...
        # SETUP AND INIT NODES
        if not verbose:
            verbose_in_funcs = False
        self.data.edge_index = self.data.edge_index_original 
        
        t_train_model = time.time()
        
        if self.configs_dict is None:
            raise ValueError(" in train_model_on_params: trainer doesn't have a config dict.")
        
        printd(f'\n starting optimization of {self.model_name} on {self.dataset_name} on device {self.device}')
        print('\n configs_dict: \n' + json.dumps(self.configs_dict, indent=4))
          
        if init_feats or (self.data.x is None):
            if verbose:
                printd(f'\n train_model_on_params, initializing feats with {init_type}')
            t = time.time()
            self.data.x = self.clamiter.init_node_feats(
                                            graph_given=self.data, 
                                            init_type=init_type, 
                                            node_feats_given=node_feats_for_init) #GPU nothing significant
            
            if verbose:
                printd(f'\n init_node_feats took {time.time() - t} seconds')
        # ====================================================================================
        
        # OMIT DYADS
        if dyads_to_omit:
            '''omitting dyads happens indirectly with an algebraic trick.'''
            self.data.edge_index, self.data.edge_attr = self.omit_dyads(dyads_to_omit)
        
        # ====================================================================================
        self.data.to(self.device) 
    
        # OPTIMIZATION
        try:
            # FIT VANILLA 
            if self.vanilla:
                losses_prior = None
                losses_feats, accuracies_test, accuracies_val = self.clamiter.fit_feats(
                        graph=self.data,
                        acc_every=acc_every,
                        task=self.task,
                        dyads_to_omit=dyads_to_omit, 
                        performance_metric=performance_metric,
                        plot_every=plot_every,
                        **self.configs_dict['feat_opt'], 
                        verbose=verbose_in_funcs or verbose)
                if verbose:
                    printd(f'\n train: finished vanilla fit feats')
                                   
                return losses_feats, accuracies_test, accuracies_val
            # =====================================================
            
            # FIT WITH PRIOR
            else:
                #  OPTIMIZER AND SCHEDULER 
                if not self.optimizer:
                    self.optimizer = torch.optim.Adam(
                            self.clamiter.prior.parameters(), 
                            lr=self.configs_dict['prior_opt']['lr'], 
                            weight_decay=self.configs_dict['prior_opt']['weight_decay'])
                
                if not self.scheduler:
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                            self.optimizer, 
                            step_size=self.configs_dict['back_forth']['scheduler_step_size'], 
                            gamma=self.configs_dict['back_forth']['scheduler_gamma'])

                fit_opt_params = {
                        'feat_params': self.configs_dict['feat_opt'],
                        'prior_params': self.configs_dict['prior_opt'],
                        'back_forth_params': self.configs_dict['back_forth'],
                        'n_back_forth': self.configs_dict['back_forth']['n_back_forth'],
                        'first_func_in_fit': self.configs_dict['back_forth']['first_func_in_fit'],
                        'early_stop_fit': self.configs_dict['back_forth']['early_stop_fit']
                        }
                
                losses, accuracies_test, accuracies_val = self.clamiter.fit(
                                        self.data, 
                                        optimizer=self.optimizer, scheduler=self.scheduler,
                                        dyads_to_omit=dyads_to_omit, 
                                        prior_fit_mask=prior_fit_mask,

                                        task=self.task, 
                                        acc_every=acc_every, 
                                        performance_metric=performance_metric,
                                        configs_dict=self.configs_dict,

                                        plot_final_res=False,
                                        plot_every=plot_every,
                                        verbose=verbose,
                                        verbose_in_funcs=verbose_in_funcs,
                                        **fit_opt_params)
               
                
                if verbose:
                    printd(f'\ntrain_model_on_params on {self.model_name} {self.dataset_name} \ntook {time.time() - t_train_model} seconds')
                
                printd(f'\n\n\nFINISHED train model on params \n\n\n')

                return losses, accuracies_test, accuracies_val
            # ===============================================================
            



        except (ValueError, AssertionError) as e:
            if verbose:
                printd(f'\nERROR in train_model_on_params: {e}')
            raise
            
        finally:
            self.data.edge_index = self.data.edge_index_original.to(self.device)
            self.data.edge_attr = torch.ones(self.data.edge_index.shape[1]).bool().to(self.device)
            
    def retrain_model(self, n_iter, plot_every=1000):
        self.data.to(self.device)
        try:    
            if self.vanilla:
                feat_opt_config= self.configs_dict['feat_opt']
                feat_opt_config['n_iter'] = n_iter
                losses_feats, auc_score = self.clamiter.fit_feats(self.data, **feat_opt_config)
                losses_prior = None
                auc_scores = [auc_score]
            else:
                feat_opt_config = self.configs_dict['feat_opt']
                prior_config = self.configs_dict['prior_opt']
                back_forth_config = self.configs_dict['back_forth']
                back_forth_config['n_back_forth'] = n_iter
                
                fit_opt_params = {
                        'feat_params': feat_opt_config,
                        'prior_params': prior_config,
                        'n_back_forth': back_forth_config['n_back_forth'],
                        }
                losses_feats, losses_prior, auc_scores = self.clamiter.fit(graph=self.data, optimizer=self.optimizer, scheduler=self.scheduler, plot_every=plot_every, **fit_opt_params)
        except (ValueError, AssertionError) as e:
            printd(f'ERROR in retrain_model: {e}')
            raise

    def add_prior(self, config_triplets_to_change=[], prior=None):
        '''add a prior to the model'''
        #todo: test that this works and move trainer and clamiter file to lab
        # CHANGE NAME AND VANILLA
        self.vanilla = False
        if self.model_name == 'pclam' or self.model_name == 'piegam':
            printd(f'\n model {self.modes_name} already has a prior')
        elif self.model_name == 'bigclam':
            self.model_name = 'pclam'
        elif self.model_name == 'iegam':
            self.model_name = 'piegam'
        # ===========================================
        # LOAD CONFIGS FROM TOP AND SET PRIOR DIMENSIONS IN CLAMITER.
        self.params_name = self.dataset_name + '_' + self.model_name
        if prior is not None:
            hidden_dim = prior.hidden_dim
            num_coupling_blocks = prior.num_coupling_blocks
            num_layers_mlp = prior.num_layers_mlp
            # add_prior_dict = {
            #     'prior': prior}
            config_triplets = [['clamiter_init', 'hidden_dim', hidden_dim],
                               ['clamiter_init', 'num_coupling_blocks', num_coupling_blocks],
                               ['clamiter_init', 'num_layers_mlp', num_layers_mlp]]
        self.configs_dict_from_top_list(
                config_triplets=config_triplets)
        
        self.clamiter = ci.PCLAMIter(vanilla=self.vanilla, 
                                     lorenz=self.lorenz, 
                                     **self.configs_dict['clamiter_init'])
        self.clamiter.add_prior(prior)

        
        

    def create_clamiter(self, ci_params):
        '''creates a clamiter object with the given parameters'''
        self.clamiter = ci.PCLAMIter(vanilla=self.vanilla, lorenz=self.lorenz, **ci_params)
    
    def set_device(self, device):
        '''set the device of the trainer'''
        self.device = device
        self.data.to(device)
        self.clamiter.to(device)
        
    def omit_dyads(self, dyads_to_omit):
        
        ''' this function prepares the data for node ommition. it adds the non edges to omit to the edges array and creates a boolean mask for the edges to omit.
        dyads_to_omit: (edges_to_omit, non_edges_to_omit). dropped dyads get the edge attr 0 and the retained edges get the edge attr 1.
        PARAM: dyads_to_omit: tuple 4 elements:'''
        
        assert len(dyads_to_omit) == 4, 'dyads_to_omit should be a tuple (edges_to_omit, non_edges_to_omit, edge_index_rearanged, edge_mask_rearanged)'
        assert dyads_to_omit[2].shape[1] == self.data.edge_index.shape[1], 'dyads_to_omit[2] should be the same as self.data.edge_index but rearanged'
        assert torch_geometric.utils.coalesce(dyads_to_omit[2]).shape[1] == self.data.edge_index.shape[1], 'dyads_to_omit[2] should be the same as self.data.edge_index but rearanged'



        omitted_dyads_tot = torch.cat([dyads_to_omit[0], dyads_to_omit[1]], dim=1)
        
        rearanged_edge_index = dyads_to_omit[2]
        rearanged_edge_index_with_omitted_non_edges = torch.cat([rearanged_edge_index, dyads_to_omit[1]], dim=1)
        edge_attr = torch.cat([dyads_to_omit[3], torch.zeros(dyads_to_omit[1].shape[1]).bool()])
        assert is_undirected(rearanged_edge_index_with_omitted_non_edges), 'edges in dyads_to_omit should be undirected'
        assert (rearanged_edge_index_with_omitted_non_edges[:, ~edge_attr] == omitted_dyads_tot ).all(), 'edge_attr should be 0 for omitted dyads'
        assert rearanged_edge_index_with_omitted_non_edges.shape[1] == self.data.edge_index.shape[1] + dyads_to_omit[1].shape[1], 'rearanged_edge_index_with_omitted_non_edges should have the same number of edges as the original edge_index + the non edges to omit'
        # so edge_attr == 0 for omitted edges and ==1 for non omitted
        return rearanged_edge_index_with_omitted_non_edges, edge_attr
    
    def determine_community_affiliation(self, clustering_method, clustering_param):
        '''determine the community affiliation of the nodes in x'''
        self.data.communities_found = ca.determine_community_affiliation(self.data.x, clustering_method, self.lorenz, clustering_param)
    
    def community_detection_metrics(self, dyads_to_omit=None, verbose=False):
        '''returns the performance metrics of the model'''
        test_no_duplicity(self.data.edge_index)

        if verbose:
            printd('')
            self.plot_state(dyads_to_omit)
        if self.data.communities_found.numel() != 0:
            return {'f1_with_gt': self.f1_with_gt(), 'omega_index': self.omega_index()}

    def f1_with_gt(self):
        '''compare the community affiliation to the ground truth y
        returns the average f1 score of the BEST FIT communities (the communities found in the optimization).'''
        return ca.f1_with_gt(self.data.communities_found, self.data.y)

    def omega_index(self):
        '''compute the omega index between the found communities and the ground truth communities'''
        return ca.omega_index(self.data.communities_found, self.data.y)
    

    def plot_state(self, dyads_to_omit=None, gt_or_found_communities='gt', things_to_plot=['adj', '2dgraphs', 'losses'], calling_function_name="Trainer.plot_state"):
        '''plots the state of the features and adjacency'''
        assert gt_or_found_communities in ['gt', 'found'], 'in trainer.plot_stategt_or_found_communities should be either gt or found'
        affiliation_to_plot = self.data.y if gt_or_found_communities == 'gt' else self.data.communities_found
        
        self.clamiter.plot_state(self.data, community_affiliation=affiliation_to_plot, dyads_to_omit=dyads_to_omit,things_to_plot=things_to_plot, calling_function_name=calling_function_name)

    def save_state(self, inner_folder='trainers', suffix=''):
        '''save trainer feats prior and config'''    
        model_save_path = f'{inner_folder}/{self.dataset_name}_{self.model_name}_{suffix}.pt'    
        utils.save_feats_prior_hypers(self.data, self.clamiter.prior, self.configs_dict, model_save_path, overwrite=True)
    
    
    


    
# 888888 888888 .dP"Y8 888888 .dP"Y8 
#   88   88__   `Ybo."   88   `Ybo." 
#   88   88""   o.`Y8b   88   o.`Y8b 
#   88   888888 8bodP'   88   8bodP' 


def test_no_duplicity(edge_index):
    adj = to_dense_adj(edge_index)[0]
    assert torch.max(adj) <= 1, 'there are duplicate edges in the graph'
    return True

def test_omitted_dyads_undirected(dyads_to_omit):
    if not is_undirected(dyads_to_omit[0]):
        raise ValueError('edges to omit should be undirected')
    if not is_undirected(dyads_to_omit[1]):
        raise ValueError('non edges to omit should be undirected')

