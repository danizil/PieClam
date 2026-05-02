#!/usr/bin/env python3

import torch
import numpy as np
import argparse
from datetime import datetime
import logging
from inspect import currentframe, getframeinfo
import os
import sys
import json


script_dir = os.path.dirname(os.path.realpath(__file__))

# Traverse up 4 levels and add each directory to sys.path
for _ in range(4):
    script_dir = os.path.dirname(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

from utils.printing_utils import printd, filename_n_line_str
import experiments.optimization_utils_directed as ou

from tests import tests
from utils import utils
from utils.plotting import *
# import anomaly_detection as ad
from scripting_utils import print_prior_training_stats
from datasets.import_dataset import import_dataset
import link_prediction as lp


def main():    
    '''run big and ie on a chosen dataset to find the optimal number of communities and number of iterations.'''

    
    #           ARGS
    #=================================
    #todo: this already has test set path. i need to do it in the batch file
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ieclam', help='name of the model')
    parser.add_argument('--ds_name', type=str, default='squirrel', help='name of the dataset')
    parser.add_argument('--init_types', nargs='+', type=str, default=['small_gaus'], help='type of initialization')

    # feat config triplet range
    #todo: make the values default to the global config values
    parser.add_argument('--dim_feats', nargs='+', type=int, default=[], help='community dimension')
    parser.add_argument('--l1_regs', nargs='+', type=float, default=[0.0], help='l1 regularization')
    parser.add_argument('--s_regs', nargs='+', type=float, default=[0.0], help='s regularization')
    parser.add_argument('--n_iters_feats', nargs='+', type=int, default=[], help='number of iterations in fit feats')
    parser.add_argument('--lr_feats', nargs='+', type=float, default=[2000], help='lr feats')
    
    #prior config triplet range
    parser.add_argument('--dim_attr', nargs='+', type=int, default=[100], help='attribute dimension')
    parser.add_argument('--n_iters_prior', nargs='+', type=int, default=[], help='number of iterations in fit prior')
    parser.add_argument('--lr_prior', nargs='+', type=float, default=[], help='lr prior')
    parser.add_argument('--noise_amps', nargs='+', type=float, default=[], help='noise amplitudes')
    parser.add_argument('--n_back_forth', nargs='+', type=int, default=[], help='number of back and forth iterations')
    parser.add_argument('--first_funcs_in_fit', nargs='+', type=str, default=[], help='first function in alternation')

    parser.add_argument('--use_global_config_base', action='store_true', help='whether to use the global config base') # if not given, use_global_config base is false
    parser.add_argument('--densify', action='store_true', help='whether to densify the data')
    parser.add_argument('--attr_opt', action='store_true', help='whether to optimize the attributes')
    parser.add_argument('--test_p', type=float, default=0.1, help='test proportion')
    parser.add_argument('--val_p', type=float, default=0.0, help='validation proportion')
    parser.add_argument('--val_dyads_path', type=str, default=None, help='path to the validation dyads')
    parser.add_argument('--test_dyads_path', type=str, default=None, help='path to the test dyads')
    parser.add_argument('--random_search', action='store_true', help='whether to use random search')
    parser.add_argument('--test_only', action='store_true', help='whether to test only')
    parser.add_argument('--n_reps', type=int, default=3, help='number of repetitions')
    parser.add_argument('--to_undirected', action='store_true', help='whether to use undirected data')
    parser.add_argument('--remove_self_loops', action='store_true', help='whether to remove self loops')

    # task selection
    # parser.add_argument('--task', type=str, default='link_prediction', choices=['link_prediction', 'anomaly_detection'], help='task to run')

    # anomaly-only args
    # parser.add_argument('--ds_names', nargs='+', type=str, default=['reddit', 'photo', 'elliptic'], help='datasets for multi_ds_anomaly')
    # parser.add_argument('--densifiable_ds', nargs='+', type=str, default=['reddit', 'photo'], help='densifiable datasets')
    parser.add_argument('--acc_every', type=int, default=50, help='evaluate accuracy every N iterations')
    parser.add_argument('--name', type=str, default=None, help='suffix name for the results file')

    args = parser.parse_args()
    

    # ========= RESULTS FOLDERS =========
    if not torch.cuda.is_available():
        # raise Exception('CUDA not available')
        printd('CUDA not available')
        device = torch.device('cpu')
    else:
        printd('CUDA available')
        device = torch.device('cuda')
    printd(f'Using device: {device}')


    range_triplets = [
        ['clamiter_init','s_reg', args.s_regs],
        ['clamiter_init','l1_reg', args.l1_regs],
        ['clamiter_init', 'dim_feat', args.dim_feats],
        ['clamiter_init', 'init_type', args.init_types],
        ['feat_opt','n_iter', args.n_iters_feats],
        ['feat_opt','lr', args.lr_feats],
    ]

    if args.model_name in ['pclam', 'pieclam']:
        range_triplets += [
            # ['clamiter_init','dim_attr', args.dim_attr],
            ['back_forth', 'first_func_in_fit', args.first_funcs_in_fit],
            ['prior_opt','n_iter', args.n_iters_prior],
            ['prior_opt','lr', args.lr_prior],
            ['prior_opt','noise_amp', args.noise_amps],
            ['back_forth','n_back_forth', args.n_back_forth]
        ]
    # Create the file if it doesn't exist
    #todo: test the datasets: photo, texas, facebook, squirrel and crocodile

    ou.multi_ds_anomaly(
        model_name=args.model_name,
        range_triplets=range_triplets,
        n_reps=args.n_reps,
        use_global_config_base=args.use_global_config_base,
        device=device,
        metric='auc',
        ds_names=[args.ds_name],
        densifiable_ds=[args.ds_name if args.densify else None],
        attr_opt=args.attr_opt,
        acc_every=args.acc_every,
        to_undirected=args.to_undirected,
        remove_self_loops=args.remove_self_loops,
        random_search=args.random_search,
        name=args.name,
        )

if __name__ == "__main__":
    main()

