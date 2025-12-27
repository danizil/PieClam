#!/bin/bash

#SBATCH --job-name=pi-co-vw  # Job name
#SBATCH --output=out/out_pi-co-vw.out # Name of stdout output file
#SBATCH --error=err/err_pi-co-vw.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'cora-ml' --dim_feats 48 100 200 --s_reg 0.0 --lr_feats 0.0000005 0.000001 0.000005 --n_iters_feats 500 1000 3000 --lr_prior 0.0000005 0.000001 0.000005 --n_iters_prior 500 1000 3000 --num_coupling_blocks 28 32 42 --hidden_dim 50 64 74 --first_funcs_in_fit fit_feats fit_prior --n_reps 3 --random_search --num_draws_random 50 --test_only


