#!/bin/bash

#SBATCH --job-name=pi-ci-vw  # Job name
#SBATCH --output=out/out_pi-ci-vw.out # Name of stdout output file
#SBATCH --error=err/err_pi-ci-vw.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pieclam' --ds_name 'citeseer' --dim_feats 50 100 200 --s_reg 0.0 --lr_feats 0.0000005 0.000005 --n_iters_feats 500 1000 3000 --lr_prior 0.000005 0.0000005 --n_iters_prior 500 1000 3000  --n_reps 3 --val_p 0.1 --test_p 0.0 --random_search --num_draws_random 50 --directed


