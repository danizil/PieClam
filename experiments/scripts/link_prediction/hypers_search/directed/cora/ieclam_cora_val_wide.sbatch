#!/bin/bash

#SBATCH --job-name=ic-co-vw  # Job name
#SBATCH --output=out/ic-co-vw.out # Name of stdout output file
#SBATCH --error=err/ic-co-vw.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=11            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate clam                  

python -u ~/ICML_PIECLAM_020425/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'ieclam' --ds_name 'cora' --dim_feats 50 100 200 --s_reg 0.0 0.1 --lr_feats 0.0000005 0.000005 --n_iters_feats 2000 5000 7000  --n_reps 3 --test_only --random_search --num_draws_random 50


