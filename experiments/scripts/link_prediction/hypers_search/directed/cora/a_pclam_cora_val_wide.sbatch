#!/bin/bash
#SBATCH --job-name directed_wo_densify_
#SBATCH -o directed_wo_densify_%j.log
#SBATCH --partition=cpu-gpu-v100        # Partition name
#SBATCH --gres=gpu:1                 # Request 1 GPU

nvcc --version
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pieclam_directed
# echo "Submitting job with model=$ms, dataset=$ds, s_reg=$s_reg, dim_feat=$dim_feat, noise_amp=$noise_amp, l1_reg=$l1_reg, n_iter_prior=$n_iter_prior, n_iter_feat=$n_iter_feat, lr_prior=$lr_prior, lr_feat=$lr_feat, first_func_in_fit=$first_func_in_fit"
       

python -u ~/Github/CoraCiteseer/PieClam/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'pclam' --ds_name 'cora' --dim_feats 48 100 200 --s_reg 0.0 --lr_feats 0.0000005 0.000001 0.000005 --n_iters_feats 500 1000 3000 --lr_prior 0.0000005 0.000001 0.000005 --n_iters_prior 500 1000 3000 --first_funcs_in_fit fit_feats fit_prior --n_reps 3 --random_search --num_draws_random 50 --test_only


