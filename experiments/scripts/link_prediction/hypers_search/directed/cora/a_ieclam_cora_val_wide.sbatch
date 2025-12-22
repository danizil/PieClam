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
       

python -u ~/Github/CoraCiteseer/PieClam/experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py --model_name 'ieclam' --ds_name 'cora' --dim_feats 48 100 200 --s_reg 0.0 0.1 --lr_feats 0.0000005 0.000005 --n_iters_feats 2000 5000 7000  --n_reps 3 --test_only --random_search --num_draws_random 50


