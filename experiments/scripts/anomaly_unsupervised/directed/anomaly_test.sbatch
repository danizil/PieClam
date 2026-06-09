#!/bin/bash
#SBATCH --job-name directed_wo_densify_
#SBATCH -o directed_wo_densify_%j.log
#SBATCH --partition=cpu-gpu-v100        # Partition name
#SBATCH --gres=gpu:1                 # Request 1 GPU

nvcc --version
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyg
# echo "Submitting job with model=$ms, dataset=$ds, dim_feat=$dim_feat, noise_amp=$noise_amp, l1_reg=$l1_reg, n_iter_prior=$n_iter_prior, n_iter_feat=$n_iter_feat, lr_prior=$lr_prior, lr_feat=$lr_feat, first_func_in_fit=$first_func_in_fit"
       
python -u experiments/scripts/anomaly_unsupervised/directed/hypers_search.py --model_name 'bigclam' --ds_name 'disney' --dim_feats 48 100 200 --lr_feats 0.0000005 0.000001 0.000005 --n_iters_feats 2000 5000 7000 --n_reps 3 --pow_in 0.05 --pow_out 0.05 --init_types 'from_attr' --num_coupling_blocks 4 8 16 32 --n_iters_prior 1000 2000 --lr_prior 0.00001 0.000005 --noise_amps 0.01 0.001 0.005 --first_funcs_in_fit 'fit_prior' 'fit_feats' --n_back_forth 5