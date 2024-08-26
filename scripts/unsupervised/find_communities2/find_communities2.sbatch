#!/bin/bash

#SBATCH --job-name=it_ie  # Job name
#SBATCH --output=out.out # Name of stdout output file
#SBATCH --error=err.err  # Name of stderr error file
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=15            
#SBATCH --gres=gpu:1                  # Request one GPU

##SBATCH --partition=euler             # Partition name
##SBATCH --account=euler               # Account name
source ~/miniconda3/etc/profile.d/conda.sh                
conda activate finegrain                  

# Run the Python script with n_iter set to 1700
python -u find_communities2.py --model_name 'iegam' --ds_name 'redditGGAD' --ks 10 14 10 16 10 18 10 20 10 22 10 24 10 26 10 28 --n_iter 1700 

# Run the Python script with n_iter set to 2000
python -u find_communities2.py --model_name 'iegam' --ds_name 'redditGGAD' --ks 10 14 10 16 10 18 10 20 10 22 10 24 10 26 10 28 --n_iter 2000 

# Run the Python script with n_iter set to 2500
python -u find_communities2.py --model_name 'iegam' --ds_name 'redditGGAD' --ks 10 14 10 16 10 18 10 20 10 22 10 24 10 26 10 28 --n_iter 2500 