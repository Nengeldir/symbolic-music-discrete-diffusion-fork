#!/bin/bash
#SBATCH --job-name=music_diff_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=student  # Check if you need to specify a partition

# Initialize Conda (Assuming standard installation path, adjust if needed)
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate symbolic-music-discrete-diffusion

# OR if using modules on the cluster:
# module load gcc/8.2.0 python/3.10.4 cuda/11.8

echo "Starting training job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script
python train_all_models.py

echo "Job finished at $(date)"
