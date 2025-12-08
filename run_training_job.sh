#!/bin/bash
#SBATCH --job-name=music_diff_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=student 
#SBATCH --account deep_learning

# Initialize Conda (Assuming standard installation path, adjust if needed)
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate symbolic-music-discrete-diffusion

# OR if using modules on the cluster:

. /home/lconconi/jupyter/bin/activate
nvidia-smi
echo "Starting training job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script
python3 train_all_models.py

echo "Job finished at $(date)"
