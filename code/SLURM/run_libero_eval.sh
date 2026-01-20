#!/bin/bash
#SBATCH --job-name=libero_eval
#SBATCH --output=logs/libero_eval_%j.out
#SBATCH --error=logs/libero_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # Adjust to your cluster's GPU partition name

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Load necessary modules (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
source ~/.bashrc  # or wherever conda is initialized
conda activate upa-vla

# Navigate to project directory
cd /data1/ma1/Ishaq/ump-vla

# Setup headless rendering environment variables
# Method 1: Try OSMesa (software rendering - more compatible)
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM

# Method 2: If OSMesa doesn't work, try EGL (GPU rendering)
# Uncomment these if OSMesa fails:
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export EGL_PLATFORM=device
# export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code/openvla"

# Print environment info
echo "MUJOCO_GL: $MUJOCO_GL"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Test headless rendering first (optional but recommended)
echo "Testing headless rendering setup..."
python code/scripts/test_libero_headless.py

# Run LIBERO evaluation
echo "Starting LIBERO evaluation..."
python code/openvla/experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint checkpoints/openvla-7b \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 50 \
    --use_wandb False \
    --seed 7

echo "Evaluation complete!"




