#!/bin/bash
#SBATCH --job-name=UMP_VLA_VERIFY
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma1@alumni.cmu.edu
#SBATCH --output=logs/verify_action_extractor_%j.out
#SBATCH --error=logs/verify_action_extractor_%j.err

# === Set headless rendering for LIBERO ===
export MUJOCO_GL=osmesa

# === Use Python from conda environment (full path) ===
PYTHON="/opt/anaconda3/envs/upa-vla/bin/python3.10"

# === Debug info (optional, useful for troubleshooting) ===
echo "=========================================="
echo "Job started at: $(date)"
echo "=========================================="
echo "Python: $PYTHON"
$PYTHON --version 2>&1 || echo "Python check failed"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "=========================================="

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Run verification script ===
$PYTHON code/scripts/verify_action_extractor.py

# === Job completion ===
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

