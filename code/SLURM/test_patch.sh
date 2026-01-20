#!/bin/bash
#SBATCH --job-name=SE3_ZOO_TEST
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_%j.err

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path (bypass conda activate) ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Job Info ===
echo "=========================================="
echo "SE(3) ZOO Adversarial Patch Testing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Check patch path argument ===
if [ -z "$1" ]; then
    echo "❌ ERROR: Patch path not provided"
    echo "Usage: sbatch test_patch.sh /path/to/patch.npy [suite] [task_id]"
    exit 1
fi

PATCH_PATH=$1
SUITE=${2:-libero_spatial}
TASK_ID=${3:-0}
NUM_EPISODES=${NUM_EPISODES:-35}
FRAMES_PER_EPISODE=${FRAMES_PER_EPISODE:-10}
TARGET_TYPE=${TARGET_TYPE:-generic}

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

echo ""
echo "Testing Parameters:"
echo "  Patch Path: $PATCH_PATH"
echo "  Suite: $SUITE"
echo "  Task ID: $TASK_ID"
echo "  Num Episodes: $NUM_EPISODES"
echo "  Frames per Episode: $FRAMES_PER_EPISODE"
echo "  Target Type: $TARGET_TYPE"
echo "=========================================="
echo ""

# === Run Testing ===
$PYTHON code/scripts/test_patch.py \
    --patch_path $PATCH_PATH \
    --suite $SUITE \
    --task_id $TASK_ID \
    --num_episodes $NUM_EPISODES \
    --frames_per_episode $FRAMES_PER_EPISODE \
    --target_type $TARGET_TYPE \
    --output_dir /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTING COMPLETED SUCCESSFULLY"
else
    echo "❌ TESTING FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE



