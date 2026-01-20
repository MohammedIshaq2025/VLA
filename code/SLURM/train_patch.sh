#!/bin/bash
#SBATCH --job-name=SE3_ZOO_TRAIN
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/train_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/train_%j.err

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path (bypass conda activate) ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Job Info ===
echo "=========================================="
echo "SE(3) ZOO Adversarial Patch Training"
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

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Default parameters (can be overridden via environment variables) ===
SUITE=${SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
TRAIN_RATIO=${TRAIN_RATIO:-0.3}
QUERIES=${QUERIES:-200}
PATCH_SIZE=${PATCH_SIZE:-32}
LR=${LR:-0.01}
PERTURBATION_SCALE=${PERTURBATION_SCALE:-0.1}
ASR_THRESHOLD=${ASR_THRESHOLD:-0.5}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-95.0}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-20}
TARGET_TYPE=${TARGET_TYPE:-generic}
SEED=${SEED:-42}

echo ""
echo "Training Parameters:"
echo "  Suite: $SUITE"
echo "  Task ID: $TASK_ID"
echo "  Train Ratio: $TRAIN_RATIO"
echo "  Queries: $QUERIES"
echo "  Patch Size: $PATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Perturbation Scale: $PERTURBATION_SCALE"
echo "  ASR Threshold: $ASR_THRESHOLD"
echo "  Early Stop: ${EARLY_STOP_THRESHOLD}% for ${EARLY_STOP_PATIENCE} steps"
echo "  Target Type: $TARGET_TYPE"
echo "  Seed: $SEED"
echo "=========================================="
echo ""

# === Run Training ===
$PYTHON code/scripts/train_patch.py \
    --suite $SUITE \
    --task_id $TASK_ID \
    --train_ratio $TRAIN_RATIO \
    --queries $QUERIES \
    --patch_size $PATCH_SIZE \
    --lr $LR \
    --perturbation_scale $PERTURBATION_SCALE \
    --asr_threshold $ASR_THRESHOLD \
    --early_stop_threshold $EARLY_STOP_THRESHOLD \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --target_type $TARGET_TYPE \
    --seed $SEED \
    --output_dir /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ TRAINING FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

