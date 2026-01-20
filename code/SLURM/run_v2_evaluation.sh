#!/bin/bash
#SBATCH --job-name=VLA_V2_EVAL
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/v2_eval_%A_%a.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/v2_eval_%A_%a.err
#SBATCH --array=0-9

# ============================================================================
# VLA V2 Evaluation Script
# ============================================================================
# This script trains and tests adversarial patches on all 10 LIBERO Spatial
# tasks using the V2 optimizer with:
#   - Normalized SE(3) distance (scale-invariant)
#   - De-prioritized gripper weights (0.1 instead of 5.0)
#   - Actual trajectory drift computation (norm of sum, not sum of norms)
#   - Recalibrated metrics (drift_threshold=0.05m, sdr_windows=3,5,10)
#
# Usage:
#   sbatch code/SLURM/run_v2_evaluation.sh
#
# Or run specific tasks:
#   sbatch --array=0-4 code/SLURM/run_v2_evaluation.sh  # Tasks 0-4 only
#   sbatch --array=5 code/SLURM/run_v2_evaluation.sh    # Task 5 only
# ============================================================================

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Get task ID from SLURM array ===
TASK_ID=$SLURM_ARRAY_TASK_ID

# === Configuration ===
SUITE="libero_spatial"
QUERIES=${QUERIES:-200}
PATCH_SIZE=${PATCH_SIZE:-32}
LR=${LR:-0.01}
SEED=${SEED:-42}

# V2 Configuration (key changes from V1)
POSITION_WEIGHT=1.0
ROTATION_WEIGHT=0.5    # Reduced from 1.0
GRIPPER_WEIGHT=0.1     # Reduced from 5.0!
USE_NORMALIZED="--use_normalized"  # Scale-invariant distance

# Test configuration (recalibrated)
DRIFT_THRESHOLD=0.05   # 5cm (was 0.2m)
TASK_SCALE=0.1         # (was 0.5m)
SDR_WINDOWS="3,5,10"   # (was 10,25,50)

# Output paths
OUTPUT_DIR="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack"
EXPERIMENT_NAME="v2_eval_${SUITE}_task${TASK_ID}_q${QUERIES}"

# === Job Info ===
echo "=========================================="
echo "VLA V2 Evaluation - Task $TASK_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo "V2 Configuration:"
echo "  Position Weight: $POSITION_WEIGHT"
echo "  Rotation Weight: $ROTATION_WEIGHT"
echo "  Gripper Weight:  $GRIPPER_WEIGHT"
echo "  Use Normalized:  YES"
echo "  Drift Threshold: ${DRIFT_THRESHOLD}m"
echo "  Task Scale:      ${TASK_SCALE}m"
echo "  SDR Windows:     $SDR_WINDOWS"
echo "=========================================="

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Create output directories ===
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/patches"
mkdir -p "$OUTPUT_DIR/results"
mkdir -p "$OUTPUT_DIR/experiments/v2_evaluation"

# ============================================================================
# PHASE 1: TRAIN PATCH
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Training Patch for Task $TASK_ID"
echo "=========================================="

TRAIN_START=$(date +%s)

$PYTHON code/scripts/train_patch.py \
    --suite $SUITE \
    --task_id $TASK_ID \
    --queries $QUERIES \
    --patch_size $PATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --position_weight $POSITION_WEIGHT \
    --rotation_weight $ROTATION_WEIGHT \
    --gripper_weight $GRIPPER_WEIGHT \
    $USE_NORMALIZED \
    --experiment_name "$EXPERIMENT_NAME" \
    --output_dir "$OUTPUT_DIR"

TRAIN_EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed in ${TRAIN_TIME}s"

# === Find the trained patch ===
PATCH_PATH=$(ls -t "$OUTPUT_DIR/patches/${EXPERIMENT_NAME}"*_patch.npy 2>/dev/null | head -1)

if [ -z "$PATCH_PATH" ] || [ ! -f "$PATCH_PATH" ]; then
    echo "ERROR: Could not find trained patch"
    echo "Looking for: $OUTPUT_DIR/patches/${EXPERIMENT_NAME}*_patch.npy"
    ls -la "$OUTPUT_DIR/patches/" | grep "$EXPERIMENT_NAME" || echo "No matching files found"
    exit 1
fi

echo "Found patch: $PATCH_PATH"

# ============================================================================
# PHASE 2: TEST PATCH
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: Testing Patch for Task $TASK_ID"
echo "=========================================="

TEST_START=$(date +%s)

$PYTHON code/scripts/test_patch.py \
    --patch_path "$PATCH_PATH" \
    --suite $SUITE \
    --task_id $TASK_ID \
    --seed $SEED \
    --drift_threshold $DRIFT_THRESHOLD \
    --task_scale $TASK_SCALE \
    --sdr_windows "$SDR_WINDOWS" \
    --output_dir "$OUTPUT_DIR"

TEST_EXIT_CODE=$?
TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Testing failed with exit code $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi

echo "Testing completed in ${TEST_TIME}s"

# ============================================================================
# SUMMARY
# ============================================================================
TOTAL_TIME=$((TRAIN_TIME + TEST_TIME))

echo ""
echo "=========================================="
echo "V2 EVALUATION COMPLETE - Task $TASK_ID"
echo "=========================================="
echo "Training time: ${TRAIN_TIME}s"
echo "Testing time:  ${TEST_TIME}s"
echo "Total time:    ${TOTAL_TIME}s"
echo "Patch path:    $PATCH_PATH"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit 0
