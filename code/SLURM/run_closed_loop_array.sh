#!/bin/bash
#SBATCH --job-name=CL_array
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/cl_array_%A_%a.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/cl_array_%A_%a.err
#SBATCH --array=0-9

# =============================================================================
# CLOSED-LOOP EVALUATION - ARRAY JOB (All 10 tasks in parallel)
# =============================================================================
#
# This SLURM array job runs closed-loop evaluation for all 10 tasks in
# libero_spatial in PARALLEL (one task per array task).
#
# Usage:
#   sbatch run_closed_loop_array.sh <patch_path>
#   sbatch run_closed_loop_array.sh none  # Clean baseline
#
# Example:
#   sbatch run_closed_loop_array.sh /path/to/patch.npy
#
# Note: This submits 10 jobs simultaneously. Make sure your cluster has
# enough resources. Otherwise, use run_closed_loop_eval.sh with TASK_SPEC=all.
# =============================================================================

# === Parameters ===
PATCH_PATH="${1:-none}"
SUITE="${SUITE:-libero_spatial}"
CLEAN_EPISODES="${CLEAN_EPISODES:-50}"
ATTACKED_EPISODES="${ATTACKED_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-300}"

# Task ID comes from SLURM array index
TASK_ID=$SLURM_ARRAY_TASK_ID

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache
export CUDA_VISIBLE_DEVICES=0

# === Paths ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_ROOT="/data1/ma1/Ishaq/ump-vla"
CODE_DIR="${PROJECT_ROOT}/code"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/se3_zoo_attack/closed_loop"

mkdir -p "$OUTPUT_DIR"

# === Job Info ===
echo "=========================================="
echo "CLOSED-LOOP EVALUATION - Array Job"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task ID: $TASK_ID"
echo "Start time: $(date)"
echo "=========================================="

# === Verify Python ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found at $PYTHON"
    exit 1
fi

cd "$PROJECT_ROOT"

# === Determine experiment name ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$PATCH_PATH" = "none" ]; then
    EXP_NAME="cl_array_clean_${SUITE}_task${TASK_ID}_${TIMESTAMP}"
    PATCH_ARG=""
else
    PATCH_BASENAME=$(basename "$PATCH_PATH" .npy)
    EXP_NAME="cl_array_${PATCH_BASENAME}_task${TASK_ID}_${TIMESTAMP}"
    PATCH_ARG="--patch_path $PATCH_PATH"
fi

# === Run Evaluation ===
echo "Running closed-loop evaluation for task $TASK_ID..."

$PYTHON "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
    --suite "$SUITE" \
    --task_id "$TASK_ID" \
    $PATCH_ARG \
    --clean_episodes "$CLEAN_EPISODES" \
    --attacked_episodes "$ATTACKED_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --experiment_name "$EXP_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task $TASK_ID completed successfully"
else
    echo "❌ Task $TASK_ID failed - Exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
