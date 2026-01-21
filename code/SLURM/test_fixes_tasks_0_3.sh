#!/bin/bash
#SBATCH --job-name=test_fixes
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_fixes_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_fixes_%j.err

# =============================================================================
# QUICK TEST: Validate Fixes on Tasks 0-3
# =============================================================================
#
# This script tests the critical bug fixes:
# 1. Correct unnorm_key for LIBERO (libero_spatial_no_noops)
# 2. Correct camera resolution (224x224)
# 3. Proper action denormalization
#
# Tests clean baseline on tasks 0-3 to verify OpenVLA can solve them.
# Expected: >0% success rate on at least some tasks
#
# Usage:
#   sbatch code/SLURM/test_fixes_tasks_0_3.sh
#
# =============================================================================

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache
export CUDA_VISIBLE_DEVICES=0

# === Paths ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_ROOT="/data1/ma1/Ishaq/ump-vla"
CODE_DIR="${PROJECT_ROOT}/code"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/se3_zoo_attack/closed_loop"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/outputs/se3_zoo_attack/logs"

# === Parameters ===
SUITE="libero_spatial"
CLEAN_EPISODES=10  # Reduced for faster testing
MAX_STEPS=300
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================================================================"
echo "TESTING BUG FIXES ON LIBERO TASKS 0-3"
echo "============================================================================================================"
echo "Start time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "============================================================================================================"
echo "Configuration:"
echo "  Suite: $SUITE"
echo "  Tasks: 0, 1, 2, 3"
echo "  Episodes per task: $CLEAN_EPISODES"
echo "  Camera resolution: 224x224 (OpenVLA native)"
echo "  Unnorm key: libero_spatial_no_noops (CRITICAL FIX)"
echo "============================================================================================================"
echo ""

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
fi

# === Verify Python ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found at $PYTHON"
    exit 1
fi

cd "$PROJECT_ROOT"

# === Test Each Task ===
for TASK_ID in 0 1 2 3; do
    echo "============================================================================================================"
    echo "Testing Task $TASK_ID"
    echo "============================================================================================================"

    TASK_START=$(date +%s)

    $PYTHON -u "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --clean_episodes "$CLEAN_EPISODES" \
        --attacked_episodes 0 \
        --max_steps "$MAX_STEPS" \
        --camera_height 224 \
        --camera_width 224 \
        --experiment_name "test_fixes_task${TASK_ID}_${TIMESTAMP}" \
        --output_dir "$OUTPUT_DIR" \
        --seed 42

    EXIT_CODE=$?
    TASK_END=$(date +%s)
    TASK_TIME=$((TASK_END - TASK_START))

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Task $TASK_ID completed in ${TASK_TIME}s"

        # Extract success rate
        RESULT_FILE="${OUTPUT_DIR}/test_fixes_task${TASK_ID}_${TIMESTAMP}_results.json"
        if [ -f "$RESULT_FILE" ]; then
            SUCCESS_RATE=$($PYTHON -c "
import json, sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    rate = data['aggregate_metrics']['mean_clean_success_rate']
    print(f'{rate*100:.1f}%')
except Exception as e:
    print('ERROR')
")
            echo "   Success Rate: $SUCCESS_RATE"
        fi
    else
        echo "❌ Task $TASK_ID FAILED - Exit code: $EXIT_CODE"
    fi
    echo ""
done

echo "============================================================================================================"
echo "TEST COMPLETE"
echo "============================================================================================================"
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "1. Check results in: $OUTPUT_DIR/test_fixes_*.json"
echo "2. If success rate > 0% for any task: BUG FIXES WORKED!"
echo "3. If all tasks still 0%: Additional debugging needed"
echo "============================================================================================================"

exit 0
