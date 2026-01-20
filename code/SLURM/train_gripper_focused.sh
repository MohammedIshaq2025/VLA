#!/bin/bash
#SBATCH --job-name=grip_train
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/grip_train_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/grip_train_%j.err

# =============================================================================
# PRIORITY 3: GRIPPER-FOCUSED ATTACK TRAINING
# =============================================================================
#
# This script trains adversarial patches with HEAVY emphasis on gripper
# manipulation to address the critical 0% gripper flip rate problem.
#
# Key Configuration:
#   w_pos = 1.0   (position)
#   w_rot = 0.5   (rotation)
#   w_grip = 5.0  (gripper) ← 50× INCREASE from default 0.1
#
# Hypothesis: Increasing gripper weight will increase gripper flip rate
# from 0% to 30-50%, which should dramatically increase task failure rate
# in closed-loop evaluation.
#
# Usage:
#   # Single task with gripper weight 5.0
#   sbatch train_gripper_focused.sh <task_id> 5.0
#
#   # All 10 tasks
#   for i in {0..9}; do sbatch train_gripper_focused.sh $i 5.0; done
#
# Example:
#   sbatch train_gripper_focused.sh 0 5.0
#   sbatch train_gripper_focused.sh 0 10.0  # Even higher gripper weight
#
# =============================================================================

# === Parameters ===
TASK_ID="${1:-0}"
GRIPPER_WEIGHT="${2:-5.0}"
SUITE="${SUITE:-libero_spatial}"
QUERIES="${QUERIES:-200}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
ROT_WEIGHT="${ROT_WEIGHT:-0.5}"

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache
export CUDA_VISIBLE_DEVICES=0

# === Paths ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_ROOT="/data1/ma1/Ishaq/ump-vla"
CODE_DIR="${PROJECT_ROOT}/code"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/se3_zoo_attack"

mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/patches"
mkdir -p "${OUTPUT_DIR}/results"

# === Experiment Name ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="gripper_focused_w${GRIPPER_WEIGHT}_task${TASK_ID}_${TIMESTAMP}"
EXP_LOG="${OUTPUT_DIR}/logs/${EXP_NAME}.log"

# === Job Info ===
echo "========================================" | tee "$EXP_LOG"
echo "GRIPPER-FOCUSED ATTACK TRAINING" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$EXP_LOG"
echo "Start time: $(date)" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Configuration:" | tee -a "$EXP_LOG"
echo "  Suite: $SUITE" | tee -a "$EXP_LOG"
echo "  Task ID: $TASK_ID" | tee -a "$EXP_LOG"
echo "  Queries: $QUERIES" | tee -a "$EXP_LOG"
echo "  w_pos: $POS_WEIGHT" | tee -a "$EXP_LOG"
echo "  w_rot: $ROT_WEIGHT" | tee -a "$EXP_LOG"
echo "  w_grip: $GRIPPER_WEIGHT ← HIGH WEIGHT" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | tee -a "$EXP_LOG"
fi

# === Verify Python ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found at $PYTHON" | tee -a "$EXP_LOG"
    exit 1
fi

cd "$PROJECT_ROOT"

# === Training ===
echo "" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "TRAINING WITH GRIPPER FOCUS" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

TRAIN_START=$(date +%s)

$PYTHON "${CODE_DIR}/scripts/train_patch.py" \
    --suite "$SUITE" \
    --task_id "$TASK_ID" \
    --train_ratio 0.7 \
    --queries "$QUERIES" \
    --patch_size 32 \
    --lr 0.01 \
    --position_weight "$POS_WEIGHT" \
    --rotation_weight "$ROT_WEIGHT" \
    --gripper_weight "$GRIPPER_WEIGHT" \
    --deviation_threshold 0.3 \
    --seed 42 \
    --experiment_name "$EXP_NAME" \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

echo "" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ TRAINING COMPLETED in ${TRAIN_TIME}s" | tee -a "$EXP_LOG"
else
    echo "❌ TRAINING FAILED - Exit code: $TRAIN_EXIT" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"
    exit $TRAIN_EXIT
fi
echo "========================================" | tee -a "$EXP_LOG"

# === Find trained patch ===
PATCH_FILE=$(ls -t "${OUTPUT_DIR}/patches/${EXP_NAME}"*_patch.npy 2>/dev/null | head -1)

if [ -z "$PATCH_FILE" ]; then
    echo "❌ ERROR: No patch file found" | tee -a "$EXP_LOG"
    exit 1
fi

echo "[FOUND] Patch: $PATCH_FILE" | tee -a "$EXP_LOG"

# === Testing ===
echo "" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "TESTING GRIPPER-FOCUSED PATCH" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

TEST_START=$(date +%s)

$PYTHON "${CODE_DIR}/scripts/test_patch.py" \
    --patch_path "$PATCH_FILE" \
    --suite "$SUITE" \
    --task_id "$TASK_ID" \
    --train_ratio 0.7 \
    --deviation_threshold 0.3 \
    --seed 42 \
    --frames_per_episode 10 \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

TEST_EXIT=$?
TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))

echo "" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
if [ $TEST_EXIT -eq 0 ]; then
    echo "✅ TESTING COMPLETED in ${TEST_TIME}s" | tee -a "$EXP_LOG"
else
    echo "❌ TESTING FAILED - Exit code: $TEST_EXIT" | tee -a "$EXP_LOG"
fi
echo "========================================" | tee -a "$EXP_LOG"

# === Extract results ===
RUN_ID=$(basename "$PATCH_FILE" | sed 's/_patch.npy//')
RESULTS_FILE="${OUTPUT_DIR}/results/${RUN_ID}_testing.json"

if [ -f "$RESULTS_FILE" ]; then
    echo "" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"
    echo "KEY METRICS (GRIPPER-FOCUSED)" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"

    # Extract key metrics using Python
    $PYTHON -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)

    m = data.get('metrics', {})

    # Extract key metrics
    dev = m.get('deviation', {})
    comp = m.get('components', {})
    traj = m.get('trajectory', {})

    print(f\"Average Deviation: {dev.get('average', 0):.4f}\")
    print(f\"Deviation Rate: {dev.get('rate', 0)*100:.1f}%\")
    print()
    print(f\"Position Deviation: {comp.get('position', 0):.4f}\")
    print(f\"Rotation Deviation: {comp.get('rotation', 0):.4f}\")
    print(f\"Gripper Deviation: {comp.get('gripper', 0):.4f}\")
    print()

    # Check if we have actual drift metrics
    if 'actual_drift' in traj:
        actual = traj['actual_drift']
        print(f\"Actual Drift: {actual.get('mean', 0):.4f}m\")
        print(f\"Drift Consistency: {actual.get('consistency_mean', 0):.2f}\")
        print()

        # Check gripper flip success
        if comp.get('gripper', 0) >= 1.0:
            print('✅ GRIPPER FLIP ACHIEVED! (change >= 1.0)')
        elif comp.get('gripper', 0) >= 0.7:
            print('⚠️  Close to gripper flip (change >= 0.7)')
        else:
            print(f'❌ Gripper flip not achieved (change = {comp.get(\"gripper\", 0):.2f}, need >= 1.0)')

except Exception as e:
    print(f'Error parsing results: {e}', file=sys.stderr)
" | tee -a "$EXP_LOG"

    echo "========================================" | tee -a "$EXP_LOG"
fi

echo "" | tee -a "$EXP_LOG"
echo "Experiment Complete: $(date)" | tee -a "$EXP_LOG"
echo "Patch: $PATCH_FILE" | tee -a "$EXP_LOG"
echo "Results: $RESULTS_FILE" | tee -a "$EXP_LOG"
echo "Log: $EXP_LOG" | tee -a "$EXP_LOG"

exit 0
