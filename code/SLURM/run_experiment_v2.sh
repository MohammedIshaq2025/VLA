#!/bin/bash
#SBATCH --job-name=se3_zoo_v2
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/exp_v2_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/exp_v2_%j.err
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au

# SE(3) ZOO V2: MAXIMIZE DEVIATION Experiment
# 
# This uses the new objective: maximize deviation from clean prediction
# with 10x weight on gripper changes

EXP_NAME="${1:-exp4_maximize_deviation}"
QUERIES="${2:-200}"
GRIPPER_WEIGHT="${3:-10.0}"

export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0

PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
CODE_DIR="/data1/ma1/Ishaq/ump-vla/code"
OUTPUT_DIR="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack"

mkdir -p "${OUTPUT_DIR}/experiments/${EXP_NAME}"

EXP_LOG="${OUTPUT_DIR}/experiments/${EXP_NAME}/experiment_log.txt"

echo "========================================" | tee "$EXP_LOG"
echo "SE(3) ZOO V2: MAXIMIZE DEVIATION" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Experiment Name: $EXP_NAME" | tee -a "$EXP_LOG"
echo "SLURM Job ID: $SLURM_JOB_ID" | tee -a "$EXP_LOG"
echo "Start Time: $(date)" | tee -a "$EXP_LOG"
echo "Queries: $QUERIES" | tee -a "$EXP_LOG"
echo "Gripper Weight: ${GRIPPER_WEIGHT}x" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

nvidia-smi | tee -a "$EXP_LOG"

# Train with V2 optimizer
echo "" | tee -a "$EXP_LOG"
echo "[TRAINING V2] Starting..." | tee -a "$EXP_LOG"
TRAIN_START=$(date +%s)

$PYTHON "${CODE_DIR}/scripts/train_patch_v2.py" \
    --suite libero_spatial \
    --task_id 0 \
    --train_ratio 0.7 \
    --queries $QUERIES \
    --gripper_weight $GRIPPER_WEIGHT \
    --asr_threshold 0.5 \
    --seed 42 \
    --experiment_name "${EXP_NAME}_task0" \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))

echo "[TRAINING V2] Completed in ${TRAIN_TIME}s (exit: $TRAIN_EXIT)" | tee -a "$EXP_LOG"

# Find patch
PATCH_FILE=$(ls -t "${OUTPUT_DIR}/patches/${EXP_NAME}"*_patch.npy 2>/dev/null | head -1)

if [ -z "$PATCH_FILE" ]; then
    echo "[ERROR] No patch file found" | tee -a "$EXP_LOG"
    exit 1
fi

echo "[FOUND] Patch: $PATCH_FILE" | tee -a "$EXP_LOG"

# Test with standard test script
echo "" | tee -a "$EXP_LOG"
echo "[TESTING] Starting..." | tee -a "$EXP_LOG"
TEST_START=$(date +%s)

$PYTHON "${CODE_DIR}/scripts/test_patch.py" \
    --patch_path "$PATCH_FILE" \
    --suite libero_spatial \
    --task_id 0 \
    --train_ratio 0.7 \
    --asr_threshold 0.5 \
    --seed 42 \
    --frames_per_episode 10 \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

TEST_EXIT=$?
TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))

echo "[TESTING] Completed in ${TEST_TIME}s (exit: $TEST_EXIT)" | tee -a "$EXP_LOG"

# Extract results
RUN_ID=$(basename "$PATCH_FILE" | sed 's/_patch.npy//')
RESULTS_FILE="${OUTPUT_DIR}/results/${RUN_ID}_testing.json"

if [ -f "$RESULTS_FILE" ]; then
    echo "" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"
    echo "RESULTS (V2 - MAXIMIZE DEVIATION)" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"
    cat "$RESULTS_FILE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['metrics']
print(f\"ASR: {m['primary']['asr']*100:.1f}%\")
print(f\"Task Failure ASR: {m['primary']['task_failure_asr']*100:.1f}%\")
print(f\"Gripper Flip Rate: {m['success_rates']['gripper_flip_rate']*100:.1f}%\")
print(f\"Avg Patch Effect: {m['patch_effect']['avg_patch_effect']:.4f}\")
print(f\"Avg Gripper Change: {m['patch_effect']['avg_gripper_change']:.4f}\")
" | tee -a "$EXP_LOG"
    echo "========================================" | tee -a "$EXP_LOG"
fi

echo "" | tee -a "$EXP_LOG"
echo "Experiment Complete: $(date)" | tee -a "$EXP_LOG"
echo "Log: $EXP_LOG" | tee -a "$EXP_LOG"


