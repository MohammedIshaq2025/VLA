#!/bin/bash
#SBATCH --job-name=dir2_zoo_exp
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/exp_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/exp_%j.err
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au

# Direction 2: Maximize Deviation Attack Experiment Runner
#
# Usage:
#   sbatch run_experiment.sh --exp_name dir2_baseline --queries 200
#   sbatch run_experiment.sh --exp_name dir2_all_tasks --task_ids "0,1,2,3,4,5,6,7,8,9"
#   sbatch run_experiment.sh --exp_name dir2_high_grip --gripper_weight 10.0
#
# Arguments:
#   --exp_name: Experiment name (required)
#   --suite: libero_spatial, libero_object, libero_goal (default: libero_spatial)
#   --task_ids: Comma-separated task IDs (default: 0)
#   --train_ratio: Training ratio (default: 0.7)
#   --queries: Query budget (default: 200)
#   --mini_batch_size: Mini-batch size for gradient estimation (default: 3)
#   --deviation_threshold: Deviation threshold (default: 0.3)
#   --position_weight: Weight on position deviation (default: 1.0)
#   --rotation_weight: Weight on rotation deviation (default: 1.0)
#   --gripper_weight: Weight on gripper deviation (default: 5.0)
#   --lr: Learning rate (default: 0.01)
#   --seed: Random seed (default: 42)

# Parse arguments
EXP_NAME=""
SUITE="libero_spatial"
TASK_IDS="0"
TRAIN_RATIO="0.7"
QUERIES="200"
MINI_BATCH_SIZE="3"
DEVIATION_THRESHOLD="0.3"
POSITION_WEIGHT="1.0"
ROTATION_WEIGHT="1.0"
GRIPPER_WEIGHT="5.0"
LR="0.01"
SEED="42"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name) EXP_NAME="$2"; shift 2;;
        --suite) SUITE="$2"; shift 2;;
        --task_ids) TASK_IDS="$2"; shift 2;;
        --train_ratio) TRAIN_RATIO="$2"; shift 2;;
        --queries) QUERIES="$2"; shift 2;;
        --mini_batch_size) MINI_BATCH_SIZE="$2"; shift 2;;
        --deviation_threshold) DEVIATION_THRESHOLD="$2"; shift 2;;
        --position_weight) POSITION_WEIGHT="$2"; shift 2;;
        --rotation_weight) ROTATION_WEIGHT="$2"; shift 2;;
        --gripper_weight) GRIPPER_WEIGHT="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo "ERROR: --exp_name is required"
    exit 1
fi

# Environment setup
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0

# Paths
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
CODE_DIR="/data1/ma1/Ishaq/ump-vla/code"
OUTPUT_DIR="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack"

# Create directories
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/patches"
mkdir -p "${OUTPUT_DIR}/results"
mkdir -p "${OUTPUT_DIR}/experiments/${EXP_NAME}"

# Log file for this experiment
EXP_LOG="${OUTPUT_DIR}/experiments/${EXP_NAME}/experiment_log.txt"

echo "========================================" | tee -a "$EXP_LOG"
echo "DIRECTION 2: MAXIMIZE DEVIATION ATTACK" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Goal: Create patches that maximize deviation from clean predictions" | tee -a "$EXP_LOG"
echo "      for closed-loop trajectory attack evaluation" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Experiment Name: $EXP_NAME" | tee -a "$EXP_LOG"
echo "SLURM Job ID: $SLURM_JOB_ID" | tee -a "$EXP_LOG"
echo "Start Time: $(date)" | tee -a "$EXP_LOG"
echo "Suite: $SUITE" | tee -a "$EXP_LOG"
echo "Task IDs: $TASK_IDS" | tee -a "$EXP_LOG"
echo "Train Ratio: $TRAIN_RATIO" | tee -a "$EXP_LOG"
echo "Queries: $QUERIES" | tee -a "$EXP_LOG"
echo "Mini-batch Size: $MINI_BATCH_SIZE" | tee -a "$EXP_LOG"
echo "Deviation Threshold: $DEVIATION_THRESHOLD" | tee -a "$EXP_LOG"
echo "Weights: pos=$POSITION_WEIGHT, rot=$ROTATION_WEIGHT, grip=$GRIPPER_WEIGHT" | tee -a "$EXP_LOG"
echo "Learning Rate: $LR" | tee -a "$EXP_LOG"
echo "Seed: $SEED" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

# Check GPU
nvidia-smi | tee -a "$EXP_LOG"
echo "" | tee -a "$EXP_LOG"

# Check Python
echo "Python: $PYTHON" | tee -a "$EXP_LOG"
$PYTHON --version | tee -a "$EXP_LOG"
echo "" | tee -a "$EXP_LOG"

# Convert task IDs to array
IFS=',' read -ra TASKS <<< "$TASK_IDS"

# Results summary file
SUMMARY_FILE="${OUTPUT_DIR}/experiments/${EXP_NAME}/results_summary.json"
echo "[" > "$SUMMARY_FILE"
FIRST_TASK=true

# Run experiment for each task
for TASK_ID in "${TASKS[@]}"; do
    echo "" | tee -a "$EXP_LOG"
    echo "======================================" | tee -a "$EXP_LOG"
    echo "TASK $TASK_ID" | tee -a "$EXP_LOG"
    echo "======================================" | tee -a "$EXP_LOG"

    TASK_EXP_NAME="${EXP_NAME}_task${TASK_ID}"

    # ====== TRAINING ======
    echo "[TRAINING] Task $TASK_ID starting..." | tee -a "$EXP_LOG"
    TRAIN_START=$(date +%s)

    $PYTHON "${CODE_DIR}/scripts/train_patch.py" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio "$TRAIN_RATIO" \
        --queries "$QUERIES" \
        --mini_batch_size "$MINI_BATCH_SIZE" \
        --deviation_threshold "$DEVIATION_THRESHOLD" \
        --position_weight "$POSITION_WEIGHT" \
        --rotation_weight "$ROTATION_WEIGHT" \
        --gripper_weight "$GRIPPER_WEIGHT" \
        --lr "$LR" \
        --seed "$SEED" \
        --experiment_name "$TASK_EXP_NAME" \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

    TRAIN_EXIT=$?
    TRAIN_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_END - TRAIN_START))

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[ERROR] Training failed for task $TASK_ID with exit code $TRAIN_EXIT" | tee -a "$EXP_LOG"
        continue
    fi

    echo "[TRAINING] Task $TASK_ID completed in ${TRAIN_TIME}s" | tee -a "$EXP_LOG"

    # Find the latest patch file
    PATCH_FILE=$(ls -t "${OUTPUT_DIR}/patches/${TASK_EXP_NAME}"*_patch.npy 2>/dev/null | head -1)

    if [ -z "$PATCH_FILE" ]; then
        echo "[ERROR] No patch file found for task $TASK_ID" | tee -a "$EXP_LOG"
        continue
    fi

    echo "[FOUND] Patch: $PATCH_FILE" | tee -a "$EXP_LOG"

    # ====== TESTING ======
    echo "" | tee -a "$EXP_LOG"
    echo "[TESTING] Task $TASK_ID starting..." | tee -a "$EXP_LOG"
    TEST_START=$(date +%s)

    $PYTHON "${CODE_DIR}/scripts/test_patch.py" \
        --patch_path "$PATCH_FILE" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio "$TRAIN_RATIO" \
        --deviation_threshold "$DEVIATION_THRESHOLD" \
        --seed "$SEED" \
        --frames_per_episode 10 \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$EXP_LOG"

    TEST_EXIT=$?
    TEST_END=$(date +%s)
    TEST_TIME=$((TEST_END - TEST_START))

    if [ $TEST_EXIT -ne 0 ]; then
        echo "[ERROR] Testing failed for task $TASK_ID with exit code $TEST_EXIT" | tee -a "$EXP_LOG"
    else
        echo "[TESTING] Task $TASK_ID completed in ${TEST_TIME}s" | tee -a "$EXP_LOG"
    fi

    # Find the latest testing results
    RUN_ID=$(basename "$PATCH_FILE" | sed 's/_patch.npy//')
    RESULTS_FILE="${OUTPUT_DIR}/results/${RUN_ID}_testing.json"

    if [ -f "$RESULTS_FILE" ]; then
        # Extract Direction 2 key metrics
        AVG_DEV=$(cat "$RESULTS_FILE" | grep -oP '"average":\s*\K[0-9.]+' | head -1)
        DEV_RATE=$(cat "$RESULTS_FILE" | grep -oP '"rate":\s*\K[0-9.]+' | head -1)
        POS_DEV=$(cat "$RESULTS_FILE" | grep -oP '"position":\s*\K[0-9.]+' | head -1)
        CUMUL_POS=$(cat "$RESULTS_FILE" | grep -oP '"total_position":\s*\K[0-9.]+' | head -1)

        echo "" | tee -a "$EXP_LOG"
        echo "[RESULTS] Task $TASK_ID (Direction 2 Metrics):" | tee -a "$EXP_LOG"
        echo "  Avg Deviation: ${AVG_DEV:-N/A}" | tee -a "$EXP_LOG"
        echo "  Deviation Rate: ${DEV_RATE:-N/A}" | tee -a "$EXP_LOG"
        echo "  Position Deviation: ${POS_DEV:-N/A}" | tee -a "$EXP_LOG"
        echo "  Cumulative Position: ${CUMUL_POS:-N/A}m" | tee -a "$EXP_LOG"

        # Add to summary
        if [ "$FIRST_TASK" = false ]; then
            echo "," >> "$SUMMARY_FILE"
        fi
        FIRST_TASK=false

        echo "  {" >> "$SUMMARY_FILE"
        echo "    \"task_id\": $TASK_ID," >> "$SUMMARY_FILE"
        echo "    \"avg_deviation\": ${AVG_DEV:-0}," >> "$SUMMARY_FILE"
        echo "    \"deviation_rate\": ${DEV_RATE:-0}," >> "$SUMMARY_FILE"
        echo "    \"position_deviation\": ${POS_DEV:-0}," >> "$SUMMARY_FILE"
        echo "    \"cumulative_position\": ${CUMUL_POS:-0}," >> "$SUMMARY_FILE"
        echo "    \"train_time\": $TRAIN_TIME," >> "$SUMMARY_FILE"
        echo "    \"test_time\": $TEST_TIME," >> "$SUMMARY_FILE"
        echo "    \"patch_path\": \"$PATCH_FILE\"," >> "$SUMMARY_FILE"
        echo "    \"results_path\": \"$RESULTS_FILE\"" >> "$SUMMARY_FILE"
        echo "  }" >> "$SUMMARY_FILE"
    fi

    echo "" | tee -a "$EXP_LOG"
done

# Close summary JSON
echo "]" >> "$SUMMARY_FILE"

# Final summary
echo "" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "EXPERIMENT COMPLETE" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"
echo "Experiment: $EXP_NAME" | tee -a "$EXP_LOG"
echo "End Time: $(date)" | tee -a "$EXP_LOG"
echo "Summary: $SUMMARY_FILE" | tee -a "$EXP_LOG"
echo "Log: $EXP_LOG" | tee -a "$EXP_LOG"
echo "========================================" | tee -a "$EXP_LOG"

# Print summary
echo "" | tee -a "$EXP_LOG"
echo "Results Summary:" | tee -a "$EXP_LOG"
cat "$SUMMARY_FILE" | tee -a "$EXP_LOG"
