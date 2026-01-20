#!/bin/bash
#SBATCH --job-name=closed_loop
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/closed_loop_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/closed_loop_%j.err

# =============================================================================
# CLOSED-LOOP LIBERO EVALUATION - THE CRITICAL VALIDATION
# =============================================================================
#
# This script runs TRUE closed-loop evaluation by executing the robot policy
# in actual LIBERO environments and measuring TASK SUCCESS RATES.
#
# This is Priority #1 - the make-or-break experiment for the ECCV paper.
#
# Usage:
#   # Single task with patch
#   sbatch run_closed_loop_eval.sh <patch_path> <task_id>
#
#   # Single task clean baseline (no patch)
#   sbatch run_closed_loop_eval.sh none <task_id>
#
#   # All 10 tasks with patch
#   sbatch run_closed_loop_eval.sh <patch_path> all
#
# Example:
#   sbatch run_closed_loop_eval.sh /path/to/patch.npy 0
#   sbatch run_closed_loop_eval.sh none 0  # Clean baseline
#   sbatch run_closed_loop_eval.sh /path/to/patch.npy all  # All tasks
#
# =============================================================================

# === Parameters ===
PATCH_PATH="${1:-none}"
TASK_SPEC="${2:-0}"
SUITE="${SUITE:-libero_spatial}"
CLEAN_EPISODES="${CLEAN_EPISODES:-50}"
ATTACKED_EPISODES="${ATTACKED_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-300}"

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache
export CUDA_VISIBLE_DEVICES=0

# === Paths ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_ROOT="/data1/ma1/Ishaq/ump-vla"
CODE_DIR="${PROJECT_ROOT}/code"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/se3_zoo_attack/closed_loop"

# === Create output directory ===
mkdir -p "$OUTPUT_DIR"
mkdir -p "${PROJECT_ROOT}/outputs/se3_zoo_attack/logs"

# === Job Info ===
echo "=========================================="
echo "CLOSED-LOOP LIBERO EVALUATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "=========================================="
echo "Configuration:"
echo "  Suite: $SUITE"
echo "  Task(s): $TASK_SPEC"
echo "  Patch: $PATCH_PATH"
echo "  Clean episodes: $CLEAN_EPISODES"
echo "  Attacked episodes: $ATTACKED_EPISODES"
echo "  Max steps: $MAX_STEPS"
echo "=========================================="

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    echo "=========================================="
fi

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Change to project root ===
cd "$PROJECT_ROOT"

# === Determine experiment name ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$PATCH_PATH" = "none" ]; then
    EXP_NAME="closed_loop_clean_${SUITE}_${TIMESTAMP}"
    PATCH_ARG=""
else
    # Extract patch basename
    PATCH_BASENAME=$(basename "$PATCH_PATH" .npy)
    EXP_NAME="closed_loop_${PATCH_BASENAME}_${TIMESTAMP}"
    PATCH_ARG="--patch_path $PATCH_PATH"
fi

# === Build task argument ===
if [ "$TASK_SPEC" = "all" ]; then
    # All 10 tasks in suite
    TASK_ARG="--task_ids 0,1,2,3,4,5,6,7,8,9"
elif [[ "$TASK_SPEC" =~ ^[0-9,]+$ ]]; then
    # Comma-separated task IDs
    TASK_ARG="--task_ids $TASK_SPEC"
else
    # Single task ID
    TASK_ARG="--task_id $TASK_SPEC"
fi

# === Run Closed-Loop Evaluation ===
echo ""
echo "=========================================="
echo "RUNNING CLOSED-LOOP EVALUATION"
echo "=========================================="
echo "Experiment: $EXP_NAME"
echo "Command:"
echo "$PYTHON ${CODE_DIR}/scripts/evaluate_closed_loop.py \\"
echo "    --suite $SUITE \\"
echo "    $TASK_ARG \\"
echo "    $PATCH_ARG \\"
echo "    --clean_episodes $CLEAN_EPISODES \\"
echo "    --attacked_episodes $ATTACKED_EPISODES \\"
echo "    --max_steps $MAX_STEPS \\"
echo "    --experiment_name $EXP_NAME \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --seed 42"
echo "=========================================="
echo ""

EVAL_START=$(date +%s)

$PYTHON "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
    --suite "$SUITE" \
    $TASK_ARG \
    $PATCH_ARG \
    --clean_episodes "$CLEAN_EPISODES" \
    --attacked_episodes "$ATTACKED_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --experiment_name "$EXP_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

EXIT_CODE=$?
EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ CLOSED-LOOP EVALUATION COMPLETED SUCCESSFULLY"
else
    echo "❌ CLOSED-LOOP EVALUATION FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "Evaluation time: ${EVAL_TIME}s ($((EVAL_TIME / 60)) minutes)"
echo "Results saved to: ${OUTPUT_DIR}/${EXP_NAME}_results.json"
echo "End time: $(date)"
echo "=========================================="

# === Extract and display key results ===
if [ $EXIT_CODE -eq 0 ]; then
    RESULTS_FILE="${OUTPUT_DIR}/${EXP_NAME}_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "=========================================="
        echo "KEY RESULTS SUMMARY"
        echo "=========================================="

        # Use Python to extract metrics
        $PYTHON -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)

    agg = data.get('aggregate_metrics', {})

    print(f\"Number of tasks evaluated: {len(data.get('per_task_results', []))}\")
    print()

    if 'mean_true_asr' in agg:
        print(f\"Mean Clean Success Rate:    {agg['mean_clean_success_rate']*100:.1f}% ± {agg['std_clean_success_rate']*100:.1f}%\")
        print(f\"Mean Attacked Success Rate: {agg['mean_attacked_success_rate']*100:.1f}% ± {agg['std_attacked_success_rate']*100:.1f}%\")
        print(f\"Mean TRUE ASR:              {agg['mean_true_asr']*100:.1f}% ± {agg['std_true_asr']*100:.1f}%\")
        print()

        # Verdict
        mean_asr = agg['mean_true_asr']
        if mean_asr >= 0.60:
            print('VERDICT: ✓✓✓ STRONG ATTACK (ECCV-worthy!)')
        elif mean_asr >= 0.40:
            print('VERDICT: ✓✓ MODERATE ATTACK (Publishable)')
        elif mean_asr >= 0.20:
            print('VERDICT: ✓ WEAK ATTACK (Needs improvement)')
        else:
            print('VERDICT: ✗ INEFFECTIVE ATTACK (Major improvements needed)')
    else:
        print(f\"Clean Baseline Success Rate: {agg['mean_clean_success_rate']*100:.1f}% ± {agg['std_clean_success_rate']*100:.1f}%\")
        print()
        print('VERDICT: Clean baseline established')

except Exception as e:
    print(f'Error parsing results: {e}', file=sys.stderr)
    sys.exit(1)
"

        echo "=========================================="
    fi
fi

exit $EXIT_CODE
