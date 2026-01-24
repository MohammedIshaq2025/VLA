#!/bin/bash
#SBATCH --job-name=BASELINE_3T
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=00:59:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --output=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/baseline_verify_3tasks_%j.out
#SBATCH --error=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/baseline_verify_3tasks_%j.err

# ============================================================================
# Baseline Verification: 3 Tasks x 5 Episodes
# ============================================================================
# Purpose: Verify baseline is working correctly across multiple tasks
# Expected: ~75-80% success rate (matching baseline_evaluation.py)
# ============================================================================

set -e

export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache
export PYTHONUNBUFFERED=1
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

PROJECT_DIR="/data1/ma1/Ishaq/VLA_Frequency_Attack"
RESULTS_DIR="$PROJECT_DIR/results/baseline_verify_3tasks"

NUM_EPISODES=5
MAX_STEPS=220
TASKS=(0 1 2)

echo "========================================================================"
echo "Baseline Verification: 3 Tasks"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""
echo "Configuration:"
echo "  - Tasks: ${TASKS[@]}"
echo "  - Episodes per task: $NUM_EPISODES"
echo "  - Max steps: $MAX_STEPS"
echo "  - Timeout per task: 3 minutes"
echo "========================================================================"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$RESULTS_DIR"

TOTAL_SUCCESS=0
TOTAL_EPISODES=0

for task_idx in "${TASKS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Task $task_idx"
    echo "------------------------------------------------------------------------"

    TASK_OUTPUT="$RESULTS_DIR/task_${task_idx}"
    mkdir -p "$TASK_OUTPUT"

    # Run with 3 minute (180s) timeout
    set +e
    timeout 180 $PYTHON evaluate_closed_loop.py \
        --attack baseline \
        --task_id "$task_idx" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --output "$TASK_OUTPUT"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT: Task $task_idx exceeded 3 minutes"
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Task $task_idx failed with exit code $EXIT_CODE"
    else
        # Extract success rate from results
        RESULT_FILE="$TASK_OUTPUT/baseline_results.json"
        if [ -f "$RESULT_FILE" ]; then
            SUCCESS_RATE=$($PYTHON -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"{r['success_rate']*100:.0f}%\")")
            SUCCESSES=$($PYTHON -c "import json; r=json.load(open('$RESULT_FILE')); print(sum(1 for e in r['episode_results'] if e['success']))")
            echo "Task $task_idx: $SUCCESS_RATE ($SUCCESSES/$NUM_EPISODES)"
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + SUCCESSES))
            TOTAL_EPISODES=$((TOTAL_EPISODES + NUM_EPISODES))
        fi
    fi
done

echo ""
echo "========================================================================"
echo "VERIFICATION COMPLETE"
echo "========================================================================"
echo "End: $(date)"
echo ""

if [ $TOTAL_EPISODES -gt 0 ]; then
    OVERALL_RATE=$($PYTHON -c "print(f'{$TOTAL_SUCCESS/$TOTAL_EPISODES*100:.1f}%')")
    echo "Overall Success Rate: $OVERALL_RATE ($TOTAL_SUCCESS/$TOTAL_EPISODES)"
    echo ""
    echo "Expected: ~75-80%"
    echo ""

    if [ $TOTAL_SUCCESS -ge $((TOTAL_EPISODES * 60 / 100)) ]; then
        echo "PASS: Baseline is working correctly (>60%)"
    else
        echo "WARNING: Success rate lower than expected"
    fi
else
    echo "ERROR: No episodes completed"
fi

echo "========================================================================"
