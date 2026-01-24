#!/bin/bash
#SBATCH --job-name=ATTACK_TEST
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --output=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/attack_test_%j.out
#SBATCH --error=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/attack_test_%j.err

# ============================================================================
# Comprehensive Attack Test
# Tasks 0-3, 10 episodes each, baseline + low + high frequency attacks
# ============================================================================

set -e

export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache
export PYTHONUNBUFFERED=1
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

PROJECT_DIR="/data1/ma1/Ishaq/VLA_Frequency_Attack"
RESULTS_DIR="$PROJECT_DIR/results/attack_test"

NUM_EPISODES=10
MAX_STEPS=220
TASKS=(0 1 2 3)
ATTACKS=(baseline low high)

echo "========================================================================"
echo "Comprehensive Attack Test"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""
echo "Configuration:"
echo "  - Tasks: ${TASKS[@]}"
echo "  - Attacks: ${ATTACKS[@]}"
echo "  - Episodes per condition: $NUM_EPISODES"
echo "  - Max steps: $MAX_STEPS"
echo "  - Total conditions: $((${#TASKS[@]} * ${#ATTACKS[@]}))"
echo "========================================================================"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$RESULTS_DIR"

FAILED=0

for task_idx in "${TASKS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "TASK $task_idx"
    echo "========================================================================"

    TASK_DIR="$RESULTS_DIR/task_$task_idx"
    mkdir -p "$TASK_DIR"

    for attack in "${ATTACKS[@]}"; do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Task $task_idx, Attack: $attack"
        echo "------------------------------------------------------------------------"
        echo "Start: $(date)"

        set +e
        $PYTHON evaluate_closed_loop.py \
            --attack "$attack" \
            --task_id "$task_idx" \
            --num_episodes "$NUM_EPISODES" \
            --max_steps "$MAX_STEPS" \
            --output "$TASK_DIR"
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Status: SUCCESS"
        else
            echo "Status: FAILED (exit code: $EXIT_CODE)"
            FAILED=1
        fi
        echo "End: $(date)"
        echo "------------------------------------------------------------------------"
    done
done

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "========================================================================"
echo "End: $(date)"
echo ""

# Print summary table
echo "RESULTS SUMMARY:"
echo "----------------"
printf "%-8s %-12s %-12s %-12s\n" "Task" "Baseline" "Low-Freq" "High-Freq"
printf "%-8s %-12s %-12s %-12s\n" "----" "--------" "--------" "---------"

for task_idx in "${TASKS[@]}"; do
    BASELINE_FILE="$RESULTS_DIR/task_$task_idx/baseline_results.json"
    LOW_FILE="$RESULTS_DIR/task_$task_idx/low_results.json"
    HIGH_FILE="$RESULTS_DIR/task_$task_idx/high_results.json"

    BASELINE_SR="N/A"
    LOW_SR="N/A"
    HIGH_SR="N/A"

    if [ -f "$BASELINE_FILE" ]; then
        BASELINE_SR=$($PYTHON -c "import json; print(f\"{json.load(open('$BASELINE_FILE'))['success_rate']*100:.0f}%\")")
    fi
    if [ -f "$LOW_FILE" ]; then
        LOW_SR=$($PYTHON -c "import json; r=json.load(open('$LOW_FILE')); print(f\"{r['success_rate']*100:.0f}% ({r['avg_bin_flip_rate']*100:.0f}%BFR)\")")
    fi
    if [ -f "$HIGH_FILE" ]; then
        HIGH_SR=$($PYTHON -c "import json; r=json.load(open('$HIGH_FILE')); print(f\"{r['success_rate']*100:.0f}% ({r['avg_bin_flip_rate']*100:.0f}%BFR)\")")
    fi

    printf "%-8s %-12s %-12s %-12s\n" "$task_idx" "$BASELINE_SR" "$LOW_SR" "$HIGH_SR"
done

echo ""
echo "========================================================================"

exit $FAILED
