#!/bin/bash
#SBATCH --job-name=K15_REMAIN
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --output=/data1/ma1/Ishaq/VLA_Frequency_Attack/slurm_jobs/logs/k15_remain_%j.out
#SBATCH --error=/data1/ma1/Ishaq/VLA_Frequency_Attack/slurm_jobs/logs/k15_remain_%j.err

# ============================================================================
# CONTINUATION: kappa=15, 150 iterations - Remaining Tasks 5-9
# ============================================================================

set -e

export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache
export PYTHONUNBUFFERED=1
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

PROJECT_DIR="/data1/ma1/Ishaq/VLA_Frequency_Attack"
RESULTS_DIR="$PROJECT_DIR/results/ablation_k15_iter150"
SCRIPT="$PROJECT_DIR/scripts/evaluate_closed_loop_k15_iter150.py"

NUM_EPISODES=10
MAX_STEPS=220

echo "========================================================================"
echo "CONTINUATION: kappa=15, 150 iterations - Remaining Tasks"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""
echo "Task 5: low, high only (baseline already exists)"
echo "Tasks 6-9: baseline, low, high"
echo "========================================================================"

cd "$PROJECT_DIR" || exit 1

FAILED=0

# Task 5: Only low and high (baseline exists)
echo ""
echo "======================================================================="
echo "TASK 5 (attacks only - baseline exists)"
echo "======================================================================="

TASK_DIR="$RESULTS_DIR/task_5"
mkdir -p "$TASK_DIR"

for attack in low high; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Task 5, Attack: $attack"
    echo "------------------------------------------------------------------------"
    echo "Start: $(date)"

    set +e
    $PYTHON "$SCRIPT"         --attack "$attack"         --task_id 5         --num_episodes $NUM_EPISODES         --max_steps $MAX_STEPS         --output "$TASK_DIR"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED (exit code: $EXIT_CODE)"
        FAILED=1
    fi
    echo "End: $(date)"
done

# Tasks 6-9: Full suite (baseline + low + high)
for task_idx in 6 7 8 9; do
    echo ""
    echo "======================================================================="
    echo "TASK $task_idx (full suite)"
    echo "======================================================================="

    TASK_DIR="$RESULTS_DIR/task_$task_idx"
    mkdir -p "$TASK_DIR"

    for attack in baseline low high; do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Task $task_idx, Attack: $attack"
        echo "------------------------------------------------------------------------"
        echo "Start: $(date)"

        set +e
        $PYTHON "$SCRIPT"             --attack "$attack"             --task_id "$task_idx"             --num_episodes $NUM_EPISODES             --max_steps $MAX_STEPS             --output "$TASK_DIR"
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Status: SUCCESS"
        else
            echo "Status: FAILED (exit code: $EXIT_CODE)"
            FAILED=1
        fi
        echo "End: $(date)"
    done
done

echo ""
echo "========================================================================"
echo "REMAINING TASKS COMPLETE"
echo "========================================================================"
echo "End: $(date)"

exit $FAILED

