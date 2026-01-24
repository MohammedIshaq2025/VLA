#!/bin/bash
#SBATCH --job-name=ATTACK_T4_9
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --output=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/attack_test_t4_9_%j.out
#SBATCH --error=/data1/ma1/Ishaq/VLA_Frequency_Attack/results/attack_test_t4_9_%j.err

# ============================================================================
# Comprehensive Attack Test - Tasks 4-9
# 10 episodes each, baseline + low + high frequency attacks
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
TASKS=(4 5 6 7 8 9)
ATTACKS=(baseline low high)

echo "========================================================================"
echo "Comprehensive Attack Test - Tasks 4-9"
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

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "TASK $TASK"
    echo "========================================================================"

    mkdir -p "$RESULTS_DIR/task_$TASK"

    for ATTACK in "${ATTACKS[@]}"; do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Task $TASK, Attack: $ATTACK"
        echo "------------------------------------------------------------------------"
        echo "Start: $(date)"

        OUTPUT_FILE="$RESULTS_DIR/task_$TASK/${ATTACK}_results.json"

        if [ "$ATTACK" == "baseline" ]; then
            ATTACK_ARG="--no-attack"
        elif [ "$ATTACK" == "low" ]; then
            ATTACK_ARG="--attack --freq-constraint low --freq-ratio 0.25"
        else
            ATTACK_ARG="--attack --freq-constraint high --freq-ratio 0.25"
        fi

        if $PYTHON "$PROJECT_DIR/evaluate_closed_loop.py" \
            --task-idx $TASK \
            --num-episodes $NUM_EPISODES \
            --max-steps $MAX_STEPS \
            --seed 7 \
            --center-crop \
            $ATTACK_ARG \
            --output "$OUTPUT_FILE"; then
            echo "Status: SUCCESS"
        else
            echo "Status: FAILED"
            FAILED=$((FAILED + 1))
        fi

        echo "End: $(date)"
        echo "------------------------------------------------------------------------"
    done
done

echo ""
echo "========================================================================"
echo "ALL TASKS COMPLETE"
echo "========================================================================"
echo "End: $(date)"
echo "Failed conditions: $FAILED"
echo "========================================================================"
