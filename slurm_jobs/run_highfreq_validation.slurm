#!/bin/bash
#SBATCH --job-name=HIGHFREQ_VAL
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --output=results/phase5/highfreq_validation_%j.out
#SBATCH --error=results/phase5/highfreq_validation_%j.err

################################################################################
# HIGH-FREQUENCY ATTACK VALIDATION TEST
# Purpose: Validate the bug fix for high-frequency mask creation
# Expected: BFR > 0%, Purity > 0% (should NOT be exactly zero)
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "HIGH-FREQUENCY ATTACK VALIDATION TEST"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""

# Environment setup
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_DIR="/data1/ma1/Ishaq/VLA_Frequency_Attack"
RESULTS_DIR="$PROJECT_DIR/results/phase5/highfreq_validation_${SLURM_JOB_ID}"

# Navigate to project directory
cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR"

# Configuration
NUM_EPISODES=3
MAX_STEPS=50
TASK_ID=0  # pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate

echo "Configuration:"
echo "  Episodes: $NUM_EPISODES"
echo "  Max steps: $MAX_STEPS"
echo "  Task ID: $TASK_ID"
echo "  Attack: high (freq_ratio = -0.25)"
echo ""

# Run high-frequency attack validation
echo "========================================================================"
echo "RUNNING HIGH-FREQUENCY ATTACK TEST"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

timeout 30m $PYTHON evaluate_closed_loop.py \
    --attack high \
    --task_id "$TASK_ID" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/high_output.log"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================================"
echo "VALIDATION RESULTS"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "Duration: ${DURATION}s"
echo ""

# Check if results file exists
RESULTS_FILE="$RESULTS_DIR/high_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "Results file found: $RESULTS_FILE"
    echo ""

    # Extract key metrics using Python
    echo "Extracting metrics..."
    $PYTHON -c "
import json
import sys

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

avg_bfr = data.get('avg_bin_flip_rate', 0) * 100
avg_purity = data.get('avg_frequency_purity', 0) * 100
episodes = data.get('episode_results', [])

print('CRITICAL METRICS:')
print(f'  Average BFR: {avg_bfr:.1f}%')
print(f'  Average Purity: {avg_purity:.1f}%')
print(f'  Episodes: {len(episodes)}')
print('')

# Check if bug is fixed
if avg_bfr == 0.0 and avg_purity == 0.0:
    print('STATUS: FAILED - Bug still present (all zeros)')
    print('  Problem: Mask is still all zeros')
    print('  Action: Check frequency_attack.py for correct fix')
    sys.exit(1)
elif avg_bfr > 0.0 and avg_purity > 0.0:
    print('STATUS: SUCCESS - Bug is FIXED!')
    print(f'  High-frequency attack produces non-zero perturbation')
    print(f'  BFR: {avg_bfr:.1f}% (expected: 10-35%)')
    print(f'  Purity: {avg_purity:.1f}% (expected: 70-95%)')
else:
    print('STATUS: PARTIAL - Unexpected results')
    print(f'  BFR={avg_bfr:.1f}%, Purity={avg_purity:.1f}%')
    print('  Action: Review results manually')

print('')
print('Episode breakdown:')
for i, ep in enumerate(episodes, 1):
    bfr = ep.get('avg_bin_flip_rate', 0) * 100
    purity = ep.get('avg_purity', 0) * 100
    print(f'  Episode {i}: BFR={bfr:.1f}%, Purity={purity:.1f}%')
"

    VALIDATION_EXIT=$?
    echo ""

    if [ $VALIDATION_EXIT -eq 0 ]; then
        echo "========================================================================"
        echo "VALIDATION PASSED - HIGH-FREQUENCY ATTACK IS WORKING"
        echo "========================================================================"
        echo "The bug fix is confirmed. You can now proceed to full Phase 5 test."
    else
        echo "========================================================================"
        echo "VALIDATION FAILED - BUG STILL PRESENT"
        echo "========================================================================"
        echo "Please review the fix in experiments/frequency_attack.py"
    fi
else
    echo "ERROR: Results file not found!"
    echo "Expected: $RESULTS_FILE"
    echo ""
    echo "Check error log for details:"
    echo "  $RESULTS_DIR/high_output.log"
    exit 1
fi

echo ""
echo "Completed: $(date)"
echo "Total duration: ${DURATION}s"
echo "========================================================================"
