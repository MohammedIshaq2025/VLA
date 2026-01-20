#!/bin/bash
#SBATCH --job-name=dir2_full_eval
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/full_eval_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/full_eval_%j.err
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au

# =============================================================================
# Direction 2: Full Evaluation Across All LIBERO-Spatial Tasks
# =============================================================================
#
# This script runs comprehensive experiments:
# - All 10 LIBERO-Spatial tasks (task_id 0-9)
# - Multiple query budgets (200, 500, 1000)
# - Trajectory-level metrics (CDT, TTF, SDR, TFP)
#
# Usage:
#   sbatch run_full_evaluation.sh
#
# Output:
#   - Per-task patches and results
#   - Aggregated summary across all tasks/queries
#   - Trajectory-level metric analysis
#
# =============================================================================

set -e  # Exit on error

# Environment setup
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0

# Paths
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
CODE_DIR="/data1/ma1/Ishaq/ump-vla/code"
OUTPUT_DIR="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack"

# Experiment configuration
SUITE="libero_spatial"
TASKS=(0 1 2 3 4 5 6 7 8 9)
QUERY_BUDGETS=(200 500 1000)
TRAIN_RATIO="0.7"
SEED="42"

# Fixed hyperparameters (from validation)
MINI_BATCH_SIZE="3"
DEVIATION_THRESHOLD="0.3"
POSITION_WEIGHT="1.0"
ROTATION_WEIGHT="1.0"
GRIPPER_WEIGHT="5.0"
LR="0.01"
PATCH_SIZE="32"

# Trajectory-level metric thresholds
DRIFT_THRESHOLD="0.2"
TASK_SCALE="0.5"
SDR_WINDOWS="10,25,50"
FRAMES_PER_EPISODE="10"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="full_eval_${TIMESTAMP}"
EXP_DIR="${OUTPUT_DIR}/experiments/${EXP_NAME}"

# Create directories
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/patches"
mkdir -p "${OUTPUT_DIR}/results"
mkdir -p "${EXP_DIR}"

# Main log file
MAIN_LOG="${EXP_DIR}/main_log.txt"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_section() {
    echo "" | tee -a "$MAIN_LOG"
    echo "=============================================================================" | tee -a "$MAIN_LOG"
    echo "$1" | tee -a "$MAIN_LOG"
    echo "=============================================================================" | tee -a "$MAIN_LOG"
}

# =============================================================================
# Start Experiment
# =============================================================================

log_section "DIRECTION 2: FULL EVALUATION - ALL LIBERO-SPATIAL TASKS"

log "Experiment Name: $EXP_NAME"
log "SLURM Job ID: $SLURM_JOB_ID"
log "Start Time: $(date)"
log ""
log "Configuration:"
log "  Suite: $SUITE"
log "  Tasks: ${TASKS[*]}"
log "  Query Budgets: ${QUERY_BUDGETS[*]}"
log "  Train Ratio: $TRAIN_RATIO"
log "  Seed: $SEED"
log "  Drift Threshold: ${DRIFT_THRESHOLD}m"
log "  Task Scale: ${TASK_SCALE}m"
log "  SDR Windows: $SDR_WINDOWS"

# Check GPU
log ""
log "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tee -a "$MAIN_LOG"

# Check Python
log ""
log "Python: $PYTHON"
$PYTHON --version | tee -a "$MAIN_LOG"

# =============================================================================
# Initialize Results Tracking
# =============================================================================

# Master results file (JSON)
MASTER_RESULTS="${EXP_DIR}/all_results.json"
echo "{" > "$MASTER_RESULTS"
echo "  \"experiment_name\": \"$EXP_NAME\"," >> "$MASTER_RESULTS"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$MASTER_RESULTS"
echo "  \"config\": {" >> "$MASTER_RESULTS"
echo "    \"suite\": \"$SUITE\"," >> "$MASTER_RESULTS"
echo "    \"tasks\": [${TASKS[*]// /, }]," >> "$MASTER_RESULTS"
echo "    \"query_budgets\": [${QUERY_BUDGETS[*]// /, }]," >> "$MASTER_RESULTS"
echo "    \"train_ratio\": $TRAIN_RATIO," >> "$MASTER_RESULTS"
echo "    \"seed\": $SEED," >> "$MASTER_RESULTS"
echo "    \"drift_threshold\": $DRIFT_THRESHOLD," >> "$MASTER_RESULTS"
echo "    \"task_scale\": $TASK_SCALE" >> "$MASTER_RESULTS"
echo "  }," >> "$MASTER_RESULTS"
echo "  \"results\": [" >> "$MASTER_RESULTS"

FIRST_RESULT=true

# Counters
TOTAL_RUNS=$((${#TASKS[@]} * ${#QUERY_BUDGETS[@]}))
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# =============================================================================
# Main Experiment Loop
# =============================================================================

for QUERIES in "${QUERY_BUDGETS[@]}"; do
    log_section "QUERY BUDGET: $QUERIES"

    for TASK_ID in "${TASKS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        log ""
        log "[$CURRENT_RUN/$TOTAL_RUNS] Task $TASK_ID with $QUERIES queries"
        log "-------------------------------------------"

        RUN_NAME="${EXP_NAME}_q${QUERIES}_task${TASK_ID}"
        RUN_START=$(date +%s)

        # =====================================================================
        # TRAINING
        # =====================================================================

        log "[TRAIN] Starting training for task $TASK_ID..."

        TRAIN_OUTPUT=$($PYTHON "${CODE_DIR}/scripts/train_patch.py" \
            --suite "$SUITE" \
            --task_id "$TASK_ID" \
            --train_ratio "$TRAIN_RATIO" \
            --queries "$QUERIES" \
            --mini_batch_size "$MINI_BATCH_SIZE" \
            --patch_size "$PATCH_SIZE" \
            --deviation_threshold "$DEVIATION_THRESHOLD" \
            --position_weight "$POSITION_WEIGHT" \
            --rotation_weight "$ROTATION_WEIGHT" \
            --gripper_weight "$GRIPPER_WEIGHT" \
            --lr "$LR" \
            --seed "$SEED" \
            --experiment_name "$RUN_NAME" \
            --output_dir "$OUTPUT_DIR" 2>&1) || {
            log "[ERROR] Training failed for task $TASK_ID with $QUERIES queries"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            continue
        }

        # Extract best deviation from training output
        TRAIN_BEST_DEV=$(echo "$TRAIN_OUTPUT" | grep -oP 'Best avg deviation:\s*\K[0-9.]+' | tail -1)
        log "[TRAIN] Completed. Best deviation: ${TRAIN_BEST_DEV:-N/A}"

        # Find patch file
        PATCH_FILE=$(ls -t "${OUTPUT_DIR}/patches/${RUN_NAME}"*_patch.npy 2>/dev/null | head -1)

        if [ -z "$PATCH_FILE" ]; then
            log "[ERROR] No patch file found for $RUN_NAME"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            continue
        fi

        log "[TRAIN] Patch saved: $(basename "$PATCH_FILE")"

        # =====================================================================
        # TESTING
        # =====================================================================

        log "[TEST] Starting testing for task $TASK_ID..."

        TEST_OUTPUT=$($PYTHON "${CODE_DIR}/scripts/test_patch.py" \
            --patch_path "$PATCH_FILE" \
            --suite "$SUITE" \
            --task_id "$TASK_ID" \
            --train_ratio "$TRAIN_RATIO" \
            --deviation_threshold "$DEVIATION_THRESHOLD" \
            --drift_threshold "$DRIFT_THRESHOLD" \
            --task_scale "$TASK_SCALE" \
            --sdr_windows "$SDR_WINDOWS" \
            --frames_per_episode "$FRAMES_PER_EPISODE" \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" 2>&1) || {
            log "[ERROR] Testing failed for task $TASK_ID with $QUERIES queries"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            continue
        }

        RUN_END=$(date +%s)
        RUN_TIME=$((RUN_END - RUN_START))

        log "[TEST] Completed in ${RUN_TIME}s"

        # Find results file
        RUN_ID=$(basename "$PATCH_FILE" | sed 's/_patch.npy//')
        RESULTS_FILE="${OUTPUT_DIR}/results/${RUN_ID}_testing.json"

        if [ ! -f "$RESULTS_FILE" ]; then
            log "[ERROR] Results file not found: $RESULTS_FILE"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            continue
        fi

        # =====================================================================
        # Extract Metrics
        # =====================================================================

        # Frame-level metrics
        AVG_DEV=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['deviation']['average'])" 2>/dev/null || echo "0")
        DEV_RATE=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['deviation']['rate'])" 2>/dev/null || echo "0")
        POS_DEV=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['components']['position'])" 2>/dev/null || echo "0")

        # Trajectory-level metrics
        CDT_RATE=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['cdt']['success_rate'])" 2>/dev/null || echo "0")
        TTF_MEAN=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); t=d['metrics']['trajectory']['ttf']['mean_frames']; print(t if t else 'null')" 2>/dev/null || echo "null")
        TFP_MEAN=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['tfp']['mean_score'])" 2>/dev/null || echo "0")
        TFP_ABOVE_1=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['tfp']['above_1_rate'])" 2>/dev/null || echo "0")
        DRIFT_MEAN=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['cumulative_drift']['mean'])" 2>/dev/null || echo "0")

        # SDR metrics
        SDR_10=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['sdr'].get('10', 0) or 0)" 2>/dev/null || echo "0")
        SDR_25=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['sdr'].get('25', 0) or 0)" 2>/dev/null || echo "0")
        SDR_50=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['metrics']['trajectory']['sdr'].get('50', 0) or 0)" 2>/dev/null || echo "0")

        # Log key metrics
        log "[METRICS] Frame: avg_dev=${AVG_DEV}, rate=${DEV_RATE}"
        log "[METRICS] Trajectory: CDT=${CDT_RATE}, TFP=${TFP_MEAN}, TTF=${TTF_MEAN}"

        # Add to master results
        if [ "$FIRST_RESULT" = false ]; then
            echo "," >> "$MASTER_RESULTS"
        fi
        FIRST_RESULT=false

        cat >> "$MASTER_RESULTS" << EOF
    {
      "task_id": $TASK_ID,
      "queries": $QUERIES,
      "run_id": "$RUN_ID",
      "run_time_seconds": $RUN_TIME,
      "patch_path": "$PATCH_FILE",
      "results_path": "$RESULTS_FILE",
      "frame_metrics": {
        "avg_deviation": $AVG_DEV,
        "deviation_rate": $DEV_RATE,
        "position_deviation": $POS_DEV
      },
      "trajectory_metrics": {
        "cdt_success_rate": $CDT_RATE,
        "ttf_mean_frames": $TTF_MEAN,
        "tfp_mean_score": $TFP_MEAN,
        "tfp_above_1_rate": $TFP_ABOVE_1,
        "drift_mean": $DRIFT_MEAN,
        "sdr_10": $SDR_10,
        "sdr_25": $SDR_25,
        "sdr_50": $SDR_50
      }
    }
EOF

        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))

    done  # End task loop
done  # End query loop

# =============================================================================
# Close Results JSON
# =============================================================================

echo "" >> "$MASTER_RESULTS"
echo "  ]," >> "$MASTER_RESULTS"
echo "  \"summary\": {" >> "$MASTER_RESULTS"
echo "    \"total_runs\": $TOTAL_RUNS," >> "$MASTER_RESULTS"
echo "    \"successful_runs\": $SUCCESSFUL_RUNS," >> "$MASTER_RESULTS"
echo "    \"failed_runs\": $FAILED_RUNS" >> "$MASTER_RESULTS"
echo "  }" >> "$MASTER_RESULTS"
echo "}" >> "$MASTER_RESULTS"

# =============================================================================
# Generate Summary Report
# =============================================================================

log_section "EXPERIMENT COMPLETE"

log "Total Runs: $TOTAL_RUNS"
log "Successful: $SUCCESSFUL_RUNS"
log "Failed: $FAILED_RUNS"
log ""
log "Results saved to:"
log "  - Master results: $MASTER_RESULTS"
log "  - Experiment log: $MAIN_LOG"
log ""
log "End Time: $(date)"

# =============================================================================
# Create Summary Table
# =============================================================================

SUMMARY_TABLE="${EXP_DIR}/summary_table.txt"

log ""
log "Generating summary table..."

cat > "$SUMMARY_TABLE" << 'EOF'
================================================================================
DIRECTION 2: FULL EVALUATION SUMMARY
================================================================================

FRAME-LEVEL METRICS (per task, per query budget)
--------------------------------------------------------------------------------
Task | Queries | Avg Dev | Dev Rate | Pos Dev
--------------------------------------------------------------------------------
EOF

# Extract and format results
$PYTHON << PYEOF >> "$SUMMARY_TABLE"
import json

try:
    with open("$MASTER_RESULTS", "r") as f:
        data = json.load(f)

    for r in data["results"]:
        print(f"{r['task_id']:4d} | {r['queries']:7d} | {r['frame_metrics']['avg_deviation']:7.4f} | {r['frame_metrics']['deviation_rate']*100:7.1f}% | {r['frame_metrics']['position_deviation']:7.4f}")
except Exception as e:
    print(f"Error: {e}")
PYEOF

cat >> "$SUMMARY_TABLE" << 'EOF'
--------------------------------------------------------------------------------

TRAJECTORY-LEVEL METRICS (per task, per query budget)
--------------------------------------------------------------------------------
Task | Queries | CDT Rate | TFP Score | TTF (frames) | Drift (m)
--------------------------------------------------------------------------------
EOF

$PYTHON << PYEOF >> "$SUMMARY_TABLE"
import json

try:
    with open("$MASTER_RESULTS", "r") as f:
        data = json.load(f)

    for r in data["results"]:
        ttf = r['trajectory_metrics']['ttf_mean_frames']
        ttf_str = f"{ttf:12.1f}" if ttf is not None else "         N/A"
        print(f"{r['task_id']:4d} | {r['queries']:7d} | {r['trajectory_metrics']['cdt_success_rate']*100:7.1f}% | {r['trajectory_metrics']['tfp_mean_score']:9.2f} | {ttf_str} | {r['trajectory_metrics']['drift_mean']:9.4f}")
except Exception as e:
    print(f"Error: {e}")
PYEOF

cat >> "$SUMMARY_TABLE" << 'EOF'
--------------------------------------------------------------------------------

AGGREGATE BY QUERY BUDGET
--------------------------------------------------------------------------------
EOF

$PYTHON << PYEOF >> "$SUMMARY_TABLE"
import json
import numpy as np

try:
    with open("$MASTER_RESULTS", "r") as f:
        data = json.load(f)

    # Group by query budget
    by_queries = {}
    for r in data["results"]:
        q = r["queries"]
        if q not in by_queries:
            by_queries[q] = []
        by_queries[q].append(r)

    print("Queries | Avg Dev (mean) | CDT Rate (mean) | TFP Score (mean) | TFP>1 Rate")
    print("-" * 80)

    for q in sorted(by_queries.keys()):
        results = by_queries[q]
        avg_dev = np.mean([r['frame_metrics']['avg_deviation'] for r in results])
        cdt_rate = np.mean([r['trajectory_metrics']['cdt_success_rate'] for r in results])
        tfp_mean = np.mean([r['trajectory_metrics']['tfp_mean_score'] for r in results])
        tfp_above_1 = np.mean([r['trajectory_metrics']['tfp_above_1_rate'] for r in results])

        print(f"{q:7d} | {avg_dev:14.4f} | {cdt_rate*100:14.1f}% | {tfp_mean:16.2f} | {tfp_above_1*100:10.1f}%")

except Exception as e:
    print(f"Error: {e}")
PYEOF

cat >> "$SUMMARY_TABLE" << EOF
--------------------------------------------------------------------------------

Experiment: $EXP_NAME
Date: $(date)
================================================================================
EOF

# Print summary table to log
cat "$SUMMARY_TABLE" | tee -a "$MAIN_LOG"

log ""
log "Summary table saved to: $SUMMARY_TABLE"
log ""
log "============================================================================="
log "EXPERIMENT FINISHED SUCCESSFULLY"
log "============================================================================="
