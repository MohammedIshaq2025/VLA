#!/bin/bash
#SBATCH --job-name=full_validation
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmohammedsho@student.unimelb.edu.au
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/full_validation_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/full_validation_%j.err

# =============================================================================
# FULL VALIDATION PIPELINE - PRIORITIES 1-3 COMBINED
# =============================================================================
#
# This master script orchestrates the complete validation of Direction 2:
#
# STEP 1: Establish clean baseline (Priority 1)
#   - Run closed-loop evaluation without attack
#   - Measure baseline task success rates
#
# STEP 2: Train baseline patch (current weights)
#   - w_pos=1.0, w_rot=0.5, w_grip=0.1
#   - Establish baseline single-frame ASR
#
# STEP 3: Train gripper-focused patch (Priority 3)
#   - w_pos=1.0, w_rot=0.5, w_grip=5.0
#   - Test if higher gripper weight increases flip rate
#
# STEP 4: Closed-loop evaluation of both patches (Priority 1)
#   - Measure TRUE task success rates
#   - Compare baseline vs gripper-focused
#
# STEP 5: Generate comparison report
#   - Single-frame ASR vs TRUE ASR
#   - Validate Direction 2 hypothesis
#
# Usage:
#   # Full pipeline on task 0
#   sbatch run_full_validation_pipeline.sh 0
#
#   # Full pipeline on all 10 tasks (WARNING: ~48 hours)
#   sbatch run_full_validation_pipeline.sh all
#
# =============================================================================

# === Parameters ===
TASK_SPEC="${1:-0}"
SUITE="${SUITE:-libero_spatial}"
QUERIES="${QUERIES:-200}"

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache
export CUDA_VISIBLE_DEVICES=0

# === Paths ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
PROJECT_ROOT="/data1/ma1/Ishaq/ump-vla"
CODE_DIR="${PROJECT_ROOT}/code"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/se3_zoo_attack"
CL_DIR="${OUTPUT_DIR}/closed_loop"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$CL_DIR"

# === Experiment Name ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_NAME="full_validation_${SUITE}_${TIMESTAMP}"
PIPELINE_LOG="${OUTPUT_DIR}/logs/${PIPELINE_NAME}.log"

# === Initialize Log ===
exec > >(tee -a "$PIPELINE_LOG")
exec 2>&1

echo "============================================================================================================"
echo "FULL VALIDATION PIPELINE - DIRECTION 2 COMPREHENSIVE TEST"
echo "============================================================================================================"
echo "Pipeline: $PIPELINE_NAME"
echo "Start time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "============================================================================================================"
echo "Configuration:"
echo "  Suite: $SUITE"
echo "  Task(s): $TASK_SPEC"
echo "  Queries: $QUERIES"
echo "============================================================================================================"
echo ""

# === Verify Python ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found at $PYTHON"
    exit 1
fi

cd "$PROJECT_ROOT"

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
fi

# === Parse Task Specification ===
if [ "$TASK_SPEC" = "all" ]; then
    TASKS=(0 1 2 3 4 5 6 7 8 9)
    echo "Running full pipeline on ALL 10 tasks (estimated: 40-48 hours)"
else
    IFS=',' read -ra TASKS <<< "$TASK_SPEC"
    echo "Running full pipeline on ${#TASKS[@]} task(s): ${TASKS[*]}"
fi

echo ""
echo "============================================================================================================"
echo "STEP 1/5: CLEAN BASELINE (No Attack)"
echo "============================================================================================================"
echo ""

STEP1_START=$(date +%s)

for TASK_ID in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Task $TASK_ID: Establishing clean baseline"
    echo "----------------------------------------"

    $PYTHON -u "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --clean_episodes 50 \
        --attacked_episodes 0 \
        --max_steps 300 \
        --experiment_name "clean_baseline_task${TASK_ID}_${TIMESTAMP}" \
        --output_dir "$CL_DIR" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ Clean baseline failed for task $TASK_ID"
    else
        echo "✅ Clean baseline complete for task $TASK_ID"
    fi
    echo ""
done

STEP1_END=$(date +%s)
STEP1_TIME=$((STEP1_END - STEP1_START))
echo "STEP 1 completed in ${STEP1_TIME}s ($((STEP1_TIME / 60)) minutes)"
echo ""

# === STEP 2: Train Baseline Patches ===
echo "============================================================================================================"
echo "STEP 2/5: TRAIN BASELINE PATCHES (w_grip=0.1)"
echo "============================================================================================================"
echo ""

STEP2_START=$(date +%s)
BASELINE_PATCHES=()

for TASK_ID in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Task $TASK_ID: Training baseline patch"
    echo "----------------------------------------"

    EXP_NAME="baseline_patch_task${TASK_ID}_${TIMESTAMP}"

    $PYTHON -u "${CODE_DIR}/scripts/train_patch.py" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio 0.7 \
        --queries "$QUERIES" \
        --position_weight 1.0 \
        --rotation_weight 0.5 \
        --gripper_weight 0.1 \
        --experiment_name "$EXP_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ Baseline training failed for task $TASK_ID"
        continue
    fi

    # Find patch
    PATCH=$(ls -t "${OUTPUT_DIR}/patches/${EXP_NAME}"*_patch.npy 2>/dev/null | head -1)
    if [ -z "$PATCH" ]; then
        echo "❌ Patch file not found for task $TASK_ID"
        continue
    fi

    BASELINE_PATCHES[$TASK_ID]="$PATCH"
    echo "✅ Baseline patch trained: $PATCH"

    # Test patch
    $PYTHON -u "${CODE_DIR}/scripts/test_patch.py" \
        --patch_path "$PATCH" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio 0.7 \
        --deviation_threshold 0.3 \
        --drift_threshold 0.05 \
        --task_scale 0.5 \
        --seed 42 \
        --output_dir "$OUTPUT_DIR"

    echo ""
done

STEP2_END=$(date +%s)
STEP2_TIME=$((STEP2_END - STEP2_START))
echo "STEP 2 completed in ${STEP2_TIME}s ($((STEP2_TIME / 60)) minutes)"
echo ""

# === STEP 3: Train Gripper-Focused Patches ===
echo "============================================================================================================"
echo "STEP 3/5: TRAIN GRIPPER-FOCUSED PATCHES (w_grip=5.0)"
echo "============================================================================================================"
echo ""

STEP3_START=$(date +%s)
GRIPPER_PATCHES=()

for TASK_ID in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Task $TASK_ID: Training gripper-focused patch"
    echo "----------------------------------------"

    EXP_NAME="gripper_focused_task${TASK_ID}_${TIMESTAMP}"

    $PYTHON -u "${CODE_DIR}/scripts/train_patch.py" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio 0.7 \
        --queries "$QUERIES" \
        --position_weight 1.0 \
        --rotation_weight 0.5 \
        --gripper_weight 5.0 \
        --experiment_name "$EXP_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ Gripper training failed for task $TASK_ID"
        continue
    fi

    # Find patch
    PATCH=$(ls -t "${OUTPUT_DIR}/patches/${EXP_NAME}"*_patch.npy 2>/dev/null | head -1)
    if [ -z "$PATCH" ]; then
        echo "❌ Patch file not found for task $TASK_ID"
        continue
    fi

    GRIPPER_PATCHES[$TASK_ID]="$PATCH"
    echo "✅ Gripper patch trained: $PATCH"

    # Test patch
    $PYTHON -u "${CODE_DIR}/scripts/test_patch.py" \
        --patch_path "$PATCH" \
        --suite "$SUITE" \
        --task_id "$TASK_ID" \
        --train_ratio 0.7 \
        --deviation_threshold 0.3 \
        --drift_threshold 0.05 \
        --task_scale 0.5 \
        --seed 42 \
        --output_dir "$OUTPUT_DIR"

    echo ""
done

STEP3_END=$(date +%s)
STEP3_TIME=$((STEP3_END - STEP3_START))
echo "STEP 3 completed in ${STEP3_TIME}s ($((STEP3_TIME / 60)) minutes)"
echo ""

# === STEP 4: Closed-Loop Evaluation ===
echo "============================================================================================================"
echo "STEP 4/5: CLOSED-LOOP EVALUATION (THE CRITICAL TEST)"
echo "============================================================================================================"
echo ""

STEP4_START=$(date +%s)

for TASK_ID in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Task $TASK_ID: Closed-Loop Evaluation"
    echo "=========================================="

    # Evaluate baseline patch
    if [ -n "${BASELINE_PATCHES[$TASK_ID]}" ]; then
        echo "[4a] Evaluating BASELINE patch..."
        $PYTHON -u "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
            --suite "$SUITE" \
            --task_id "$TASK_ID" \
            --patch_path "${BASELINE_PATCHES[$TASK_ID]}" \
            --clean_episodes 50 \
            --attacked_episodes 50 \
            --max_steps 300 \
            --experiment_name "cl_baseline_task${TASK_ID}_${TIMESTAMP}" \
            --output_dir "$CL_DIR" \
            --seed 42

        if [ $? -eq 0 ]; then
            echo "✅ Baseline closed-loop evaluation complete"
        else
            echo "❌ Baseline closed-loop evaluation failed"
        fi
        echo ""
    fi

    # Evaluate gripper-focused patch
    if [ -n "${GRIPPER_PATCHES[$TASK_ID]}" ]; then
        echo "[4b] Evaluating GRIPPER-FOCUSED patch..."
        $PYTHON -u "${CODE_DIR}/scripts/evaluate_closed_loop.py" \
            --suite "$SUITE" \
            --task_id "$TASK_ID" \
            --patch_path "${GRIPPER_PATCHES[$TASK_ID]}" \
            --clean_episodes 50 \
            --attacked_episodes 50 \
            --max_steps 300 \
            --experiment_name "cl_gripper_task${TASK_ID}_${TIMESTAMP}" \
            --output_dir "$CL_DIR" \
            --seed 42

        if [ $? -eq 0 ]; then
            echo "✅ Gripper closed-loop evaluation complete"
        else
            echo "❌ Gripper closed-loop evaluation failed"
        fi
        echo ""
    fi
done

STEP4_END=$(date +%s)
STEP4_TIME=$((STEP4_END - STEP4_START))
echo "STEP 4 completed in ${STEP4_TIME}s ($((STEP4_TIME / 60)) minutes)"
echo ""

# === STEP 5: Generate Report ===
echo "============================================================================================================"
echo "STEP 5/5: GENERATE VALIDATION REPORT"
echo "============================================================================================================"
echo ""

REPORT_FILE="${OUTPUT_DIR}/logs/${PIPELINE_NAME}_report.txt"

cat > "$REPORT_FILE" <<EOF
================================================================================
FULL VALIDATION PIPELINE REPORT
================================================================================
Pipeline: $PIPELINE_NAME
Date: $(date)
Suite: $SUITE
Tasks: ${TASKS[*]}

================================================================================
TIMING SUMMARY
================================================================================
Step 1 (Clean Baseline):        ${STEP1_TIME}s ($((STEP1_TIME / 60)) min)
Step 2 (Train Baseline):         ${STEP2_TIME}s ($((STEP2_TIME / 60)) min)
Step 3 (Train Gripper):          ${STEP3_TIME}s ($((STEP3_TIME / 60)) min)
Step 4 (Closed-Loop Eval):       ${STEP4_TIME}s ($((STEP4_TIME / 60)) min)
Total:                           $((STEP1_TIME + STEP2_TIME + STEP3_TIME + STEP4_TIME))s ($((($STEP1_TIME + STEP2_TIME + STEP3_TIME + STEP4_TIME) / 60)) min)

================================================================================
FILES GENERATED
================================================================================

Patches (Baseline):
EOF

for TASK_ID in "${TASKS[@]}"; do
    if [ -n "${BASELINE_PATCHES[$TASK_ID]}" ]; then
        echo "  Task $TASK_ID: ${BASELINE_PATCHES[$TASK_ID]}" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

Patches (Gripper-Focused):
EOF

for TASK_ID in "${TASKS[@]}"; do
    if [ -n "${GRIPPER_PATCHES[$TASK_ID]}" ]; then
        echo "  Task $TASK_ID: ${GRIPPER_PATCHES[$TASK_ID]}" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

Closed-Loop Results:
  Directory: $CL_DIR
  Files: cl_*_${TIMESTAMP}_results.json

================================================================================
NEXT STEPS
================================================================================

1. Review closed-loop results:
   ls -lh ${CL_DIR}/*_${TIMESTAMP}_results.json

2. Compare baseline vs gripper-focused:
   - Check TRUE ASR for both approaches
   - Verify if gripper weight increase improved results

3. Analyze Direction 2 hypothesis:
   - Does TRUE ASR > single-frame ASR?
   - Is cumulative drift correlated with task failure?

4. If TRUE ASR >= 60%:
   ✅ Direction 2 validated! Ready for ECCV paper.

   If TRUE ASR 40-60%:
   ⚠️  Moderate results. Consider improvements (V3 optimizations).

   If TRUE ASR < 40%:
   ❌ Hypothesis needs revision. Investigate failure modes.

================================================================================
PRIORITY ACTIONS BASED ON RESULTS
================================================================================

IF TRUE ASR > 60% (Strong Attack):
  1. Run on all 10 tasks (3 seeds each)
  2. Start writing ECCV paper
  3. Focus on analysis and comparison sections

IF TRUE ASR 40-60% (Moderate Attack):
  1. Implement V3 optimizer improvements (Improvements 4 & 6)
  2. Try larger patch sizes (48×48, 64×64)
  3. Re-run closed-loop evaluation

IF TRUE ASR < 40% (Weak Attack):
  1. Analyze failure modes (why aren't tasks failing?)
  2. Test position-only vs gripper-only attacks
  3. Consider pivot to different attack objective

================================================================================
VALIDATION CHECKLIST
================================================================================

Priority 1 (CRITICAL):
  [✓] Closed-loop evaluation implemented
  [✓] Task success rates measured
  [ ] TRUE ASR computed and compared to single-frame ASR

Priority 2 (IMPORTANT):
  [✓] Drift calculations verified (actual vs legacy)
  [ ] Drift consistency checked (oscillation detection)

Priority 3 (IMPORTANT):
  [✓] Gripper-focused attack tested
  [ ] Gripper flip rate improvement confirmed

================================================================================
END OF REPORT
================================================================================
EOF

cat "$REPORT_FILE"

echo ""
echo "============================================================================================================"
echo "FULL VALIDATION PIPELINE COMPLETE"
echo "============================================================================================================"
echo "Total time: $((($STEP1_TIME + STEP2_TIME + STEP3_TIME + STEP4_TIME) / 60)) minutes"
echo "Report saved: $REPORT_FILE"
echo "Pipeline log: $PIPELINE_LOG"
echo "============================================================================================================"
echo ""

exit 0
