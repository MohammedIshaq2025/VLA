# QUICK START COMMANDS - PRIORITIES 1-3

**Essential commands to validate Direction 2 hypothesis**

---

## SETUP (One-Time)

```bash
# SSH into cluster
ssh your_username@cluster_address

# Navigate to project
cd /data1/ma1/Ishaq/ump-vla

# Make scripts executable
chmod +x code/SLURM/*.sh

# Verify files exist
ls -lh code/scripts/evaluate_closed_loop.py
ls -lh code/SLURM/run_closed_loop_eval.sh
ls -lh code/SLURM/train_gripper_focused.sh
ls -lh code/SLURM/run_full_validation_pipeline.sh
```

---

## OPTION 1: QUICK TEST (Recommended First)

**Test existing patch on single task (~4-6 hours)**

```bash
cd /data1/ma1/Ishaq/ump-vla

# Use existing Direction 2 validation patch
PATCH="outputs/se3_zoo_attack/patches/dir2_validation_task0_20260120_001550_patch.npy"

# Run closed-loop evaluation
sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" 0

# Check job status
squeue -u $USER

# View live output
tail -f outputs/se3_zoo_attack/logs/closed_loop_*.out
```

**When complete, check results**:
```bash
# Find result file
RESULT=$(ls -t outputs/se3_zoo_attack/closed_loop/*_results.json | head -1)

# Extract key metrics
cat "$RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
agg = data['aggregate_metrics']
print('='*60)
print('CRITICAL RESULTS:')
print('='*60)
print(f'Clean Success Rate:    {agg[\"mean_clean_success_rate\"]*100:.1f}%')
print(f'Attacked Success Rate: {agg[\"mean_attacked_success_rate\"]*100:.1f}%')
print(f'TRUE ASR:              {agg[\"mean_true_asr\"]*100:.1f}%')
print('='*60)
if agg['mean_true_asr'] >= 0.60:
    print('VERDICT: ✅ STRONG ATTACK - ECCV-WORTHY!')
elif agg['mean_true_asr'] >= 0.40:
    print('VERDICT: ⚠️  MODERATE ATTACK - Try gripper-focused')
else:
    print('VERDICT: ❌ WEAK ATTACK - Needs improvement')
print('='*60)
"
```

---

## OPTION 2: FULL PIPELINE (Comprehensive)

**Runs everything: baseline, gripper-focused, closed-loop (~8-12 hours for single task)**

```bash
cd /data1/ma1/Ishaq/ump-vla

# Run full validation on task 0
sbatch code/SLURM/run_full_validation_pipeline.sh 0

# Monitor progress
tail -f outputs/se3_zoo_attack/logs/full_validation_*.log

# When complete, view report
cat outputs/se3_zoo_attack/logs/full_validation_*_report.txt
```

---

## OPTION 3: ALL 10 TASKS (If initial results are good)

**Run closed-loop on all tasks (~24-30 hours)**

```bash
cd /data1/ma1/Ishaq/ump-vla

# Use existing or newly trained patch
PATCH="outputs/se3_zoo_attack/patches/your_patch.npy"

# Run all 10 tasks sequentially
sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" all

# Or run in parallel (requires 10 GPUs)
sbatch code/SLURM/run_closed_loop_array.sh "$PATCH"
```

---

## GRIPPER-FOCUSED ATTACK

**Train patches with high gripper weight (w_grip=5.0)**

```bash
cd /data1/ma1/Ishaq/ump-vla

# Single task
sbatch code/SLURM/train_gripper_focused.sh 0 5.0

# Wait for training to complete, then find patch
GRIPPER_PATCH=$(ls -t outputs/se3_zoo_attack/patches/gripper_focused_*.npy | head -1)
echo "Gripper patch: $GRIPPER_PATCH"

# Run closed-loop evaluation on gripper patch
sbatch code/SLURM/run_closed_loop_eval.sh "$GRIPPER_PATCH" 0
```

---

## CHECKING RESULTS

### View Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>
```

### View Logs
```bash
# Live output
tail -f outputs/se3_zoo_attack/logs/closed_loop_*.out

# Latest log file
ls -t outputs/se3_zoo_attack/logs/*.out | head -1 | xargs tail -100

# Search for errors
grep -i error outputs/se3_zoo_attack/logs/*.err
```

### Extract Metrics
```bash
# Closed-loop TRUE ASR
cat outputs/se3_zoo_attack/closed_loop/*_results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"TRUE ASR: {data['aggregate_metrics']['mean_true_asr']*100:.1f}%\")
"

# Drift consistency from test results
grep -A 2 "Drift Consistency" outputs/se3_zoo_attack/logs/test_*.out

# Gripper deviation
grep "Gripper Deviation" outputs/se3_zoo_attack/logs/*.out
```

---

## DECISION TREE (Based on Results)

```bash
# After closed-loop evaluation completes:

# 1. Check TRUE ASR
TRUE_ASR=<value from results>

if [ TRUE_ASR >= 60 ]; then
    echo "✅ VALIDATED! Run on all 10 tasks and start writing paper"
    sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" all
elif [ TRUE_ASR >= 40 ]; then
    echo "⚠️  MODERATE. Try gripper-focused attack"
    sbatch code/SLURM/train_gripper_focused.sh 0 5.0
else
    echo "❌ WEAK. Investigate failure modes and check drift consistency"
fi
```

---

## COMMON ISSUES

### Job stays in PENDING
```bash
# Check partition status
sinfo -p gpu2

# Check why job is pending
scontrol show job <JOB_ID>

# Solution: Wait for resources or reduce time limit
```

### Out of Memory
```bash
# Edit SLURM script to reduce episodes:
# --clean_episodes 50 → 30
# --attacked_episodes 50 → 30
```

### LIBERO environment errors
```bash
# Test headless rendering
cd /data1/ma1/Ishaq/ump-vla
python3 code/scripts/test_libero_headless.py

# If fails, check MUJOCO_GL
echo $MUJOCO_GL  # Should be "osmesa"
```

---

## FILE LOCATIONS

### Input Files
```
Patches:           outputs/se3_zoo_attack/patches/*.npy
OpenVLA Model:     /data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b
LIBERO Data:       /data1/ma1/Ishaq/ump-vla/data/libero/
```

### Output Files
```
Closed-Loop:       outputs/se3_zoo_attack/closed_loop/*_results.json
Training Results:  outputs/se3_zoo_attack/results/*_testing.json
Logs:              outputs/se3_zoo_attack/logs/*.out
Reports:           outputs/se3_zoo_attack/logs/*_report.txt
```

### Scripts
```
Python:            code/scripts/evaluate_closed_loop.py
SLURM:             code/SLURM/run_*.sh
Documentation:     PRIORITIES_1-3_IMPLEMENTATION_GUIDE.md
```

---

## TIMELINE ESTIMATES

| Task | Duration | Command |
|------|----------|---------|
| Single task closed-loop | 4-6 hours | `sbatch ... run_closed_loop_eval.sh "$PATCH" 0` |
| All 10 tasks closed-loop | 24-30 hours | `sbatch ... run_closed_loop_eval.sh "$PATCH" all` |
| Gripper-focused training | 10-15 min | `sbatch ... train_gripper_focused.sh 0 5.0` |
| Full pipeline (single task) | 8-12 hours | `sbatch ... run_full_validation_pipeline.sh 0` |
| Full pipeline (all 10 tasks) | 48-60 hours | `sbatch ... run_full_validation_pipeline.sh all` |

---

## NEXT STEPS AFTER VALIDATION

**If TRUE ASR >= 60%**:
1. Run all 10 tasks: `sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" all`
2. Run 3 seeds for statistics
3. Start writing ECCV paper

**If TRUE ASR 40-60%**:
1. Try gripper-focused: `sbatch code/SLURM/train_gripper_focused.sh 0 5.0`
2. Re-evaluate with gripper patch
3. Consider V3 optimizations if still < 60%

**If TRUE ASR < 40%**:
1. Check drift consistency: `grep "Drift Consistency" logs/*.out`
2. Analyze failure modes
3. Consider pivot to different attack

---

## DOCUMENTATION

Full details in:
- `PRIORITIES_1-3_IMPLEMENTATION_GUIDE.md` (800 lines, comprehensive guide)
- `IMPLEMENTATION_SUMMARY_PRIORITIES_1-3.md` (Summary of what was implemented)

---

**YOU'RE READY!**

Start with OPTION 1 (Quick Test) to validate on a single task first.
Based on those results, decide whether to proceed with full evaluation.

*Last Updated: 2026-01-21*
