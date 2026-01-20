# PRIORITIES 1-3 IMPLEMENTATION GUIDE

**ECCV 2026 Critical Validation Experiments**

This guide provides complete instructions for running Priorities 1-3 experiments that will validate (or falsify) the Direction 2 hypothesis.

---

## EXECUTIVE SUMMARY

**What was implemented:**
1. ✅ **Priority 1**: Closed-loop LIBERO evaluation (THE critical experiment)
2. ✅ **Priority 2**: Fixed drift calculations (actual drift vs legacy sum-of-norms)
3. ✅ **Priority 3**: Gripper-focused attack training (w_grip=5.0)

**What you need to do:**
- Run closed-loop evaluation to measure TRUE task success rates
- Compare TRUE ASR vs single-frame ASR proxy metrics
- Based on results, decide if Direction 2 is ECCV-worthy

**Timeline:**
- Single task validation: ~4-6 hours
- Full validation (all 10 tasks): ~24-48 hours

---

## TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Priority 1: Closed-Loop Evaluation](#priority-1-closed-loop-evaluation)
3. [Priority 2: Drift Calculation Verification](#priority-2-drift-calculation-verification)
4. [Priority 3: Gripper-Focused Attack](#priority-3-gripper-focused-attack)
5. [Full Validation Pipeline](#full-validation-pipeline)
6. [Interpreting Results](#interpreting-results)
7. [Decision Tree](#decision-tree)
8. [Troubleshooting](#troubleshooting)

---

## QUICK START

### Option A: Test on Single Task (Recommended First)

```bash
# SSH into cluster
cd /data1/ma1/Ishaq/ump-vla

# Use existing Direction 2 validation patch
PATCH="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/patches/dir2_validation_task0_20260120_001550_patch.npy"

# Run closed-loop evaluation on task 0
sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" 0

# Check job status
squeue -u $USER

# View results when complete
tail -f outputs/se3_zoo_attack/logs/closed_loop_*.out
```

**Expected runtime**: 4-6 hours
**Expected result**: You'll see TRUE ASR (task success rate)

### Option B: Run Full Validation Pipeline

```bash
# This runs EVERYTHING (Steps 1-5):
# - Clean baseline
# - Train baseline patch
# - Train gripper-focused patch
# - Closed-loop evaluation of both
# - Generate comparison report

sbatch code/SLURM/run_full_validation_pipeline.sh 0
```

**Expected runtime**: 8-12 hours (single task)
**Expected result**: Complete validation report

---

## PRIORITY 1: CLOSED-LOOP EVALUATION

### What is Closed-Loop Evaluation?

**Your current evaluation** (open-loop):
- Load pre-recorded demonstration frames
- Apply patch to each frame
- Measure: "How much did the prediction change?" (proxy metric)
- **Problem**: Never actually run the robot, never see if tasks fail

**Closed-loop evaluation** (THE CRITICAL TEST):
- Initialize LIBERO environment
- Run robot policy in real-time simulation
- Actions are EXECUTED and affect next observation
- Measure: "Did the task succeed or fail?" (ground truth)

### Why This Matters

Your Direction 2 claim: "2.08m cumulative drift causes task failure"

**This is an ASSUMPTION until you run closed-loop!**

Possible outcomes:
- Best case: Clean 85% success, Attacked 20% success → **TRUE ASR = 76%** (ECCV-worthy!)
- Middle case: Clean 85%, Attacked 55% → TRUE ASR = 35% (publishable, but weaker)
- Worst case: Clean 85%, Attacked 75% → TRUE ASR = 12% (hypothesis falsified)

### Running Closed-Loop Evaluation

#### Script 1: Single Task or Multiple Tasks

```bash
# Using existing patch
cd /data1/ma1/Ishaq/ump-vla

# Single task
sbatch code/SLURM/run_closed_loop_eval.sh <patch_path> <task_id>

# All 10 tasks sequentially
sbatch code/SLURM/run_closed_loop_eval.sh <patch_path> all

# Clean baseline (no attack)
sbatch code/SLURM/run_closed_loop_eval.sh none 0
```

**Examples**:
```bash
# Test Direction 2 validation patch on task 0
PATCH="outputs/se3_zoo_attack/patches/dir2_validation_task0_20260120_001550_patch.npy"
sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" 0

# Test on all 10 tasks
sbatch code/SLURM/run_closed_loop_eval.sh "$PATCH" all

# Clean baseline for task 0
sbatch code/SLURM/run_closed_loop_eval.sh none 0
```

#### Script 2: Array Job (Parallel Evaluation)

```bash
# Runs all 10 tasks in PARALLEL (10 jobs simultaneously)
# WARNING: Requires cluster resources for 10 GPUs
sbatch code/SLURM/run_closed_loop_array.sh <patch_path>
```

**Use this only if:**
- Your cluster has enough free resources
- You want faster results (trades GPU hours for wall-clock time)

### Output Files

Results saved to: `outputs/se3_zoo_attack/closed_loop/`

**Key file**: `<experiment_name>_results.json`

**Structure**:
```json
{
  "aggregate_metrics": {
    "mean_clean_success_rate": 0.85,
    "mean_attacked_success_rate": 0.25,
    "mean_true_asr": 0.71,  // ← THIS IS THE CRITICAL NUMBER
    ...
  },
  "per_task_results": [...]
}
```

### Understanding Results

**TRUE ASR** (Attack Success Rate) = (Clean Success - Attacked Success) / Clean Success

| TRUE ASR | Interpretation | ECCV Viability |
|----------|----------------|----------------|
| > 70% | **Strong attack** - Single-frame proxies underestimated vulnerability | ✅ ACCEPT |
| 50-70% | **Moderate attack** - Effect is real but smaller than hypothesized | ⚠️ BORDERLINE |
| 30-50% | **Weak attack** - Marginal effect, needs improvement | ⚠️ REVISE |
| < 30% | **Ineffective attack** - Hypothesis needs major revision | ❌ REJECT |

---

## PRIORITY 2: DRIFT CALCULATION VERIFICATION

### The Problem

Your reported "2.08m cumulative drift" might be **MISLEADING** if calculated incorrectly.

**Two ways to calculate drift**:

1. **Sum of Norms** (LEGACY, WRONG):
   ```
   Cumulative = Σ ||Δpos[t]||
   ```
   If attack oscillates (left, right, left, right), this sums ALL movements even if they cancel out!

2. **Norm of Sum** (ACTUAL, CORRECT):
   ```
   Actual Drift = ||Σ Δpos[t]||
   ```
   This is where the robot ACTUALLY ends up relative to clean trajectory.

**Example where they differ**:
```
Frames: [+0.02m, -0.02m, +0.02m, -0.02m] × 50
Sum of norms: 0.02 × 200 = 4.0m  (looks huge!)
Norm of sum: ||0|| = 0.0m  (actually no drift!)
```

### Checking Your Drift Calculations

The `test_patch.py` script (already updated) computes BOTH metrics:

**Run your existing validation results through updated test script**:
```bash
PATCH="outputs/se3_zoo_attack/patches/dir2_validation_task0_20260120_001550_patch.npy"

sbatch code/SLURM/test_patch.sh "$PATCH" 0
```

**Look for these metrics in output**:
```
[ACTUAL TRAJECTORY DRIFT (||Σ Δ|| - the CORRECT metric)]
  Mean Actual Drift:    X.XXXXm
  Drift Consistency:    0.XX
    (1.0 = all deviations same direction, <0.5 = significant oscillation)

[LEGACY CUMULATIVE DRIFT (Σ ||Δ|| - for comparison only)]
  Mean Legacy Drift:    Y.YYYYm
```

**Critical check**:
- If `Drift Consistency >= 0.7`: Your attack is consistent ✅
- If `Drift Consistency < 0.5`: Your attack oscillates ❌ (actual drift << reported drift)

**If oscillating**:
- Your 2.08m claim is inflated
- Actual drift might be only 1.0m or less
- Update paper claims based on actual drift, not legacy

---

## PRIORITY 3: GRIPPER-FOCUSED ATTACK

### The Problem

**Across ALL experiments (Exp1-4, Dir2)**: Gripper flip rate = 0.00%

**Why this matters**:
- Manipulation tasks REQUIRE gripper control (grasp, release)
- If you can't flip gripper, many task failures won't occur
- Example: Robot goes to wrong location (position drift) but STILL grasps correctly → Task might succeed despite position error!

### The Solution

**Increase gripper weight** from 0.1 → 5.0 (50× increase)

**Hypothesis**: Higher gripper weight → more gripper flips → higher task failure rate

### Running Gripper-Focused Training

```bash
cd /data1/ma1/Ishaq/ump-vla

# Train single task with gripper focus
sbatch code/SLURM/train_gripper_focused.sh <task_id> <gripper_weight>

# Examples:
sbatch code/SLURM/train_gripper_focused.sh 0 5.0   # w_grip = 5.0
sbatch code/SLURM/train_gripper_focused.sh 0 10.0  # w_grip = 10.0

# Train all 10 tasks
for i in {0..9}; do
    sbatch code/SLURM/train_gripper_focused.sh $i 5.0
done
```

### Expected Results

**Success criteria**:
- Gripper deviation: 0.47 → 0.8-1.2 (need >= 1.0 for flip)
- Gripper flip rate: 0% → 30-50%
- TRUE ASR in closed-loop: +20-40% improvement

**If this works**:
- Run closed-loop on gripper-focused patches
- Compare to baseline patches
- Use higher TRUE ASR in paper

---

## FULL VALIDATION PIPELINE

For comprehensive validation, run the master pipeline script:

```bash
cd /data1/ma1/Ishaq/ump-vla

# Single task (recommended first)
sbatch code/SLURM/run_full_validation_pipeline.sh 0

# All 10 tasks (WARNING: ~48 hours)
sbatch code/SLURM/run_full_validation_pipeline.sh all
```

**This runs**:
1. Clean baseline (50 episodes, no attack)
2. Train baseline patch (w_grip=0.1)
3. Train gripper patch (w_grip=5.0)
4. Closed-loop eval of baseline patch
5. Closed-loop eval of gripper patch
6. Generate comparison report

**Output**: `outputs/se3_zoo_attack/logs/full_validation_*_report.txt`

**Report includes**:
- Timing for each step
- Paths to all patches
- Paths to all results
- Comparison of baseline vs gripper-focused
- Decision tree for next steps

---

## INTERPRETING RESULTS

### Key Files to Check

1. **Closed-Loop Results**:
   ```bash
   # List all closed-loop result files
   ls -lh outputs/se3_zoo_attack/closed_loop/*_results.json

   # View specific result
   cat outputs/se3_zoo_attack/closed_loop/<experiment_name>_results.json | python3 -m json.tool
   ```

2. **Extract Key Metrics**:
   ```bash
   python3 -c "
   import json
   with open('outputs/se3_zoo_attack/closed_loop/<file>.json') as f:
       data = json.load(f)
   agg = data['aggregate_metrics']
   print(f\"Clean Success: {agg['mean_clean_success_rate']*100:.1f}%\")
   print(f\"Attacked Success: {agg['mean_attacked_success_rate']*100:.1f}%\")
   print(f\"TRUE ASR: {agg['mean_true_asr']*100:.1f}%\")
   "
   ```

3. **Check Drift Consistency**:
   ```bash
   # Look for trajectory metrics in test results
   grep -A 10 "ACTUAL TRAJECTORY DRIFT" outputs/se3_zoo_attack/logs/test_*.out
   ```

### Metrics Comparison Table

| Metric | Where to Find | What It Means |
|--------|---------------|---------------|
| **Single-Frame ASR** | `test_patch.py` output → `deviation.rate` | % of frames with deviation > threshold (PROXY) |
| **TRUE ASR** | `evaluate_closed_loop.py` output → `mean_true_asr` | % reduction in task success (GROUND TRUTH) |
| **Legacy Drift** | `test_patch.py` output → `legacy_cumulative_drift` | Sum of norms (potentially inflated) |
| **Actual Drift** | `test_patch.py` output → `actual_drift.mean` | Norm of sum (TRUE displacement) |
| **Drift Consistency** | `test_patch.py` output → `drift_consistency` | 1.0 = consistent, <0.5 = oscillating |
| **Gripper Deviation** | `test_patch.py` output → `components.gripper` | Need >= 1.0 for gripper flip |

---

## DECISION TREE

### After Running Closed-Loop Evaluation

```
TRUE ASR >= 60%?
├─ YES → ✅ DIRECTION 2 VALIDATED!
│         Next steps:
│         1. Run on all 10 tasks (3 seeds each)
│         2. Start writing ECCV paper
│         3. Focus on comparison to Wang et al. (ICCV 2025)
│         4. Emphasize novel contribution: closed-loop evaluation paradigm
│
└─ NO → TRUE ASR 40-60%?
    ├─ YES → ⚠️ MODERATE RESULTS
    │         Next steps:
    │         1. Check if gripper-focused patch improves results
    │         2. Try V3 optimizer improvements (Improvements 4 & 6)
    │         3. Try larger patches (48×48, 64×64)
    │         4. Re-run closed-loop
    │
    └─ NO → TRUE ASR < 40%?
        └─ YES → ❌ WEAK/INEFFECTIVE
                  Next steps:
                  1. Analyze failure modes (why aren't tasks failing?)
                  2. Check drift consistency (is attack oscillating?)
                  3. Try position-only vs gripper-only attacks
                  4. Consider pivot to different attack objective
```

### Based on Drift Consistency

```
Drift Consistency >= 0.7?
├─ YES → ✅ ATTACK IS CONSISTENT
│         Your 2.08m drift claim is valid.
│         Proceed with confidence.
│
└─ NO → Drift Consistency < 0.5?
    └─ YES → ❌ ATTACK IS OSCILLATING
              Your attack cancels itself out!
              Actual drift << legacy drift.
              Actions needed:
              1. Update paper claims (use actual drift, not legacy)
              2. Fix optimizer to ensure consistent deviation direction
              3. Re-train patches with corrected optimizer
```

### Based on Gripper Flip Rate

```
Gripper Deviation >= 1.0?
├─ YES → ✅ GRIPPER FLIP ACHIEVED
│         w_grip=5.0 worked!
│         Use gripper-focused patches for paper.
│
└─ NO → Gripper Deviation < 0.7?
    └─ YES → ❌ STILL NO GRIPPER FLIP
              Try:
              1. w_grip=10.0 or higher
              2. Gripper-only attack (w_pos=0, w_rot=0, w_grip=1.0)
              3. Larger patches (more visual influence)
```

---

## TROUBLESHOOTING

### Common Issues

#### Issue 1: Closed-Loop Evaluation Crashes

**Symptoms**:
```
Failed to create environment: ...
```

**Solutions**:
1. Check MUJOCO_GL is set: `export MUJOCO_GL=osmesa`
2. Verify LIBERO installation: `python3 -c "import libero; print('OK')"`
3. Run headless test: `python3 code/scripts/test_libero_headless.py`

#### Issue 2: Job Stays in PENDING

**Check**:
```bash
squeue -u $USER
scontrol show job <JOB_ID>
```

**Reasons**:
- No GPU available (wait or reduce time limit)
- Partition busy (check `sinfo -p gpu2`)

#### Issue 3: Out of Memory

**Symptoms**:
```
CUDA out of memory
```

**Solutions**:
1. Reduce `--clean_episodes` and `--attacked_episodes` from 50 to 30
2. Request more GPU memory in SLURM script (if available)
3. Run tasks sequentially instead of array job

#### Issue 4: Results Show 0% Success on Clean Policy

**Problem**: LIBERO environment setup issue or task is too hard

**Debug**:
1. Check clean baseline results first (run without patch)
2. If clean success < 50%, there's an environment/model problem
3. Verify OpenVLA checkpoint path is correct
4. Try simpler task (task 0 is usually easiest)

---

## TIMELINE ESTIMATES

### Single Task Validation
- Clean baseline: 2-3 hours (50 episodes)
- Train baseline patch: 10-15 minutes (200 queries)
- Train gripper patch: 10-15 minutes (200 queries)
- Closed-loop baseline: 2-3 hours (50 + 50 episodes)
- Closed-loop gripper: 2-3 hours (50 + 50 episodes)
- **Total**: ~8-12 hours

### All 10 Tasks (Sequential)
- Clean baseline: 20-30 hours
- Train patches: 2-3 hours
- Closed-loop eval: 20-30 hours
- **Total**: ~45-65 hours (~2 days)

### All 10 Tasks (Parallel Array Jobs)
- All steps in parallel: ~5-8 hours
- **Requires**: 10 GPUs available simultaneously

---

## NEXT STEPS AFTER VALIDATION

### If TRUE ASR >= 60% (Strong Attack)

1. **Immediate** (Week 1-2):
   - Run all 10 libero_spatial tasks
   - Run 3 seeds for statistical rigor
   - Aggregate results

2. **Paper Writing** (Week 3-4):
   - Introduction: Emphasize closed-loop evaluation gap
   - Method: Describe closed-loop framework
   - Results: Single-frame ASR (50%) vs TRUE ASR (70%+)
   - Analysis: Why proxy metrics underestimate vulnerability
   - Comparison: Wang et al. ICCV 2025 (their 100% single-frame might be 50% true)

3. **Submit to ECCV 2026**: Deadline March 5, 2026

### If TRUE ASR 40-60% (Moderate Attack)

1. **Improvements** (Week 1-2):
   - V3 optimizer (Improvements 4 & 6)
   - Gripper-focused if not done
   - Larger patches

2. **Re-evaluate** (Week 2-3):
   - Re-run closed-loop with improved patches
   - Target: Push TRUE ASR above 60%

3. **Paper Strategy**:
   - Emphasize methodology (closed-loop framework) over magnitude
   - Frame as "even moderate single-frame attacks cause significant task failure"

### If TRUE ASR < 40% (Weak Attack)

1. **Deep Investigation** (Week 1):
   - Failure mode analysis
   - Check drift consistency
   - Understand why tasks aren't failing

2. **Pivot Options**:
   - Different attack objective (e.g., maximize action variance)
   - Different attack type (temporal attacks, multi-step)
   - Focus on specific failure modes (gripper-only, etc.)

3. **Alternative Venues**:
   - IROS 2026 (robotics-focused, May deadline)
   - NeurIPS 2026 (ML-focused, May deadline)

---

## FILES CREATED

### Python Scripts
- `code/scripts/evaluate_closed_loop.py` - Main closed-loop evaluation script

### SLURM Scripts
- `code/SLURM/run_closed_loop_eval.sh` - Single/multiple task closed-loop evaluation
- `code/SLURM/run_closed_loop_array.sh` - Parallel array job for all 10 tasks
- `code/SLURM/train_gripper_focused.sh` - Train patches with w_grip=5.0
- `code/SLURM/run_full_validation_pipeline.sh` - Master pipeline (all steps)

### Documentation
- `PRIORITIES_1-3_IMPLEMENTATION_GUIDE.md` - This file

---

## SUMMARY CHECKLIST

Before running experiments, verify:
- [ ] SSH access to cluster working
- [ ] Python environment accessible: `/data1/ma1/envs/upa-vla/bin/python3.10`
- [ ] OpenVLA checkpoint exists: `/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b`
- [ ] LIBERO data exists: `/data1/ma1/Ishaq/ump-vla/data/libero/`
- [ ] Output directory exists: `outputs/se3_zoo_attack/`

After experiments complete:
- [ ] Check TRUE ASR from closed-loop results
- [ ] Check drift consistency from test results
- [ ] Check gripper flip rate from test results
- [ ] Compare baseline vs gripper-focused patches
- [ ] Make decision using decision tree above

**GOOD LUCK!**

This is the moment of truth for Direction 2. The results will tell you definitively whether your hypothesis holds and if you have an ECCV-worthy contribution.

---

*Last Updated: 2026-01-21*
*Claude Code Implementation*
