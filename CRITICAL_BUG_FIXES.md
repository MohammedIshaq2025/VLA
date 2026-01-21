# CRITICAL BUG FIXES - OpenVLA LIBERO Evaluation

**Date**: 2026-01-22
**Status**: READY FOR TESTING
**Estimated Test Time**: 2-3 hours (10 episodes × 4 tasks)

---

## Summary of Bugs Found

Your OpenVLA closed-loop evaluation had **3 CRITICAL BUGS** that caused 0% success rate:

### Bug 1: Wrong Action Unnormalization Key ❌
**File**: `code/openvla_action_extractor.py:69`
**Problem**: Used `unnorm_key = "bridge_orig"` (Bridge robot statistics)
**Fix**: Changed to `"libero_spatial_no_noops"` for LIBERO spatial tasks

**Why it matters**: OpenVLA predicts actions in normalized [-1, +1] range. The `unnorm_key` determines how these are converted back to real robot delta end-effector poses. Using Bridge statistics for LIBERO/Franka Panda causes **completely wrong action scaling**.

**Source**: [OpenVLA-OFT Implementation](https://github.com/moojink/openvla-oft)

### Bug 2: Wrong Camera Resolution ❌
**File**: `code/scripts/evaluate_closed_loop.py:105-108`
**Problem**: Used 128×128 pixel camera images
**Fix**: Changed to 224×224 pixels (OpenVLA's native resolution)

**Why it matters**: OpenVLA was trained on 224×224 images. Using 128×128 causes:
- Image quality degradation
- Feature extraction mismatch
- Poor policy performance

**Source**: [OpenVLA LIBERO Evaluation](https://github.com/openvla/openvla/issues/264)

### Bug 3: No Suite-Specific Configuration ❌
**Problem**: No mechanism to set correct unnorm_key per LIBERO suite
**Fix**: Added `set_unnorm_key_for_libero()` method

Maps suites to correct keys:
- `libero_spatial` → `"libero_spatial_no_noops"`
- `libero_object` → `"libero_object_no_noops"`
- `libero_goal` → `"libero_goal_no_noops"`
- `libero_10` → `"libero_10_no_noops"`

---

## Files Modified

### 1. `code/openvla_action_extractor.py`

**Changes**:
- Added `unnorm_key` parameter to `__init__()` (line 20-22)
- Changed default from `"bridge_orig"` to configurable (line 55)
- Added `set_unnorm_key_for_libero(suite)` method (lines 140-153)
- Added logging to show which unnorm_key is being used

**Key Addition**:
```python
def set_unnorm_key_for_libero(self, suite: str):
    """Automatically set correct unnorm_key for LIBERO suite."""
    libero_unnorm_keys = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops"
    }
    if suite in libero_unnorm_keys:
        self.unnorm_key = libero_unnorm_keys[suite]
        print(f"[OpenVLA] Unnorm key set to: {self.unnorm_key}")
```

### 2. `code/scripts/evaluate_closed_loop.py`

**Changes**:
- Camera resolution: 128 → 224 (lines 105, 107)
- Added `policy.set_unnorm_key_for_libero(args.suite)` after model loading (line 469)

**Critical Addition** (line 469):
```python
# CRITICAL: Set correct unnorm_key for LIBERO suite
policy.set_unnorm_key_for_libero(args.suite)
```

### 3. `code/SLURM/run_closed_loop_eval.sh`

**Changes**:
- Added `--camera_height 224 --camera_width 224` to evaluation command

### 4. `code/SLURM/run_full_validation_pipeline.sh`

**Changes**:
- Added `--camera_height 224 --camera_width 224` to all 3 evaluation calls:
  - Step 1: Clean baseline (line 137)
  - Step 4a: Baseline patch evaluation (line 302)
  - Step 4b: Gripper patch evaluation (line 324)

### 5. `code/SLURM/test_fixes_tasks_0_3.sh` (NEW)

**Purpose**: Quick validation script to test fixes on tasks 0-3
**Features**:
- Tests clean baseline only (no attack)
- 10 episodes per task (faster validation)
- Tests tasks 0, 1, 2, 3 in sequence
- Auto-extracts success rates
- Estimated runtime: 2-3 hours

---

## Expected Behavior After Fixes

### Before Fixes (Broken)
```
Task 0: 0/50 success (0.0%)
Task 1: 0/50 success (0.0%)
Task 2: 0/50 success (0.0%)
Task 3: 0/50 success (0.0%)
```

### After Fixes (Expected)
According to [OpenVLA-OFT benchmarks](https://openvla-oft.github.io/), fine-tuned models achieve:
- **LIBERO-Spatial**: 98% average success rate
- **LIBERO-Object**: 99% average success rate

Your base OpenVLA-7B (not fine-tuned) should achieve **at least 30-50%** success on easier tasks.

**Minimal success criteria**: >0% on at least 1 task proves fixes worked.

---

## How to Test the Fixes

### Option 1: Quick Test (Recommended First)

Run the new test script on tasks 0-3:

```bash
cd /data1/ma1/Ishaq/ump-vla

# Make script executable
chmod +x code/SLURM/test_fixes_tasks_0_3.sh

# Submit job
sbatch code/SLURM/test_fixes_tasks_0_3.sh

# Monitor
squeue -u $USER
tail -f outputs/se3_zoo_attack/logs/test_fixes_*.out
```

**What to look for**:
- Log should show: `[OpenVLA] Using unnorm_key: libero_spatial_no_noops` ✓
- Camera resolution should be 224x224 ✓
- Success rate should be >0% for at least one task ✓

**Runtime**: ~2-3 hours (10 episodes × 4 tasks × ~3 min/episode)

### Option 2: Full Validation (If Quick Test Works)

If quick test shows >0% success rate:

```bash
# Run full 50-episode validation on task with highest success rate
BEST_TASK=<task_id_with_best_rate>
sbatch code/SLURM/run_closed_loop_eval.sh "" $BEST_TASK
```

---

## Verification Checklist

Before submitting job, verify:

- [ ] All 5 files modified correctly
- [ ] `test_fixes_tasks_0_3.sh` is executable (`chmod +x`)
- [ ] Python path exists: `/data1/ma1/envs/upa-vla/bin/python3.10`
- [ ] OpenVLA checkpoint exists: `/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b`
- [ ] Output directories exist:
  - [ ] `/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/closed_loop`
  - [ ] `/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs`

**During execution**, verify in logs:
- [ ] `[OpenVLA] Using unnorm_key: libero_spatial_no_noops`
- [ ] No errors loading model
- [ ] Episodes running (not all timing out)
- [ ] Success rate increments (even if small)

---

## What If It Still Fails?

If success rate is still 0% after fixes, possible causes:

1. **OpenVLA checkpoint issue**
   - Verify checkpoint is correct OpenVLA-7B model
   - Check if it needs fine-tuning on LIBERO

2. **LIBERO environment setup**
   - Test LIBERO directly: `python3 code/scripts/test_libero_headless.py`
   - Verify OSMesa rendering works

3. **Task-specific difficulty**
   - Some LIBERO tasks are genuinely hard
   - Try different tasks (especially LIBERO-10 suite might be easier)

4. **Action space mismatch**
   - Check action dimension is 7 in logs
   - Verify actions are in valid range

---

## Theoretical Foundation

### Why These Fixes Are Critical

**Action Denormalization**:
OpenVLA outputs actions in normalized space: `a_norm ∈ [-1, +1]^7`

Denormalization formula:
```
a_real = μ + σ × a_norm
```

Where `μ` and `σ` are dataset-specific statistics stored in the model.

Using **wrong unnorm_key** means:
- Bridge `μ_bridge` and `σ_bridge` ≠ LIBERO `μ_libero` and `σ_libero`
- Actions scaled completely incorrectly
- Robot moves in wrong directions/magnitudes
- Task success impossible

**Image Resolution**:
OpenVLA's vision encoder expects 224×224 inputs. Using 128×128:
- Automatic resizing introduces artifacts
- Feature extraction misaligned with training
- Policy performance degraded

---

## References

1. [OpenVLA Paper (arXiv 2406.09246)](https://arxiv.org/abs/2406.09246)
2. [OpenVLA-OFT Fine-Tuning](https://openvla-oft.github.io/)
3. [OpenVLA GitHub Repository](https://github.com/openvla/openvla)
4. [LIBERO Benchmark](https://libero-project.github.io/)
5. [OpenVLA Hugging Face](https://huggingface.co/openvla/openvla-7b)

---

## Contact

If issues persist after testing:
1. Check SLURM logs in `outputs/se3_zoo_attack/logs/`
2. Verify all prerequisites from verification checklist
3. Consider testing with fine-tuned checkpoint: `openvla/openvla-7b-finetuned-libero-spatial`

---

**Status**: READY TO TEST
**Next Action**: Run `sbatch code/SLURM/test_fixes_tasks_0_3.sh`
**Expected Result**: >0% success rate on at least 1 of 4 tasks

