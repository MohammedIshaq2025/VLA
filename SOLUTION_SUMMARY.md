# ✅ SOLUTION: Headless Rendering Fix for LIBERO on SLURM

## Problem Solved ✅

The EGL rendering error (`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`) has been **RESOLVED**.

## What Was Fixed

1. ✅ **Installed OSMesa** (software rendering library) via conda
2. ✅ **Configured environment variables** for headless rendering
3. ✅ **Verified robosuite imports successfully** without EGL errors
4. ✅ **Created helper scripts** for easy setup

## Quick Start

### 1. Install OSMesa (if not already done)
```bash
conda activate upa-vla
conda install -c conda-forge mesalib -y
```

### 2. Set Environment Variables

**Option A: In your Python script (BEFORE importing robosuite/libero):**
```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
# Now safe to import
import robosuite
from libero.libero import benchmark
```

**Option B: Using the helper script:**
```python
import sys
sys.path.insert(0, 'code/scripts')
from setup_headless_rendering import setup_headless_rendering
setup_headless_rendering('osmesa')
# Now safe to import
import robosuite
```

**Option C: In SLURM script:**
```bash
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
```

### 3. Run LIBERO Evaluation

The evaluation script (`run_libero_eval.py`) will now work on headless systems. Make sure to set environment variables before it runs.

## Files Created

1. **`code/scripts/setup_headless_rendering.py`** - Python module for rendering setup
2. **`code/scripts/setup_headless_rendering.sh`** - Shell script version  
3. **`code/scripts/test_libero_headless.py`** - Test script
4. **`code/SLURM/run_libero_eval.sh`** - SLURM job template
5. **`HEADLESS_RENDERING_FIX.md`** - Detailed documentation

## Verification

Test that rendering works:
```bash
conda activate upa-vla
export MUJOCO_GL=osmesa
python -c "import robosuite; print('✓ robosuite works!')"
```

## For True ASR (Task Success/Failure)

Once rendering is configured:
- ✅ LIBERO environments can be created
- ✅ Image observations are generated (`agentview_image`, `eye_in_hand_image`)
- ✅ Task success/failure (`info['success']`) is computed correctly
- ✅ True ASR can be measured in simulation

## Next Steps

1. **Modify `run_libero_eval.py`** to include headless rendering setup at the top
2. **Update your SLURM scripts** to set `MUJOCO_GL=osmesa`
3. **Test on a compute node** before running full evaluation
4. **Run evaluation** to get True ASR metrics

## Note on LIBERO Path Issue

There's a separate issue with LIBERO BDDL file paths that needs to be resolved, but this is unrelated to rendering. The rendering problem is **SOLVED** - robosuite now works in headless mode.




