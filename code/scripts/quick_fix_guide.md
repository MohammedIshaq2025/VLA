# Quick Fix Guide: LIBERO Headless Rendering for SLURM

## ✅ Problem Solved!

The EGL rendering error is **FIXED**. Here's what to do:

## One-Line Fix

**Before importing robosuite/libero in ANY script, add:**

```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
```

## For SLURM Jobs

Add to your SLURM script (before Python commands):

```bash
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
```

## Modified run_libero_eval.py

To make `run_libero_eval.py` work automatically, add this at the **very top** (after the docstring, before other imports):

```python
# === HEADLESS RENDERING SETUP ===
import os
os.environ['MUJOCO_GL'] = 'osmesa'
# === END HEADLESS RENDERING SETUP ===
```

Then the rest of the imports will work fine.

## Verification

```bash
conda activate upa-vla
export MUJOCO_GL=osmesa
python -c "import robosuite; print('✓ Works!')"
```

## That's It!

Once `MUJOCO_GL=osmesa` is set, robosuite will use software rendering and work on headless systems. True ASR evaluation will work correctly.




