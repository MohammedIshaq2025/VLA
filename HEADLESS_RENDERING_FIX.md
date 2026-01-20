# Fixing LIBERO Headless Rendering for SLURM/Remote Supercomputers

## Problem
When running LIBERO evaluations on a headless supercomputer via SSH/SLURM, you get:
```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```
This prevents robosuite from initializing, which blocks LIBERO simulation and True ASR (task success/failure) evaluation.

## Solution Overview

LIBERO uses `OffScreenRenderEnv` which is designed for headless rendering, but robosuite (the underlying simulator) needs proper rendering backend configuration. We'll use **OSMesa** (software rendering) which doesn't require GPU drivers or display.

---

## Step 1: Install OSMesa Libraries

### Option A: Using Conda (Recommended)
```bash
conda activate upa-vla
conda install -c conda-forge mesalib -y
```

### Option B: Using System Package Manager (if you have sudo access)
```bash
sudo apt-get update
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev
```

### Option C: If Conda/System install fails, try:
```bash
conda activate upa-vla
pip install --no-deps osmesa
```

---

## Step 2: Configure Environment Variables

### Method 1: Using the Setup Script (Recommended)

Add this to the **top** of your Python scripts (before importing robosuite/libero):

```python
import sys
import os
sys.path.insert(0, 'code/scripts')
from setup_headless_rendering import setup_headless_rendering

# Configure headless rendering BEFORE importing robosuite/libero
setup_headless_rendering('osmesa')  # or 'auto' to try both

# Now safe to import
from libero.libero import benchmark
import robosuite
```

### Method 2: Using Shell Script

Before running Python, source the setup script:
```bash
source code/scripts/setup_headless_rendering.sh
python your_script.py
```

### Method 3: Manual Environment Variables

In your SLURM script or shell:
```bash
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
```

---

## Step 3: Modify LIBERO Evaluation Script

The evaluation script needs to setup rendering before importing. Create a wrapper or modify `run_libero_eval.py`:

**Option A: Create a wrapper script** (`code/scripts/run_libero_eval_wrapper.py`):

```python
#!/usr/bin/env python3
import sys
import os

# Setup headless rendering FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from setup_headless_rendering import setup_headless_rendering
setup_headless_rendering('osmesa')

# Now import and run the actual evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openvla', 'experiments', 'robot', 'libero'))
from run_libero_eval import eval_libero, GenerateConfig
import draccus

if __name__ == "__main__":
    eval_libero()
```

**Option B: Modify `run_libero_eval.py` directly** - Add at the top (after imports but before using robosuite):

```python
# Add after line 29 (after imports)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))
from setup_headless_rendering import setup_headless_rendering
setup_headless_rendering('osmesa')
```

---

## Step 4: Test the Setup

Run the test script to verify everything works:

```bash
conda activate upa-vla
cd /data1/ma1/Ishaq/ump-vla
python code/scripts/test_libero_headless.py
```

Expected output:
```
✓✓✓ ALL TESTS PASSED - LIBERO works in headless mode! ✓✓✓
```

---

## Step 5: Run on SLURM

### Quick Test Job

```bash
# Submit a test job
sbatch code/SLURM/run_libero_eval.sh
```

### Custom SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=libero_eval
#SBATCH --output=logs/libero_eval_%j.out
#SBATCH --error=logs/libero_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # Adjust to your cluster

# Setup environment
source ~/.bashrc
conda activate upa-vla
cd /data1/ma1/Ishaq/ump-vla

# Configure headless rendering
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM

# Run evaluation
python code/openvla/experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint checkpoints/openvla-7b \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 50
```

---

## Troubleshooting

### Issue: OSMesa still fails

1. **Check if mesalib is installed:**
   ```bash
   conda list | grep mesa
   python -c "import osmesa; print('OSMesa available')"
   ```

2. **Try EGL instead** (requires GPU with proper drivers):
   ```bash
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export EGL_PLATFORM=device
   export MUJOCO_EGL_DEVICE_ID=0
   ```

3. **Check GPU availability:**
   ```bash
   nvidia-smi  # Should show GPU info
   ```

### Issue: Import errors persist

Make sure you're setting environment variables **BEFORE** importing:
```python
# ❌ WRONG - imports first
import robosuite
os.environ['MUJOCO_GL'] = 'osmesa'

# ✅ CORRECT - setup first
os.environ['MUJOCO_GL'] = 'osmesa'
import robosuite
```

### Issue: SLURM job fails

1. Check SLURM output logs: `cat logs/libero_eval_*.out`
2. Verify GPU allocation: `squeue -u $USER`
3. Test interactively on a compute node:
   ```bash
   srun --gres=gpu:1 --pty bash
   # Then run test_libero_headless.py
   ```

---

## Files Created

1. **`code/scripts/setup_headless_rendering.py`** - Python module to configure rendering
2. **`code/scripts/setup_headless_rendering.sh`** - Shell script version
3. **`code/scripts/test_libero_headless.py`** - Test script to verify setup
4. **`code/SLURM/run_libero_eval.sh`** - SLURM job script template

---

## Verification Checklist

- [ ] OSMesa libraries installed (`conda list | grep mesa`)
- [ ] Test script passes (`python code/scripts/test_libero_headless.py`)
- [ ] Environment variables set before imports
- [ ] SLURM script includes rendering setup
- [ ] Can create LIBERO environment successfully
- [ ] Can run `env.step()` and get observations
- [ ] Task success/failure (`info['success']`) is computed correctly

---

## Expected Behavior After Fix

Once configured correctly:
- ✅ `import robosuite` works without errors
- ✅ `import libero` works without errors  
- ✅ LIBERO environments can be created
- ✅ Image observations are generated (`agentview_image`, `eye_in_hand_image`)
- ✅ Task success/failure is computed correctly (`info['success']`)
- ✅ True ASR (Action Success Rate) can be measured in simulation

---

## Additional Notes

- **OSMesa** uses software rendering (CPU-based), so it's slower but more compatible
- **EGL** uses GPU rendering (faster) but requires proper GPU drivers
- For True ASR evaluation, you need rendering to work so the simulator can:
  1. Generate image observations
  2. Compute task success criteria (which may depend on visual state)
  3. Determine episode completion (`done` flag)

If rendering doesn't work, the environment may fail to initialize or always return `success=False`.




