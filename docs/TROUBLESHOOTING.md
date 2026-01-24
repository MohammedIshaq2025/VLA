# Phase 0 Troubleshooting Guide

## Common Issues and Solutions

### 1. Python Environment Issues

#### Issue: Python not found at expected path
```
❌ ERROR: Python not found at /data1/ma1/envs/vla_freq_attack/bin/python3.10
```

**Solution:**
```bash
# Check if environment exists
ls -la /data1/ma1/envs/vla_freq_attack/

# If missing, create environment
conda create -p /data1/ma1/envs/vla_freq_attack python=3.10 -y

# Verify
/data1/ma1/envs/vla_freq_attack/bin/python3.10 --version
```

#### Issue: Wrong Python version being used
```
⚠️ Python version: 3.9.x (expected 3.10.x)
```

**Solution:**
- Always use direct Python path in scripts: `/data1/ma1/envs/vla_freq_attack/bin/python3.10`
- Don't rely on `python` or `python3` commands
- For SLURM jobs, never use `conda activate`

#### Issue: Module not found errors
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Verify you're using the correct Python
which python
# Should output: /data1/ma1/envs/vla_freq_attack/bin/python3.10

# Re-install dependencies
/data1/ma1/envs/vla_freq_attack/bin/pip install -r requirements.txt
```

---

### 2. SLURM Job Issues

#### Issue: Job stays in pending (PD) state forever
```bash
$ squeue -u ma1
JOBID  PARTITION  NAME       USER  ST  TIME  NODES
12345  gpu2       TEST_GPU   ma1   PD  0:00  1
```

**Possible Causes & Solutions:**

1. **No GPUs available**
   ```bash
   # Check GPU availability
   sinfo -p gpu2 -o "%P %C %G"

   # Wait or try different time
   ```

2. **Missing MCS label**
   ```bash
   # Ensure SLURM script has:
   #SBATCH --mcs-label=mcs
   ```

3. **Resource request too large**
   ```bash
   # Reduce memory or time request
   #SBATCH --mem=32G  # Instead of 128G
   #SBATCH --time=00:30:00  # Instead of 10:00:00
   ```

4. **Incorrect partition**
   ```bash
   # Use correct partition
   #SBATCH --partition=gpu2
   ```

#### Issue: Job fails immediately
```bash
$ squeue -u ma1
# Job not shown (already completed/failed)
```

**Solution:**
```bash
# Check job status
sacct -j JOBID

# View error output
cat slurm_jobs/logs/test_*_JOBID.err

# Common fixes:
# - Check SBATCH directives
# - Verify Python path
# - Check file permissions
```

#### Issue: SLURM output file not created
```
cat: slurm_jobs/logs/test_gpu_*.out: No such file or directory
```

**Solution:**
```bash
# Ensure logs directory exists
mkdir -p /data1/ma1/Ishaq/VLA_Frequency_Attack/slurm_jobs/logs

# Check SBATCH output path is correct
grep "^#SBATCH --output" slurm_jobs/test_gpu.sh

# Wait - file is created when job starts, not when submitted
```

---

### 3. GPU and CUDA Issues

#### Issue: CUDA not available in PyTorch
```
CUDA available: False
```

**Possible Causes & Solutions:**

1. **PyTorch installed without CUDA support**
   ```bash
   PYTHON="/data1/ma1/envs/vla_freq_attack/bin/python3.10"
   $PYTHON -c "import torch; print(torch.version.cuda)"

   # If prints "None", reinstall PyTorch
   $PYTHON -m pip uninstall torch torchvision torchaudio -y
   $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **GPU not allocated to job**
   ```bash
   # Check SLURM script has:
   #SBATCH --gres=gpu:nvidia_h200_2g.35gb:1

   # Inside job, check:
   echo $CUDA_VISIBLE_DEVICES
   # Should NOT be empty
   ```

3. **CUDA driver issues**
   ```bash
   # Inside SLURM job:
   nvidia-smi

   # If fails, contact cluster admin
   ```

#### Issue: Out of memory error
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Use FP16 instead of FP32**
   ```python
   model = load_vla(
       "openvla/openvla-7b",
       torch_dtype=torch.float16,  # Not torch.float32
       device_map="auto"
   )
   ```

2. **Reduce batch size**
   ```python
   batch_size = 1  # Use smallest possible
   ```

3. **Clear GPU cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   import gc
   gc.collect()
   ```

4. **Request more memory in SLURM**
   ```bash
   #SBATCH --mem=64G  # Instead of 32G
   ```

---

### 4. Model Download Issues

#### Issue: Model download fails or times out
```
❌ Download failed: Connection timeout
```

**Solutions:**

1. **Use resume capability**
   ```bash
   # Just run again - it will resume
   /data1/ma1/envs/vla_freq_attack/bin/python3.10 scripts/download_model.py
   ```

2. **Use SLURM job for longer timeout**
   ```bash
   sbatch slurm_jobs/download_model.sh
   # Monitors: squeue -u ma1
   ```

3. **Check network connectivity**
   ```bash
   ping huggingface.co
   curl -I https://huggingface.co/
   ```

4. **Check disk space**
   ```bash
   df -h /data1/ma1/Ishaq/
   # Need at least 20 GB free
   ```

#### Issue: Model downloaded but very small
```
Total size: 2.34 GB  # Expected: 15-20 GB
```

**Solution:**
```bash
# Download incomplete, run again
/data1/ma1/envs/vla_freq_attack/bin/python3.10 scripts/download_model.py

# Or delete and redownload
rm -rf /data1/ma1/Ishaq/VLA_Frequency_Attack/cache/hub/models--openvla--openvla-7b/
```

#### Issue: HuggingFace cache fills home directory
```
WARNING: Your home directory is almost full
```

**Solution:**
```bash
# Ensure HF_HOME is set before any Python commands
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache

# Add to SLURM scripts:
# export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache

# Clean old cache if needed
rm -rf ~/.cache/huggingface/
```

---

### 5. LIBERO and MuJoCo Issues

#### Issue: MuJoCo rendering fails
```
❌ ERROR: Failed to initialize OpenGL
```

**Solution:**
```bash
# CRITICAL: Set MUJOCO_GL environment variable
export MUJOCO_GL=osmesa

# In SLURM scripts:
# export MUJOCO_GL=osmesa

# Verify in Python:
python -c "import os; print(os.environ.get('MUJOCO_GL'))"
# Should output: osmesa
```

#### Issue: LIBERO import fails
```
ModuleNotFoundError: No module named 'libero'
```

**Solution:**
```bash
cd /data1/ma1/Ishaq/VLA_Frequency_Attack/LIBERO
/data1/ma1/envs/vla_freq_attack/bin/pip install -e .

# Verify
/data1/ma1/envs/vla_freq_attack/bin/python3.10 -c "import libero; print('OK')"
```

#### Issue: LIBERO task data not found
```
FileNotFoundError: Task definition file not found
```

**Solution:**
```python
# Data is downloaded on first use
from libero.libero import get_libero_path

# This will download data if needed
path = get_libero_path("libero_spatial")

# Or manually download from:
# https://github.com/Lifelong-Robot-Learning/LIBERO/releases
```

---

### 6. Repository and File Issues

#### Issue: Git clone fails
```
fatal: could not create work tree dir 'openvla': Permission denied
```

**Solution:**
```bash
# Check permissions
ls -la /data1/ma1/Ishaq/VLA_Frequency_Attack/

# Ensure you're in the right directory
cd /data1/ma1/Ishaq/VLA_Frequency_Attack

# Try with sudo if needed (unlikely on cluster)
```

#### Issue: DCT file not found
```
❌ src/utils/dct_utils.py (missing)
```

**Solution:**
```bash
# Copy from SSA repository
cp /data1/ma1/Ishaq/VLA_Frequency_Attack/SSA/dct.py \
   /data1/ma1/Ishaq/VLA_Frequency_Attack/src/utils/dct_utils.py

# Verify
ls -la /data1/ma1/Ishaq/VLA_Frequency_Attack/src/utils/dct_utils.py
```

#### Issue: Script not executable
```
bash: ./scripts/master_setup.sh: Permission denied
```

**Solution:**
```bash
chmod +x scripts/*.sh scripts/*.py
chmod +x slurm_jobs/*.sh

# Verify
ls -la scripts/master_setup.sh
# Should show: -rwxr-xr-x
```

---

### 7. Import and Dependency Issues

#### Issue: OpenVLA import fails
```
ModuleNotFoundError: No module named 'openvla'
```

**Solutions:**

1. **Install OpenVLA**
   ```bash
   cd /data1/ma1/Ishaq/VLA_Frequency_Attack/openvla
   /data1/ma1/envs/vla_freq_attack/bin/pip install -e .
   ```

2. **Check installation**
   ```bash
   /data1/ma1/envs/vla_freq_attack/bin/pip list | grep openvla
   ```

3. **Add to Python path (temporary fix)**
   ```python
   import sys
   sys.path.insert(0, '/data1/ma1/Ishaq/VLA_Frequency_Attack/openvla')
   from openvla import load_vla
   ```

#### Issue: Transformers version conflict
```
ERROR: transformers requires torch>=1.10.0
```

**Solution:**
```bash
# Upgrade transformers
/data1/ma1/envs/vla_freq_attack/bin/pip install --upgrade transformers

# Or reinstall specific version
/data1/ma1/envs/vla_freq_attack/bin/pip install transformers==4.35.0
```

---

### 8. Configuration and Environment Variable Issues

#### Issue: HF_HOME not set
```
⚠️ HF_HOME: NOT SET (expected: /data1/ma1/Ishaq/VLA_Frequency_Attack/cache)
```

**Solution:**
```bash
# Set for current session
export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache

# Add to ~/.bashrc for persistence
echo 'export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache' >> ~/.bashrc
source ~/.bashrc

# ALWAYS set in SLURM scripts
# export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache
```

#### Issue: MUJOCO_GL not set
```
⚠️ MUJOCO_GL: NOT SET (should be 'osmesa')
```

**Solution:**
```bash
# Set for current session
export MUJOCO_GL=osmesa

# Add to ~/.bashrc
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
source ~/.bashrc

# CRITICAL: Always set in SLURM scripts
# export MUJOCO_GL=osmesa
```

---

## Getting Help

### 1. Check Logs
```bash
# CPU verification output
cat /data1/ma1/Ishaq/VLA_Frequency_Attack/scripts/verify_cpu_setup.log

# SLURM job outputs
cat /data1/ma1/Ishaq/VLA_Frequency_Attack/slurm_jobs/logs/test_*_JOBID.out
cat /data1/ma1/Ishaq/VLA_Frequency_Attack/slurm_jobs/logs/test_*_JOBID.err

# Setup log
cat /data1/ma1/Ishaq/VLA_Frequency_Attack/setup_log.txt
```

### 2. Verify System State
```bash
# Run verification script
/data1/ma1/envs/vla_freq_attack/bin/python3.10 scripts/verify_cpu_setup.py

# Check disk space
df -h /data1/ma1/Ishaq/

# Check GPU allocation (in SLURM job)
nvidia-smi
```

### 3. Test Individual Components
```bash
# Test Python
/data1/ma1/envs/vla_freq_attack/bin/python3.10 --version

# Test PyTorch (CPU)
/data1/ma1/envs/vla_freq_attack/bin/python3.10 -c "import torch; print(torch.__version__)"

# Test imports
/data1/ma1/envs/vla_freq_attack/bin/python3.10 -c "from openvla import load_vla; import libero"
```

### 4. Contact Support
- **Cluster issues**: Contact Qatar CMU IT support
- **SLURM questions**: `man sbatch`, `man squeue`
- **Research questions**: Review Phase 0 plan document

---

## Quick Reference: Essential Commands

### Environment
```bash
# Activate (interactive only)
conda activate /data1/ma1/envs/vla_freq_attack

# Direct Python (for scripts and SLURM)
PYTHON="/data1/ma1/envs/vla_freq_attack/bin/python3.10"
```

### SLURM
```bash
# Submit job
sbatch slurm_jobs/test_gpu.sh

# Check status
squeue -u ma1

# Cancel job
scancel JOBID

# View output (wait for job to start)
cat slurm_jobs/logs/test_gpu_JOBID.out
```

### Verification
```bash
# CPU verification
$PYTHON scripts/verify_cpu_setup.py

# Model checkpoint
du -sh cache/hub/models--openvla--openvla-7b/

# Test results
grep "TEST PASSED" slurm_jobs/logs/*.out
```

### Debugging
```bash
# Check job details
scontrol show job JOBID

# View job history
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode

# Check GPU allocation
sinfo -p gpu2 -o "%P %C %G"
```

---

## Prevention Tips

1. **Always set environment variables in SLURM scripts**
   - `export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache`
   - `export MUJOCO_GL=osmesa`

2. **Always use direct Python path in SLURM**
   - `/data1/ma1/envs/vla_freq_attack/bin/python3.10`
   - Never use `conda activate`

3. **Always include MCS label in SLURM jobs**
   - `#SBATCH --mcs-label=mcs`

4. **Always verify before proceeding**
   - Run CPU verification before GPU tests
   - Check each test passes before next step

5. **Keep logs**
   - Save all SLURM job outputs
   - Keep setup log files
   - Document any errors encountered

---

**Last Updated**: 2026-01-23
**Version**: Phase 0.1
