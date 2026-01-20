# SLURM Job Submission Guide

**Complete guide for creating and running SLURM batch scripts in this SSH environment**

---

## Table of Contents

1. [Overview](#1-overview)
2. [SLURM Script Structure](#2-slurm-script-structure)
3. [Essential SBATCH Directives](#3-essential-sbatch-directives)
4. [Environment Setup](#4-environment-setup)
5. [Creating a New SLURM Script](#5-creating-a-new-slurm-script)
6. [Submitting and Managing Jobs](#6-submitting-and-managing-jobs)
7. [Common Patterns and Best Practices](#7-common-patterns-and-best-practices)
8. [Troubleshooting](#8-troubleshooting)
9. [Complete Examples](#9-complete-examples)

---

## 1. Overview

SLURM (Simple Linux Utility for Resource Management) is the job scheduler used in this cluster environment. This guide covers everything needed to create, submit, and manage SLURM jobs for this project.

### Key Concepts

- **Partition**: A group of compute nodes (e.g., `gpu2`)
- **Job**: A batch script that requests resources and runs commands
- **Node**: A physical machine in the cluster
- **GPU**: Graphics processing unit (e.g., `nvidia_h200_2g.35gb`)

### Project-Specific Information

- **Partition**: `gpu2` (2-day time limit)
- **Python Environment**: `/data1/ma1/envs/upa-vla/bin/python3.10`
- **Project Root**: `/data1/ma1/Ishaq/ump-vla`
- **Output Directory**: `/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/`
- **Cache Directory**: `/data1/ma1/Ishaq/ump-vla/cache`

---

## 2. SLURM Script Structure

Every SLURM script follows this structure:

```bash
#!/bin/bash
#SBATCH --directive1=value1
#SBATCH --directive2=value2
# ... more directives ...

# === Environment Setup ===
export VARIABLE=value

# === Script Logic ===
# Your commands here

# === Exit ===
exit $EXIT_CODE
```

**Important**: All `#SBATCH` directives must be at the top of the file, before any executable commands (except the shebang `#!/bin/bash`).

---

## 3. Essential SBATCH Directives

### 3.1 Required Directives

These directives are **essential** for every job:

```bash
#SBATCH --job-name=JOB_NAME          # Human-readable job name
#SBATCH --partition=gpu2              # Partition to use
#SBATCH --ntasks=1                    # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1  # GPU type and count
#SBATCH --time=HH:MM:SS               # Time limit (format: HH:MM:SS)
#SBATCH --mem=64G                     # Memory per node
#SBATCH --mcs-label=mcs               # Security label (required)
```

### 3.2 Output and Logging

```bash
#SBATCH --output=/path/to/output_%j.out  # Standard output (%j = job ID)
#SBATCH --error=/path/to/error_%j.err    # Standard error
```

**Placeholders**:
- `%j` = Job ID (automatically replaced by SLURM)
- `%A` = Array job ID
- `%a` = Array task ID

### 3.3 Email Notifications

```bash
#SBATCH --mail-type=ALL              # When to send email (ALL, BEGIN, END, FAIL, NONE)
#SBATCH --mail-user=your@email.com   # Email address
```

**Mail types**:
- `ALL`: All events (begin, end, fail)
- `BEGIN,END,FAIL`: Specific events
- `NONE`: No emails

### 3.4 Time Limits

**Format**: `HH:MM:SS` or `DD-HH:MM:SS`

Examples:
- `--time=02:00:00` = 2 hours
- `--time=04:00:00` = 4 hours
- `--time=2-00:00:00` = 2 days

**Partition limits**: Check with `sinfo -p gpu2` to see maximum time limits.

---

## 4. Environment Setup

### 4.1 Required Environment Variables

For this project, you **must** set these environment variables:

```bash
# Headless rendering for MuJoCo/LIBERO
export MUJOCO_GL=osmesa

# HuggingFace cache directory
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# Optional: Explicitly set CUDA device (usually handled by SLURM)
export CUDA_VISIBLE_DEVICES=0
```

### 4.2 Python Path

**Always use the direct Python path** (do not rely on `conda activate`):

```bash
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
```

**Why?** SLURM jobs don't automatically activate conda environments. Using the direct path is more reliable.

### 4.3 Working Directory

Always change to the project root:

```bash
cd /data1/ma1/Ishaq/ump-vla
```

This ensures relative paths in your Python scripts work correctly.

---

## 5. Creating a New SLURM Script

### 5.1 Step-by-Step Template

Here's a complete template you can copy and modify:

```bash
#!/bin/bash
#SBATCH --job-name=YOUR_JOB_NAME
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/job_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/job_%j.err

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Job Info ===
echo "=========================================="
echo "Job: YOUR_JOB_NAME"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Your Script Logic Here ===
# Example: Run a Python script
$PYTHON code/scripts/your_script.py \
    --arg1 value1 \
    --arg2 value2

EXIT_CODE=$?

# === Final Status ===
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ JOB COMPLETED SUCCESSFULLY"
else
    echo "❌ JOB FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
```

### 5.2 Parameter Handling

You can pass parameters to your script in several ways:

#### Method 1: Command-line arguments (when submitting)

```bash
# In your SLURM script:
ARG1=${1:-default_value1}
ARG2=${2:-default_value2}

# Submit with:
# sbatch your_script.sh value1 value2
```

#### Method 2: Environment variables

```bash
# In your SLURM script:
SUITE=${SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
QUERIES=${QUERIES:-200}

# Submit with:
# SUITE=libero_object TASK_ID=1 sbatch your_script.sh
```

#### Method 3: Default values in script

```bash
# Hardcoded defaults
SUITE="libero_spatial"
TASK_ID=0
QUERIES=200
```

**Best practice**: Use environment variables with defaults for flexibility.

---

## 6. Submitting and Managing Jobs

### 6.1 Submitting a Job

```bash
# Basic submission
sbatch code/SLURM/your_script.sh

# With command-line arguments
sbatch code/SLURM/your_script.sh arg1 arg2

# With environment variables
SUITE=libero_object TASK_ID=1 sbatch code/SLURM/your_script.sh

# Multiple environment variables
SUITE=libero_spatial TASK_ID=0 QUERIES=500 sbatch code/SLURM/your_script.sh
```

### 6.2 Checking Job Status

```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j JOB_ID

# Detailed job information
scontrol show job JOB_ID

# View all jobs in partition
squeue -p gpu2
```

### 6.3 Job States

- **PENDING (PD)**: Waiting for resources
- **RUNNING (R)**: Currently executing
- **COMPLETED (CD)**: Finished successfully
- **FAILED (F)**: Failed
- **CANCELLED (CA)**: Cancelled by user or system
- **TIMEOUT (TO)**: Exceeded time limit

### 6.4 Cancelling Jobs

```bash
# Cancel a specific job
scancel JOB_ID

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs
scancel -t PENDING -u $USER
```

### 6.5 Viewing Job Output

```bash
# View output file (replace JOB_ID with actual ID)
cat /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/job_JOB_ID.out

# Follow output in real-time (if job is running)
tail -f /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/job_JOB_ID.out

# View error file
cat /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/job_JOB_ID.err
```

### 6.6 Job Dependencies

Run job B only after job A completes:

```bash
# Submit job A
JOB_A=$(sbatch job_A.sh | grep -oP '\d+')

# Submit job B that depends on job A
sbatch --dependency=afterok:$JOB_A job_B.sh

# Multiple dependencies
sbatch --dependency=afterok:$JOB_A:$JOB_B job_C.sh
```

**Dependency types**:
- `afterok:JOB_ID`: Start only if job completes successfully
- `afterany:JOB_ID`: Start after job finishes (success or failure)
- `afternotok:JOB_ID`: Start only if job fails

---

## 7. Common Patterns and Best Practices

### 7.1 Logging and Debugging

Always include:
- Job start/end times
- Environment information
- Parameter values
- Exit codes

```bash
echo "=========================================="
echo "Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Parameters: SUITE=$SUITE, TASK_ID=$TASK_ID"
echo "=========================================="
```

### 7.2 Error Handling

Always check:
- Python executable exists
- Required files exist
- Exit codes from commands

```bash
# Check Python
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found"
    exit 1
fi

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Command failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
```

### 7.3 Creating Output Directories

Always create output directories if they don't exist:

```bash
OUTPUT_DIR="/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs"
mkdir -p "$OUTPUT_DIR"
```

### 7.4 GPU Verification

Check GPU availability:

```bash
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi
```

### 7.5 Time Estimation

Estimate job time based on:
- Training scripts: 2-4 hours (200 queries)
- Testing scripts: 30 minutes - 1 hour
- Quick tests: 5-15 minutes

Always add buffer time (e.g., if you expect 1 hour, request 2 hours).

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: Job stays in PENDING state

**Causes**:
- No available resources
- Time limit too long
- Memory request too high

**Solutions**:
```bash
# Check partition status
sinfo -p gpu2

# Check why job is pending
scontrol show job JOB_ID

# Reduce resource requests if possible
```

#### Issue: Job fails immediately

**Check**:
1. Python path is correct
2. Environment variables are set
3. Output directory exists
4. Script has execute permissions: `chmod +x script.sh`

#### Issue: "Python not found"

**Solution**: Verify Python path:
```bash
ls -la /data1/ma1/envs/upa-vla/bin/python3.10
```

#### Issue: CUDA out of memory

**Solutions**:
- Reduce batch size in Python script
- Request more GPU memory (if available)
- Use gradient checkpointing

#### Issue: "Permission denied"

**Solution**: Make script executable:
```bash
chmod +x code/SLURM/your_script.sh
```

### 8.2 Debugging Tips

1. **Test locally first** (if possible):
   ```bash
   # Run script directly (not through SLURM)
   bash code/SLURM/your_script.sh
   ```

2. **Add debug output**:
   ```bash
   set -x  # Print each command before executing
   set +x  # Stop printing
   ```

3. **Check environment**:
   ```bash
   echo "PATH: $PATH"
   echo "PYTHONPATH: $PYTHONPATH"
   which python3
   ```

4. **Verify file paths**:
   ```bash
   ls -la /path/to/file
   ```

### 8.3 Reading Logs

**Output file** (`*.out`): Contains standard output (print statements, normal output)

**Error file** (`*.err`): Contains standard error (error messages, exceptions)

**Best practice**: Check both files when debugging:
```bash
# View last 50 lines of output
tail -n 50 job_12345.out

# View last 50 lines of error
tail -n 50 job_12345.err

# Search for errors
grep -i error job_12345.out job_12345.err
```

---

## 9. Complete Examples

### 9.1 Training Script Example

```bash
#!/bin/bash
#SBATCH --job-name=TRAIN_PATCH
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/train_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/train_%j.err

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Job Info ===
echo "=========================================="
echo "SE(3) ZOO Adversarial Patch Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === GPU Info ===
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Default parameters (can be overridden via environment variables) ===
SUITE=${SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
QUERIES=${QUERIES:-200}
PATCH_SIZE=${PATCH_SIZE:-32}
LR=${LR:-0.01}
SEED=${SEED:-42}

echo ""
echo "Training Parameters:"
echo "  Suite: $SUITE"
echo "  Task ID: $TASK_ID"
echo "  Queries: $QUERIES"
echo "  Patch Size: $PATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Seed: $SEED"
echo "=========================================="
echo ""

# === Run Training ===
$PYTHON code/scripts/train_patch.py \
    --suite $SUITE \
    --task_id $TASK_ID \
    --queries $QUERIES \
    --patch_size $PATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ TRAINING FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
```

**Usage**:
```bash
# Default parameters
sbatch code/SLURM/train_patch.sh

# Custom parameters
SUITE=libero_object TASK_ID=1 QUERIES=500 sbatch code/SLURM/train_patch.sh
```

### 9.2 Testing Script Example

```bash
#!/bin/bash
#SBATCH --job-name=TEST_PATCH
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/test_%j.err

# === Environment Setup ===
export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

# === Direct Python path ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Job Info ===
echo "=========================================="
echo "SE(3) ZOO Adversarial Patch Testing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Check patch path argument ===
if [ -z "$1" ]; then
    echo "❌ ERROR: Patch path not provided"
    echo "Usage: sbatch test_patch.sh /path/to/patch.npy [suite] [task_id]"
    exit 1
fi

PATCH_PATH=$1
SUITE=${2:-libero_spatial}
TASK_ID=${3:-0}

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

echo ""
echo "Testing Parameters:"
echo "  Patch Path: $PATCH_PATH"
echo "  Suite: $SUITE"
echo "  Task ID: $TASK_ID"
echo "=========================================="
echo ""

# === Run Testing ===
$PYTHON code/scripts/test_patch.py \
    --patch_path $PATCH_PATH \
    --suite $SUITE \
    --task_id $TASK_ID \
    --output_dir /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTING COMPLETED SUCCESSFULLY"
else
    echo "❌ TESTING FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
```

**Usage**:
```bash
# Find patch file first
PATCH=$(ls -t /data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/patches/*_patch.npy | head -1)

# Submit test job
sbatch code/SLURM/test_patch.sh $PATCH libero_spatial 0
```

### 9.3 Quick Test Script Example

```bash
#!/bin/bash
#SBATCH --job-name=QUICK_TEST
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --mcs-label=mcs
#SBATCH --output=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/quick_test_%j.out
#SBATCH --error=/data1/ma1/Ishaq/ump-vla/outputs/se3_zoo_attack/logs/quick_test_%j.err

export MUJOCO_GL=osmesa
export HF_HOME=/data1/ma1/Ishaq/ump-vla/cache

PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

cd /data1/ma1/Ishaq/ump-vla

echo "Quick test starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run quick test
$PYTHON code/scripts/your_quick_test.py

EXIT_CODE=$?
echo "Test completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
```

---

## 10. Quick Reference

### 10.1 Essential Commands

```bash
# Submit job
sbatch script.sh

# Check job status
squeue -u $USER

# Cancel job
scancel JOB_ID

# View job details
scontrol show job JOB_ID

# View output
tail -f /path/to/job_12345.out

# Check partition info
sinfo -p gpu2
```

### 10.2 Common Resource Requests

| Job Type | Time | Memory | CPUs | GPU |
|----------|------|--------|------|-----|
| Quick test | 15 min | 32G | 2 | 1 |
| Training | 2-4 hours | 64G | 4 | 1 |
| Testing | 30-60 min | 64G | 4 | 1 |
| Long training | 1 day | 64G | 4 | 1 |

### 10.3 File Locations

- **SLURM scripts**: `code/SLURM/`
- **Output logs**: `outputs/se3_zoo_attack/logs/`
- **Python scripts**: `code/scripts/`
- **Trained patches**: `outputs/se3_zoo_attack/patches/`
- **Results**: `outputs/se3_zoo_attack/results/`

---

## 11. Advanced Topics

### 11.1 Job Arrays

Run the same job with different parameters:

```bash
#!/bin/bash
#SBATCH --job-name=ARRAY_JOB
#SBATCH --array=0-9          # Run 10 jobs (0 to 9)
#SBATCH --partition=gpu2
# ... other directives ...

TASK_ID=$SLURM_ARRAY_TASK_ID  # Gets 0, 1, 2, ..., 9

$PYTHON code/scripts/train_patch.py --task_id $TASK_ID
```

**Submit**:
```bash
sbatch array_job.sh
```

### 11.2 Interactive Jobs

For debugging, request an interactive session:

```bash
# Request interactive GPU session
srun --partition=gpu2 --gres=gpu:nvidia_h200_2g.35gb:1 --time=01:00:00 --pty bash

# Then run your commands interactively
export MUJOCO_GL=osmesa
cd /data1/ma1/Ishaq/ump-vla
python code/scripts/your_script.py
```

### 11.3 Resource Monitoring

Monitor resource usage during job execution:

```bash
# In your SLURM script, add:
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,MaxVMSize,CPUUtil
```

---

## 12. Checklist for New SLURM Scripts

Before submitting a new script, verify:

- [ ] All `#SBATCH` directives are at the top
- [ ] `--mcs-label=mcs` is included
- [ ] `MUJOCO_GL=osmesa` is exported
- [ ] `HF_HOME` is set correctly
- [ ] Python path is correct: `/data1/ma1/envs/upa-vla/bin/python3.10`
- [ ] Script changes to project root: `cd /data1/ma1/Ishaq/ump-vla`
- [ ] Output/error paths exist (or use `mkdir -p`)
- [ ] Python executable is verified before use
- [ ] Exit codes are checked and returned
- [ ] Time limit is appropriate (add buffer)
- [ ] Script has execute permissions: `chmod +x script.sh`

---

## 13. Additional Resources

### 13.1 SLURM Documentation

- Official SLURM documentation: https://slurm.schedmd.com/
- Quick start: https://slurm.schedmd.com/quickstart.html
- sbatch options: `man sbatch`

### 13.2 Useful SLURM Commands

```bash
# View all partitions
sinfo

# View your account limits
sacctmgr show user $USER

# View job efficiency (after completion)
seff JOB_ID

# View detailed job accounting
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

---

## 14. Summary

**Key Takeaways**:

1. **Always include** `--mcs-label=mcs` and environment variables (`MUJOCO_GL=osmesa`, `HF_HOME`)
2. **Use direct Python path** instead of conda activate
3. **Change to project root** before running scripts
4. **Verify Python exists** before using it
5. **Check exit codes** and handle errors gracefully
6. **Use appropriate time limits** (add buffer time)
7. **Create output directories** if they don't exist
8. **Include logging** for debugging

**Common Workflow**:

1. Create/edit SLURM script in `code/SLURM/`
2. Make executable: `chmod +x code/SLURM/script.sh`
3. Test locally (optional): `bash code/SLURM/script.sh`
4. Submit: `sbatch code/SLURM/script.sh` (with optional args/env vars)
5. Monitor: `squeue -u $USER`
6. Check results: `tail -f outputs/se3_zoo_attack/logs/job_*.out`

---

*Document Version: 1.0*  
*Last Updated: 2026-01-20*  
*For: UMP-VLA Project - SLURM Environment*



