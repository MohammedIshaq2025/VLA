# Phase 0 Implementation Package - VLA Frequency Attack Research

**Version**: 1.0
**Date**: 2026-01-23
**Target Cluster**: Deepnet (Qatar CMU) - gpujobs.qatar.cmu.edu

## Package Overview

This package contains all scripts, configurations, and documentation needed to set up Phase 0 (Foundational Setup) for implementing frequency-domain adversarial attacks on Vision-Language-Action models.

## What's Included

### Documentation
- **DEPLOYMENT_GUIDE.md** - Complete deployment instructions
- **EXECUTION_CHECKLIST.md** - Step-by-step progress tracker
- **docs/TROUBLESHOOTING.md** - Common issues and solutions

### Scripts (CPU-based Setup)
- **scripts/master_setup.sh** - Main automated setup script (sections 0.1-0.5)
- **scripts/download_model.py** - Download OpenVLA-7B model
- **scripts/verify_cpu_setup.py** - CPU verification test
- **scripts/generate_jigsaw.py** - Generate synthetic test images

### SLURM Jobs (GPU Testing)
- **slurm_jobs/test_gpu.sh** - GPU verification test
- **slurm_jobs/test_model_loading.sh** - Model loading test
- **slurm_jobs/test_libero.sh** - LIBERO environment test
- **slurm_jobs/download_model.sh** - Model download via SLURM (backup)

### Configuration
- **configs/base_config.py** - Configuration dataclasses for experiments

## Quick Start

### 1. Transfer to Cluster

```bash
# On your local machine
cd "/Users/mmohammedsho/Work/UniMelb Research"

# Create tarball
tar -czf phase0_implementation.tar.gz VLA_Frequency_Attack_Phase0_Implementation/

# Transfer
scp phase0_implementation.tar.gz ma1@gpujobs.qatar.cmu.edu:/data1/ma1/Ishaq/
```

### 2. Extract and Setup

```bash
# On the cluster
ssh ma1@gpujobs.qatar.cmu.edu
cd /data1/ma1/Ishaq
tar -xzf phase0_implementation.tar.gz
cd VLA_Frequency_Attack_Phase0_Implementation

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py slurm_jobs/*.sh
```

### 3. Run Master Setup

```bash
# This automates sections 0.1-0.5 (CPU setup)
./scripts/master_setup.sh 2>&1 | tee setup_log.txt
```

Wait 60-90 minutes for completion.

### 4. Download Model

```bash
cd /data1/ma1/Ishaq/VLA_Frequency_Attack
/data1/ma1/envs/vla_freq_attack/bin/python3.10 scripts/download_model.py
```

Wait 60-120 minutes for ~15-20 GB download.

### 5. Verify CPU Setup

```bash
cd /data1/ma1/Ishaq/VLA_Frequency_Attack
/data1/ma1/envs/vla_freq_attack/bin/python3.10 scripts/verify_cpu_setup.py
```

All tests should show "✓ PASS".

### 6. Submit GPU Tests

```bash
cd /data1/ma1/Ishaq/VLA_Frequency_Attack

# Submit all GPU tests via SLURM
sbatch slurm_jobs/test_gpu.sh
sbatch slurm_jobs/test_model_loading.sh
sbatch slurm_jobs/test_libero.sh

# Monitor
squeue -u ma1

# Check results once complete
cat slurm_jobs/logs/test_*_*.out | grep "TEST PASSED"
```

Should see 3 "PASSED" messages.

### 7. Verify Complete Setup

```bash
# Quick verification
cd /data1/ma1/Ishaq/VLA_Frequency_Attack
grep -r "PASSED" slurm_jobs/logs/

# Should show:
# - GPU TEST PASSED
# - MODEL TESTS PASSED
# - LIBERO TEST PASSED
```

## Phase 0 Completion Criteria

Phase 0 is complete when:

- [x] All repositories cloned (SSA, OpenVLA, LIBERO)
- [x] Conda environment created
- [x] All dependencies installed
- [x] Model checkpoint downloaded (~15-20 GB)
- [x] CPU verification passes
- [x] GPU test passes (via SLURM)
- [x] Model loading test passes (via SLURM)
- [x] LIBERO test passes (via SLURM)

## Time Estimates

| Phase | Duration | Type |
|-------|----------|------|
| Transfer to cluster | 5-10 min | Manual |
| Master setup (CPU) | 60-90 min | Automated |
| Model download | 60-120 min | Automated |
| GPU tests (SLURM) | 30-60 min | Automated |
| **Total** | **3-5 hours** | |

Can parallelize model download with other steps.

## Package Structure

```
VLA_Frequency_Attack_Phase0_Implementation/
├── README.md                    # This file
├── DEPLOYMENT_GUIDE.md          # Detailed deployment instructions
├── EXECUTION_CHECKLIST.md       # Progress tracker
│
├── scripts/                     # CPU-based setup scripts
│   ├── master_setup.sh          # Main setup automation
│   ├── download_model.py        # Model download
│   ├── verify_cpu_setup.py      # CPU verification
│   └── generate_jigsaw.py       # Synthetic data generation
│
├── slurm_jobs/                  # GPU test scripts
│   ├── test_gpu.sh              # GPU verification
│   ├── test_model_loading.sh   # Model loading test
│   ├── test_libero.sh           # LIBERO test
│   └── download_model.sh        # Model download via SLURM
│
├── configs/                     # Configuration system
│   └── base_config.py           # Dataclass configurations
│
└── docs/                        # Additional documentation
    └── TROUBLESHOOTING.md       # Common issues and solutions
```

## Target Cluster Configuration

- **Host**: gpujobs.qatar.cmu.edu
- **User**: ma1
- **Project Root**: `/data1/ma1/Ishaq/VLA_Frequency_Attack`
- **Environment**: `/data1/ma1/envs/vla_freq_attack`
- **GPU**: nvidia_h200_2g.35gb (~35 GB memory)
- **Scheduler**: SLURM
- **Required Label**: `--mcs-label=mcs`

## Important Reminders

### For SLURM Jobs
1. **Always use direct Python path**: `/data1/ma1/envs/vla_freq_attack/bin/python3.10`
2. **Never use `conda activate`** in SLURM scripts
3. **Always include**: `#SBATCH --mcs-label=mcs`
4. **Always set environment variables**:
   - `export HF_HOME=/data1/ma1/Ishaq/VLA_Frequency_Attack/cache`
   - `export MUJOCO_GL=osmesa`

### For GPU Testing
- GPU testing **requires SLURM job submission** - cannot test interactively
- Wait for jobs to complete before checking results
- Check both `.out` and `.err` log files

### For Model Loading
- Always use FP16: `torch_dtype=torch.float16`
- Model is ~14 GB in FP16, fits in 35 GB GPU
- Cache is ~15-20 GB on disk

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for detailed solutions to common issues.

**Quick checks:**
```bash
# Python path
/data1/ma1/envs/vla_freq_attack/bin/python3.10 --version

# Environment variables
echo $HF_HOME
echo $MUJOCO_GL

# SLURM job status
squeue -u ma1

# Test results
cat slurm_jobs/logs/test_*_*.out | grep -E "(PASSED|FAILED)"
```

## Next Steps After Phase 0

Once all Phase 0 tests pass:

1. **Phase 1**: Validation Experiments (Week 1)
   - Encoder frequency sensitivity testing
   - Basic frequency attack implementation
   - Go/no-go decision

2. **Phase 2**: Full Attack Implementation (Week 2-3)
   - SSA-inspired frequency attacks
   - Hybrid methods
   - Data-free variants

3. **Phase 3**: Evaluation (Week 4)
   - LIBERO benchmark evaluation
   - Comparative analysis
   - Documentation

## Support and Contact

- **Cluster issues**: Contact Qatar CMU IT support
- **SLURM questions**: `man sbatch`, `man squeue`, `sinfo -p gpu2`
- **Research questions**: Review full Phase 0 plan document

## Version History

- **v1.0** (2026-01-23): Initial release
  - Complete Phase 0 implementation
  - All scripts and documentation
  - Tested structure and format

## Credits

- **Phase 0 Plan**: Based on "Direction1_FrequencyDomain_DeepDive.md"
- **Implementation**: Claude Code
- **Target**: VLA Frequency Attack Research (Qatar CMU)

---

**Ready to Deploy!**

Follow the steps in `DEPLOYMENT_GUIDE.md` to begin Phase 0 setup.

For detailed progress tracking, use `EXECUTION_CHECKLIST.md`.

For troubleshooting, consult `docs/TROUBLESHOOTING.md`.
