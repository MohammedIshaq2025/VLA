# UMP-VLA Project Progress

## Overview
Zero-Order Optimization (ZOO) adversarial patch attacks on OpenVLA for robot manipulation.

---

## ✅ Phase 1-4: Core Infrastructure (COMPLETE)

| Component | File | Status |
|-----------|------|--------|
| LIBERO Loader | `code/utils/libero_loader.py` | ✅ Verified |
| OpenVLA Extractor | `code/openvla_action_extractor.py` | ✅ Verified |
| SE(3) Distance | `code/utils/se3_distance.py` | ✅ Verified |
| Target Generator | `code/utils/target_generator.py` | ✅ Fixed (deterministic) |
| ZOO Optimizer V1 | `code/attacks/zoo_optimizer.py` | ✅ Fixed |
| ZOO Optimizer V2 | `code/attacks/zoo_optimizer_v2.py` | ✅ New (maximize deviation) |

---

## 🔧 Bug Fixes Applied (2026-01-19)

### Bug 1: Train/Test Split Inconsistency
- **Problem**: Train used random split, test used sequential split → potential overlap
- **Fix**: Both now use `loader.split_episodes()` with same seed
- **Impact**: Results now valid

### Bug 2: Random Adversarial Target
- **Problem**: `np.random.uniform()` noise each query → moving target
- **Fix**: Deterministic perturbation based on action values
- **Impact**: Stable optimization

### Bug 3: Inverted Train/Test Ratio
- **Problem**: 30% train, 70% test (inverted from standard ML)
- **Fix**: Changed to 70% train (35 episodes), 30% test (15 episodes)
- **Impact**: More training data

---

## 📊 Experimental Results (2026-01-19)

### Summary Table

| Exp | Changes | Queries | Train | ASR | Gripper Flip | Patch Effect |
|-----|---------|---------|-------|-----|--------------|--------------|
| 1 | Baseline 70/30 | 200 | 35 eps | 46.7% | **0.0%** | 0.547 |
| 2 | All 10 tasks | 200 | 35 eps | 38.2% avg | **0.07%** | 0.453 |
| 3 | More queries | 500 | 35 eps | 46.7% | **0.0%** | 0.547 |
| 4 | Max deviation obj | 200 | 35 eps | 47.3% | **0.0%** | 0.553 |

### Experiment Details

#### Exp 1: Baseline (70/30 Split)
- **Goal**: Establish baseline with proper train/test split
- **Result**: ASR 46.7%, 0% gripper flip
- **Shortcoming**: Patch changes predictions but never flips gripper

#### Exp 2: All 10 LIBERO Spatial Tasks
- **Goal**: Test generalization across tasks
- **Result**: ASR ranges 30-47% across tasks, avg 38.2%
- **Shortcoming**: Consistent 0% gripper flip across ALL tasks

#### Exp 3: Increased Queries (500)
- **Goal**: Test if more optimization helps
- **Result**: **Identical** to 200 queries (46.7% ASR)
- **Shortcoming**: More queries ≠ better attack. Optimization is saturated.

#### Exp 4: Maximize Deviation Objective (V2)
- **Goal**: Change from "minimize distance to target" to "maximize deviation from clean"
- **Result**: +0.6% ASR improvement (46.7% → 47.3%)
- **Shortcoming**: Marginal improvement. Gripper still 0%.

---

## ❌ Critical Issue: 0% Gripper Flip Rate

**All experiments show 0% gripper flip despite ~47% ASR.**

### Analysis
1. **ASR is misleading**: Threshold 0.5 SE3 distance is easily triggered by noise
2. **Model baseline error**: OpenVLA already has 2.0 SE3 error without patch
3. **Gripper is robust**: Strongly tied to instruction ("pick up" → close gripper)
4. **Patch too small**: 32×32 is 6.25% of 128×128 image

### Why Gripper Won't Flip
- Gripper prediction is binary (-1 or +1)
- Avg gripper change is 0.47, but need >1.0 for flip
- Small patches don't affect semantic understanding
- Vision encoder likely filters out local perturbations

---

## 📁 Output Files

```
outputs/se3_zoo_attack/
├── experiments/
│   ├── exp1_baseline_70_30/      # Baseline results
│   ├── exp2_all_tasks/           # 10 task results
│   ├── exp3_500_queries/         # High query results
│   └── exp4_maximize_deviation/  # V2 objective results
├── patches/                       # Trained .npy patches
├── results/                       # JSON metrics
└── logs/                          # SLURM logs
```

---

## 🔬 Next Steps (Research Directions)

### Validated Hypotheses Needed
1. **Patch size**: Does 64×64 improve ASR?
2. **Patch position**: Does position (9 locations) affect ASR?
3. **Coordinate-wise ZOO**: Is gradient estimation too noisy?

### Promising Research Directions
1. **Attention-guided placement** - Attack where model "looks"
2. **Block-coordinate descent** - More efficient gradient estimation
3. **Semantic patches** - Meaningful visual content, not noise
4. **Temporal attacks** - Attack consecutive frames

---

## Environment

```bash
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"
export MUJOCO_GL=osmesa

# Run experiments
sbatch code/SLURM/run_experiment.sh --exp_name NAME --queries 200
sbatch code/SLURM/run_experiment_v2.sh exp_name queries gripper_weight
```

---

**Last Updated**: 2026-01-19 13:30

**Status Summary**:
- ✅ Infrastructure complete and verified
- ✅ Bug fixes applied (splits, targets, objectives)
- ✅ 4 experiments completed with comprehensive logging
- ❌ **Critical**: 0% gripper flip across all experiments
- ⚠️ ASR ~47% but may be measuring noise, not attack success
- 🔬 Need fundamental approach change for ECCV-level results
