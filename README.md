# VLA Frequency Attack

Frequency-domain adversarial attacks on Vision-Language-Action (VLA) models, specifically targeting OpenVLA.

## Project Structure

```
VLA/
├── scripts/                    # Main attack and evaluation scripts
│   ├── frequency_attack.py     # Core attack implementation
│   ├── evaluate_closed_loop.py # Closed-loop evaluation
│   ├── phase0_verify.py        # Environment verification
│   ├── phase1_gradient_flow.py # Gradient flow verification
│   └── phase2_frequency_projection.py # Frequency projection tests
├── experiments/                # Experiment code
│   └── frequency_attack.py     # Attack class (symlinked from scripts)
├── openvla/                    # OpenVLA repository (for libero_utils.py)
├── slurm_jobs/                 # SLURM job scripts
├── src/                        # Source utilities
├── configs/                    # Configuration files
├── docs/                       # Documentation
└── notebooks/                  # Jupyter notebooks
```

## Setup

### Prerequisites (Clone separately)

These large repositories should be cloned separately:

```bash
# LIBERO benchmark
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# SSA (Spectrum Simulation Attack) - reference
git clone https://github.com/yuyang-long/SSA.git

# TransferAttack - reference
git clone https://github.com/Trustworthy-AI-Group/TransferAttack.git
```

### Model Checkpoints

Download OpenVLA checkpoints from HuggingFace:
- `openvla/openvla-7b`
- `openvla/openvla-7b-finetuned-libero-spatial`

## Critical Constants

```python
ACTION_TOKEN_OFFSET = 31808  # 32064 - 256
VOCAB_SIZE = 32064
UNNORM_KEY = "libero_spatial"
```

## Attack Methodology

The attack uses frequency-constrained Carlini-Wagner loss:
- **Low-frequency attack**: `freq_ratio = 0.25` (targets ViT vulnerabilities)
- **High-frequency attack**: `freq_ratio = -0.25` (high-pass filter)

## Clusters

- **CMU Qatar**: Primary cluster for development
- **SPARTAN (UniMelb)**: Secondary cluster for batch experiments

## License

Research use only.
