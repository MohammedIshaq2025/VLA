# Comprehensive Literature Review: Adversarial Attacks on Vision-Language-Action Models

## Document Information
- **Date**: January 2026
- **Scope**: Peer-reviewed and preprint papers on adversarial attacks targeting VLA models
- **Focus**: Attack methodologies, threat models, benchmarks, and identified gaps

---

## Table of Contents
1. [Overview of VLA Architecture](#1-overview-of-vla-architecture)
2. [Taxonomy of Existing VLA Attacks](#2-taxonomy-of-existing-vla-attacks)
3. [Detailed Paper Analysis](#3-detailed-paper-analysis)
4. [Benchmarks and Evaluation Protocols](#4-benchmarks-and-evaluation-protocols)
5. [Summary of Attack Landscape](#5-summary-of-attack-landscape)
6. [Identified Gaps in Literature](#6-identified-gaps-in-literature)

---

## 1. Overview of VLA Architecture

### 1.1 Standard VLA Pipeline

Vision-Language-Action models follow a unified architecture:

```
Input Image (I) --> Visual Encoder --> Projector --> LLM Backbone --> Action Decoder --> Action (a)
       +
Language Instruction (s) --> Tokenizer ----^
```

### 1.2 OpenVLA Architecture (Reference Model)

OpenVLA, the most commonly attacked VLA model, consists of:

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| Visual Encoder | SigLIP + DINOv2 (fused) | ~400M |
| Projector | 2-layer MLP | ~50M |
| LLM Backbone | Llama 2 7B | 7B |
| Action Decoder | 256-bin discretization per DoF | - |

**Key Properties**:
- Input resolution: 224x224 or 384x384 RGB
- Action space: 7-DoF (position delta, rotation delta, gripper state)
- Each DoF discretized into 256 bins
- Control frequency: 5-10 Hz recommended

**Source**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.

### 1.3 Other VLA Models in Literature

| Model | Visual Encoder | LLM | Action Head | Reference |
|-------|---------------|-----|-------------|-----------|
| OpenVLA | SigLIP+DINOv2 | Llama 2 7B | Discretized | Kim et al., 2024 |
| RT-2 | ViT | PaLM-E | Discretized | Brohan et al., 2023 |
| Octo | ViT | Transformer | Diffusion | Team et al., 2024 |
| pi0 | ViT | Transformer | Flow Matching | Physical Intelligence, 2024 |
| SpatialVLA | SigLIP | Qwen2 | Discretized | Qu et al., 2025 |

---

## 2. Taxonomy of Existing VLA Attacks

### 2.1 Attack Modality Classification

```
VLA Adversarial Attacks
|
+-- Vision-Based Attacks
|   +-- Pixel-level perturbations (PGD, FGSM adaptations)
|   +-- Patch-based attacks (UADA, UPA, TMA, EDPA)
|   +-- Universal patches (UPA-RFAS)
|   +-- Action-freezing (FreezeVLA)
|   +-- Attention-guided (ADVLA)
|
+-- Language-Based Attacks
|   +-- Jailbreak attacks (RobotGCG)
|   +-- Prompt injection
|
+-- Multi-Modal Attacks
|   +-- Cross-modal misalignment (VLA-Fool)
|   +-- Bi-modal triggers (BadVLA, BackdoorVLA)
|
+-- Backdoor Attacks
    +-- Visual triggers (TabVLA-V)
    +-- Textual triggers (TabVLA-T)
    +-- Bi-modal triggers (BackdoorVLA)
    +-- Action chunking exploitation (SilentDrift)
```

### 2.2 Threat Model Classification

| Threat Model | Access Level | Data Required | Example Attacks |
|--------------|--------------|---------------|-----------------|
| White-box | Full model access | Training data | UADA, UPA, TMA |
| Gray-box | Surrogate model | Some data | UPA-RFAS |
| Black-box | Query access only | None | FreezeVLA (transfer) |
| Backdoor | Training pipeline | Poisoning data | BadVLA, BackdoorVLA |

---

## 3. Detailed Paper Analysis

### 3.1 Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics

**Citation**: Wang et al., ICCV 2025, arXiv:2411.13587

**Venue**: International Conference on Computer Vision (ICCV) 2025

**Contribution**: First systematic study of adversarial vulnerabilities in VLA models

**Attack Methods**:

1. **Untargeted Action Discrepancy Attack (UADA)**
   - Objective: Maximize deviation from correct action
   - Formulation: max L(a*, f(I+delta, s)) s.t. ||delta||_inf <= epsilon
   - Target: All 7 DoF simultaneously

2. **Untargeted Position-aware Attack (UPA)**
   - Objective: Destabilize position-related actions
   - Leverages spatial foundation models for guidance
   - Focuses on end-effector position deviation

3. **Targeted Manipulation Attack (TMA)**
   - Objective: Force specific trajectory
   - Guides robot to attacker-specified positions

4. **Adversarial Patch Generation**
   - Approach: Optimize small colorful patch placed in camera view
   - Physical realizability: Tested in both simulation and real-world

**Evaluation**:
- Benchmark: LIBERO (Object, Spatial, Goal, Long-horizon)
- Models: OpenVLA
- Results: Up to 100% task failure rate

**Key Findings**:
- Generated patches visually resemble robotic arm joints
- Suggests "learned representation bias" in VLA training
- Digital attack success rate: 85.9%
- Physical attack success rate: >43%

**Limitations**:
- Only tested on OpenVLA
- Patch-based attacks require physical placement
- No cross-model transferability analysis

---

### 3.2 Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models

**Citation**: Anonymous, arXiv:2510.13237, October 2025

**Contribution**: First model-agnostic attack (EDPA) and defense for VLA

**Attack Method: EDPA (Embedding Disruption Patch Attack)**

**Key Innovation**:
- Operates in embedding space, not pixel space
- Does not require knowledge of VLA backbone architecture
- Agnostic to robotic manipulator type

**Mathematical Formulation**:
```
L_EDPA = L_semantic_disruption + lambda * L_representation_discrepancy

L_semantic_disruption: Disrupts visual-textual alignment in latent space
L_representation_discrepancy: Maximizes distance between clean and adversarial representations
```

**Evaluation**:
- Benchmark: LIBERO
- Models: OpenVLA, OpenVLA-OFT, pi0
- Failure rate increase: ~62% (OpenVLA-OFT), ~31% (others)

**Defense Mechanism**:
- Adversarial fine-tuning of visual encoder
- Objective: Produce similar latent representations for clean and perturbed inputs
- Effectiveness: Reduces attack success significantly

**Limitations**:
- Still requires optimization on task-relevant images
- Defense requires additional training
- Limited to patch-based perturbations

---

### 3.3 When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models

**Citation**: Lu et al., arXiv:2511.21192, November 2025

**Contribution**: First universal and transferable patch attack on VLA (UPA-RFAS)

**Attack Method: UPA-RFAS**

**Framework Components**:

1. **Feature-space Objective**
   - L1 deviation prior
   - Repulsive InfoNCE contrastive loss
   - Induces transferable representation shifts

2. **Two-phase Min-Max Optimization**
   - Inner loop: Learn invisible sample-wise perturbations
   - Outer loop: Optimize universal patch against hardened neighborhood

3. **VLA-specific Losses**
   - Patch Attention Dominance (L_PAD): Hijacks text-to-vision attention
   - Patch Semantic Misalignment (L_PSM): Induces image-text mismatch

**Overall Objective**:
```
J_out = L_1 + lambda_con * L_con + lambda_PAD * L_PAD + lambda_PSM * L_PSM
```

**Evaluation**:
- Training data: BridgeData V2 (60,096 trajectories), LIBERO
- Surrogate: OpenVLA-7B
- Target models: Multiple VLA architectures
- Results: Consistent transfer across models, tasks, viewpoints

**Key Innovation**:
- Single patch works across multiple VLA models
- Location-agnostic (works regardless of patch placement)
- Sim-to-real transfer demonstrated

**Limitations**:
- Requires training data from robotic domain
- Patch-based (localized perturbation)
- Dependent on surrogate model quality

---

### 3.4 AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models

**Citation**: Anonymous, arXiv:2511.12149, November 2025

**Contribution**: First unified evaluation framework for VLA attacks

**Framework Structure**:
1. Data Construction (simulation + real-world)
2. Model Training (backdoor injection)
3. Inference (adversarial attacks)

**New Attack: BackdoorVLA**

**Objective**: Force VLA to execute attacker-specified long-horizon action sequence

**Trigger Types**:
- Visual: Blue cube in scene
- Textual: Phrase "~*magic*~" in instruction
- Bi-modal: Both triggers combined

**Evaluation**:
- Benchmarks: LIBERO (Object, Spatial, Goal, LIBERO-10)
- Models: OpenVLA, SpatialVLA, pi0-fast
- Results: 58.4% average targeted success, 100% on selected tasks

**Defense Evaluation**:
- Textual defenses: Safe prompting
- Visual defenses: Image preprocessing
- Bi-modal defenses: Combined approaches
- Finding: Current defenses insufficient

**Limitations**:
- Backdoor requires training pipeline access
- Limited to specific trigger patterns
- Real-world deployment challenging

---

### 3.5 FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models

**Citation**: Anonymous, arXiv:2509.19870, September 2025

**Contribution**: Action-freezing attack inducing robot paralysis

**Attack Mechanism**:
- Min-max bi-level optimization
- Learnable multi-prompts expand prompt embedding space coverage
- Single adversarial image causes action freeze across prompts

**Mathematical Framework**:
```
min_delta max_p L_freeze(f(I + delta, p))

where p spans diverse prompt embeddings
```

**Evaluation**:
- Models: SpatialVLA (73.3% ASR), OpenVLA (95.4% ASR), pi0 (59.8% ASR)
- Benchmark: LIBERO

**Key Finding**:
- Adversarial images exhibit strong transferability across prompts
- Single optimized perturbation works for multiple instructions

**Limitations**:
- Untargeted attack only (induces failure, not specific behavior)
- Requires white-box access for optimization
- Transfer to real-world not extensively tested

---

### 3.6 VLA-Fool: When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models

**Citation**: Anonymous, arXiv:2511.16203, November 2025

**Contribution**: Comprehensive multimodal attack framework

**Attack Levels**:

1. **Textual Perturbations**
   - Gradient-based manipulation
   - Prompt-based manipulation

2. **Visual Perturbations**
   - Patch attacks
   - Noise distortions

3. **Cross-modal Misalignment**
   - Disrupts semantic correspondence between perception and instruction

**Evaluation**:
- Benchmark: LIBERO (Spatial, Object, Goal, Long-horizon)
- Model: OpenVLA fine-tuned
- Results: Complete failure (100% FR) with arm patch attacks

**Key Finding**:
- Visual attacks more destructive than linguistic attacks
- Small localized perturbations often more effective than broad attacks

---

### 3.7 ADVLA: Attention-Guided, Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Model

**Citation**: Anonymous, arXiv:2511.21663, November 2025

**Contribution**: Attention-guided sparse attack in feature space

**Key Innovation**:
- Perturbs features in projection space (visual encoder to LLM)
- Attention guidance makes perturbations focused and sparse
- Top-K masking modifies <10% of patches

**Results**:
- L_inf = 4/255 constraint
- Attack success rate: ~100%
- Patches modified: <10%

**Advantages**:
- Faster than traditional patch attacks
- More stealthy (sparser perturbation)
- Works in feature space

---

### 3.8 SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks

**Citation**: Anonymous, arXiv:2601.14323, January 2026

**Contribution**: Exploits action chunking vulnerability

**Key Insight**:
- Action chunking predicts K future actions simultaneously
- Creates intra-chunk visual open-loop
- Tiny per-step drift compounds over chunk
- Example: 1mm drift per step becomes 5cm over K=50 chunk

**Attack Properties**:
- Black-box
- Stealthy (imperceptible per-step perturbation)
- Exploits architectural feature, not model weights

---

### 3.9 Adversarial Attacks on Robotic Vision Language Action Models (RobotGCG)

**Citation**: Jones et al., arXiv:2506.03350, June 2025

**Contribution**: Adapts LLM jailbreaking to VLA

**Key Innovation**:
- Applies Greedy Coordinate Gradient (GCG) attack to VLA
- Textual attacks persist over multiple rollout steps
- More efficient than standard LLM jailbreaks

**Efficiency**:
- Optimization steps: 30-110 (vs 500 for LLM chatbots)
- Time per success: 3-10 min (vs >1 hour for LLMs)

**Results**:
- Achieves complete control authority over VLA
- Attacks persist as model observes new visual inputs
- Environmental agnosticism: sim-to-real transfer works

---

## 4. Benchmarks and Evaluation Protocols

### 4.1 LIBERO Benchmark

**Structure**:
| Suite | Tasks | Focus |
|-------|-------|-------|
| LIBERO-Spatial | 10 | Spatial reasoning |
| LIBERO-Object | 10 | Object manipulation |
| LIBERO-Goal | 10 | Goal-directed behavior |
| LIBERO-10 | 10 | Diverse long-horizon |
| LIBERO-100 | 100 | Comprehensive evaluation |

**Metrics Used in Attack Papers**:
- Task Success Rate (TSR)
- Task Failure Rate (TFR)
- Attack Success Rate (ASR)
- Action Deviation (L2 distance)
- Trajectory Error

### 4.2 Other Benchmarks

| Benchmark | Environment | Use in Attack Papers |
|-----------|-------------|---------------------|
| SimplerEnv | MuJoCo | BadVLA, SpatialVLA evaluation |
| MetaWorld | MuJoCo | Limited VLA evaluation |
| BridgeData V2 | Real-world | Training data for UPA-RFAS |

### 4.3 Evaluation Gaps

1. **Cross-model evaluation**: Most papers test on single model (OpenVLA)
2. **Real-world evaluation**: Limited physical robot experiments
3. **Defense evaluation**: Inconsistent defense baselines
4. **Long-horizon tasks**: Most attacks tested on short episodes

---

## 5. Summary of Attack Landscape

### 5.1 What Has Been Done

| Attack Type | Representative Work | Venue | Code |
|-------------|---------------------|-------|------|
| Patch-based (single model) | UADA/UPA/TMA | ICCV 2025 | Yes |
| Patch-based (universal) | UPA-RFAS | arXiv 2025 | No |
| Embedding disruption | EDPA | arXiv 2025 | No |
| Action freezing | FreezeVLA | arXiv 2025 | No |
| Attention-guided sparse | ADVLA | arXiv 2025 | No |
| Jailbreak/textual | RobotGCG | arXiv 2025 | Yes |
| Cross-modal | VLA-Fool | arXiv 2025 | No |
| Backdoor (bi-modal) | BadVLA, BackdoorVLA | arXiv 2025 | No |
| Action chunking exploit | SilentDrift | arXiv 2026 | No |

### 5.2 Attack Success Rates Summary

| Attack | Model | Benchmark | Success Rate |
|--------|-------|-----------|--------------|
| UADA | OpenVLA | LIBERO | ~100% failure |
| EDPA | OpenVLA-OFT | LIBERO | 62% increase |
| FreezeVLA | OpenVLA | LIBERO | 95.4% ASR |
| ADVLA | OpenVLA | LIBERO | ~100% ASR |
| BackdoorVLA | OpenVLA | LIBERO | 58.4% targeted |

---

## 6. Identified Gaps in Literature

### 6.1 Confirmed Unexplored Directions

| Direction | Evidence of Gap | Relevant Classical Work |
|-----------|-----------------|------------------------|
| Frequency-domain attacks (DCT/FFT) | No VLA paper uses spectral perturbations | SSA (ECCV 2022), Yin et al. (NeurIPS 2019) |
| Data-free universal perturbations | UPA-RFAS requires training data; no Jigsaw-based VLA attack | Zhang et al. (ICCV 2021) |
| Adversarial reprogramming | No VLA task hijacking work | Elsayed et al. (ICLR 2019) |
| Intermediate-layer attacks (full ILA) | EDPA is partial; no full ILA on VLA | Huang et al. (ICCV 2019) |

### 6.2 Methodological Gaps

1. **No frequency-domain analysis**: All attacks operate in spatial/pixel domain
2. **No data-free attacks**: All require some form of task-relevant data
3. **No reprogramming attacks**: No work on repurposing VLA for unintended tasks
4. **Limited transferability study**: Most work is single-model focused
5. **No certified robustness**: No provable bounds on VLA robustness

### 6.3 Evaluation Gaps

1. **Standardized defense baselines**: Each paper uses different defenses
2. **Real-world evaluation**: Most work simulation-only
3. **Cross-benchmark evaluation**: Papers use different LIBERO subsets
4. **Computational cost analysis**: Attack efficiency not consistently reported

---

## References

1. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
2. Wang et al., "Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics," ICCV 2025, arXiv:2411.13587.
3. Lu et al., "When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models," arXiv:2511.21192, 2025.
4. "Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models," arXiv:2510.13237, 2025.
5. "AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models," arXiv:2511.12149, 2025.
6. "FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models," arXiv:2509.19870, 2025.
7. "VLA-Fool: When Alignment Fails," arXiv:2511.16203, 2025.
8. "ADVLA: Attention-Guided, Patch-Wise Sparse Adversarial Attacks," arXiv:2511.21663, 2025.
9. "SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks," arXiv:2601.14323, 2026.
10. Jones et al., "Adversarial Attacks on Robotic Vision Language Action Models," arXiv:2506.03350, 2025.
11. Moosavi-Dezfooli et al., "Universal Adversarial Perturbations," CVPR 2017.
12. Zhang et al., "Data-Free Universal Adversarial Perturbation and Black-Box Attack," ICCV 2021.
13. Long et al., "Frequency Domain Model Augmentation for Adversarial Attack," ECCV 2022.
14. Elsayed et al., "Adversarial Reprogramming of Neural Networks," ICLR 2019.
15. Huang et al., "Enhancing Adversarial Example Transferability with an Intermediate Level Attack," ICCV 2019.
16. Yin et al., "A Fourier Perspective on Model Robustness in Computer Vision," NeurIPS 2019.

---

*Document prepared for research planning purposes. All citations should be verified against original sources.*
