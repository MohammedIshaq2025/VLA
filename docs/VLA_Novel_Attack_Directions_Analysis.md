# Critical Analysis: Novel Adversarial Attack Directions for VLA Models

## Document Information
- **Date**: January 2026
- **Purpose**: Rigorous evaluation of proposed research directions
- **Methodology**: Multi-angle validation (novelty, theory, implementation, benchmarking)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Evaluation Framework](#2-evaluation-framework)
3. [Direction 1: Frequency-Domain Attacks](#3-direction-1-frequency-domain-adversarial-attacks-on-vla)
4. [Direction 2: Data-Free Universal Adversarial Perturbations](#4-direction-2-data-free-universal-adversarial-perturbations-for-vla)
5. [Direction 3: Adversarial Reprogramming](#5-direction-3-adversarial-reprogramming-of-vla)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Quick Validation Experiments](#7-quick-validation-experiments)
8. [Risk Assessment](#8-risk-assessment)
9. [Final Recommendations](#9-final-recommendations)

---

## 1. Executive Summary

After rigorous literature review and critical analysis, three research directions have been identified as genuinely unexplored in VLA adversarial attacks. This document provides unbiased evaluation from multiple perspectives.

**Key Finding**: All three directions are technically feasible but differ significantly in implementation complexity, theoretical grounding, and publication risk.

| Direction | Novelty | Theory | Implementation | Risk |
|-----------|---------|--------|----------------|------|
| Frequency-Domain | High | Strong | Medium | Low |
| Data-Free UAP | High | Strong | Low | Medium |
| Adversarial Reprogramming | Very High | Moderate | High | High |

---

## 2. Evaluation Framework

Each direction is evaluated across six dimensions:

### 2.1 Novelty Assessment
- Is this truly unexplored in VLA?
- What distinguishes it from existing work?
- Can novelty claims withstand reviewer scrutiny?

### 2.2 Theoretical Soundness
- Is there mathematical justification?
- Does theory from related domains transfer?
- Are there potential theoretical holes?

### 2.3 Implementation Feasibility
- Code availability from related work
- Compatibility with LIBERO/simulation
- Engineering complexity

### 2.4 Benchmarking Validity
- Appropriate baselines exist?
- Metrics well-defined?
- Comparison to existing VLA attacks possible?

### 2.5 Publication Viability
- Suitable for top venues?
- Sufficient contribution for acceptance?
- Potential reviewer concerns?

### 2.6 Quick Validation Potential
- Can we get signal in 1-2 weeks?
- What would validate/invalidate the approach?

---

## 3. Direction 1: Frequency-Domain Adversarial Attacks on VLA

### 3.1 Precise Definition

**Claim**: Adversarial perturbations crafted in the frequency domain (using DCT or FFT) have not been applied to VLA models.

**Distinction from Existing Work**:
- All current VLA attacks (UADA, UPA, EDPA, FreezeVLA, UPA-RFAS) operate in spatial/pixel domain
- Frequency-domain attacks optimize perturbations in DCT coefficient space
- This is fundamentally different from patch-based attacks

### 3.2 Novelty Validation

**Evidence of Gap**:
1. Searched "VLA + frequency + DCT + Fourier + spectral + attack" - zero relevant results
2. Examined all 10+ VLA attack papers - none mention frequency-domain perturbations
3. Recent frequency work in VLA (FAST tokenizer, pi0) is for action encoding, not attacks

**Potential Novelty Concerns**:
- Reviewer might argue this is "incremental" (just applying known technique to new domain)
- Counter: Domain transfer is non-trivial; VLA has unique properties (action discretization, temporal dependencies)

**Novelty Score**: 8/10

### 3.3 Theoretical Analysis

**Foundation Papers**:
1. Yin et al., "A Fourier Perspective on Model Robustness in Computer Vision," NeurIPS 2019
2. Long et al., "Frequency Domain Model Augmentation for Adversarial Attack," ECCV 2022

**Key Theoretical Results**:

From Yin et al. (NeurIPS 2019):
```
Theorem (Informal): CNNs and ViTs exhibit non-uniform sensitivity across frequency bands.
- High-frequency perturbations: High error rate, low human perceptibility
- Low-frequency perturbations: More perceptible but affect different failure modes
```

**Transfer to VLA**:

VLA visual encoders (SigLIP, DINOv2) are ViT-based, inheriting frequency sensitivities:

```
Let I be input image, F = DCT(I) its frequency representation
Let delta_f be perturbation in frequency domain

Adversarial image: I_adv = IDCT(F + delta_f)

For VLA with action discretization into B=256 bins:
Action token: a_i = argmax_{b in [0,B-1]} p(b | I_adv, s)

Key insight: Small feature perturbations can cause bin-boundary crossings
If continuous action estimate is near bin boundary, frequency perturbation can flip the token
```

**Mathematical Formulation**:

```
Optimization objective:
max_{delta_f} L_VLA(I_adv, s, a*)
subject to: ||IDCT(delta_f)||_inf <= epsilon

where:
- L_VLA is cross-entropy loss on action tokens
- a* is ground-truth action sequence
- epsilon is perturbation budget (e.g., 8/255)

Gradient computation:
d L / d delta_f = d L / d I_adv * d I_adv / d delta_f
                = d L / d I_adv * IDCT_matrix
```

**Theoretical Concerns**:
1. Frequency sensitivity results are from image classifiers; VLA action decoders may behave differently
2. Action discretization may provide robustness (quantization as defense)
3. Temporal dynamics may dilute single-frame frequency attacks

**Theory Score**: 7/10

### 3.4 Implementation Analysis

**Required Components**:
1. DCT/IDCT implementation (available in PyTorch, scipy)
2. VLA model (OpenVLA open-source)
3. LIBERO environment (open-source)
4. Gradient computation through VLA

**Code Availability**:
- SSA (ECCV 2022): https://github.com/yuyang-long/SSA - Full PyTorch implementation
- VAFA (MICCAI 2023): https://github.com/asif-hanif/vafa - 3D DCT for medical imaging
- PyTorch DCT: torch.fft module supports DCT operations

**Implementation Sketch**:
```python
import torch
import torch.fft as fft

class FrequencyDomainVLAAttack:
    def __init__(self, vla_model, epsilon=8/255, steps=50):
        self.model = vla_model
        self.epsilon = epsilon
        self.steps = steps

    def dct_2d(self, x):
        # 2D DCT via FFT
        return fft.dct(fft.dct(x.transpose(-1,-2), norm='ortho').transpose(-1,-2), norm='ortho')

    def idct_2d(self, x):
        return fft.idct(fft.idct(x.transpose(-1,-2), norm='ortho').transpose(-1,-2), norm='ortho')

    def attack(self, image, instruction, target_action=None):
        image_dct = self.dct_2d(image)
        delta_f = torch.zeros_like(image_dct, requires_grad=True)

        optimizer = torch.optim.Adam([delta_f], lr=0.01)

        for step in range(self.steps):
            adv_image = self.idct_2d(image_dct + delta_f)
            adv_image = torch.clamp(adv_image, 0, 1)

            # Forward through VLA
            action_logits = self.model(adv_image, instruction)

            # Loss: maximize deviation from correct action
            if target_action is None:  # Untargeted
                loss = -self.model.compute_action_loss(action_logits, correct_action)
            else:  # Targeted
                loss = self.model.compute_action_loss(action_logits, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project to epsilon ball in spatial domain
            with torch.no_grad():
                spatial_perturbation = self.idct_2d(delta_f)
                spatial_perturbation = torch.clamp(spatial_perturbation, -self.epsilon, self.epsilon)
                delta_f.data = self.dct_2d(spatial_perturbation)

        return self.idct_2d(image_dct + delta_f)
```

**Engineering Challenges**:
1. OpenVLA gradient computation may require modifications
2. DCT operations need careful handling of image normalization
3. Batch processing for efficiency

**Implementation Score**: 8/10

### 3.5 Benchmarking Strategy

**Baselines for Comparison**:
1. PGD attack (spatial domain) - standard baseline
2. UADA/UPA (existing VLA attacks) - direct comparison
3. EDPA (embedding disruption) - feature-level comparison

**Metrics**:
- Attack Success Rate (ASR): % of episodes where attack causes failure
- Action Deviation: L2 distance between clean and adversarial actions
- Task Failure Rate (TFR): % of tasks that fail to complete
- Perturbation Invisibility: PSNR, SSIM, LPIPS

**Experimental Design**:
```
Benchmarks: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10
Models: OpenVLA, OpenVLA-OFT, pi0 (if accessible)
Perturbation budgets: epsilon in {4/255, 8/255, 16/255}
Attack variants:
  - Low-frequency attack (perturb only low DCT coefficients)
  - High-frequency attack (perturb only high DCT coefficients)
  - Full-spectrum attack (perturb all coefficients)
  - Adaptive attack (learn which frequencies to perturb)
```

**Benchmarking Concerns**:
1. Need to match evaluation protocol of existing VLA attack papers
2. Should use same LIBERO tasks as Wang et al. (ICCV 2025) for direct comparison
3. Statistical significance requires multiple seeds

**Benchmarking Score**: 8/10

### 3.6 Publication Viability

**Target Venues**:
- Tier 1: NeurIPS, ICML, ICLR (ML venues)
- Tier 1: CVPR, ICCV, ECCV (vision venues)
- Tier 1: CoRL, RSS, ICRA (robotics venues)

**Contribution Framing**:
1. "First frequency-domain adversarial attack on VLA models"
2. "Theoretical and empirical analysis of VLA frequency sensitivity"
3. "New attack vector with defense implications"

**Potential Reviewer Concerns**:
1. "Incremental - just applying existing technique to new domain"
   - Counter: Demonstrate VLA-specific findings (action discretization effects, temporal dynamics)
2. "Limited novelty in attack formulation"
   - Counter: Introduce VLA-specific loss functions, analyze frequency-action relationships
3. "Why not just use existing spatial attacks?"
   - Counter: Show frequency attacks have different properties (transferability, defense bypass)

**Publication Score**: 7/10

### 3.7 Direction 1 Summary

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 8/10 | Clear gap, but technique exists in other domains |
| Theory | 7/10 | Strong foundation, some transfer uncertainty |
| Implementation | 8/10 | Good code availability, moderate complexity |
| Benchmarking | 8/10 | Clear baselines and metrics |
| Publication | 7/10 | Solid contribution, some incrementality concerns |
| **Overall** | **7.6/10** | Strong direction with manageable risks |

---

## 4. Direction 2: Data-Free Universal Adversarial Perturbations for VLA

### 4.1 Precise Definition

**Claim**: Universal adversarial perturbations computed without any access to robotic training data (using artificial images like Jigsaw patterns) have not been applied to VLA.

**Critical Distinction from UPA-RFAS**:

| Property | UPA-RFAS (Lu et al., 2025) | Proposed Data-Free UAP |
|----------|---------------------------|------------------------|
| Data required | BridgeData V2 (60k trajectories) + LIBERO | None (Jigsaw patterns) |
| Perturbation type | Localized patch | Full-image perturbation |
| Optimization | On surrogate model with task data | Data-free objective |
| Threat model | Gray-box (needs surrogate) | Black-box (no model access needed after training) |

### 4.2 Novelty Validation

**Evidence of Gap**:
1. UPA-RFAS (Nov 2025) requires training data - explicitly stated
2. All other VLA attacks require task-relevant data or model access
3. Data-free UAP (ICCV 2021) never applied to VLA

**Novelty Strength**:
- Data-free setting is strictly more challenging
- Has clear practical motivation (attacker has no access to robotic data)
- Different perturbation type (full-image vs patch)

**Novelty Concerns**:
- Reviewer might question whether data-free UAP can work on VLA
- Need strong empirical validation

**Novelty Score**: 9/10

### 4.3 Theoretical Analysis

**Foundation Paper**: Zhang et al., "Data-Free Universal Adversarial Perturbation and Black-Box Attack," ICCV 2021

**Core Insight**:
```
Theorem (Zhang et al.): Universal adversarial perturbations can be computed using
artificial Jigsaw images (random patch permutations) instead of natural images.

Key observation: Jigsaw images contain diverse edge orientations and frequencies
that activate similar neural pathways as natural images.
```

**Transfer to VLA**:

```
VLA Pipeline: Image -> Visual Encoder (SigLIP+DINOv2) -> Projector -> LLM -> Action

Key insight: The visual encoder processes images identically regardless of downstream task.
If we can find a universal perturbation that disrupts SigLIP+DINOv2 features,
it should transfer to VLA action prediction.

Data-free objective:
delta* = argmax_{||delta||_inf <= eps} sum_{I_j in Jigsaw} D_KL(f(I_j + delta) || f(I_j))

where f is the visual encoder output distribution.
```

**Mathematical Formulation**:

```
Let J = {J_1, ..., J_N} be a set of Jigsaw images (randomly generated)
Let E_v be the visual encoder (SigLIP + DINOv2 fusion)
Let delta be the universal perturbation

Objective 1 (Feature disruption):
max_delta sum_{J_i in J} ||E_v(J_i + delta) - E_v(J_i)||_2

Objective 2 (Dominant label - adapted from Zhang et al.):
max_delta sum_{J_i in J} -log(p(y != y* | J_i + delta))

For VLA, adapt to action space:
max_delta sum_{J_i in J} H(p(a | J_i + delta, s_fixed))

where H is entropy (maximize uncertainty in action prediction)
```

**Theoretical Concerns**:
1. Jigsaw images may not capture VLA-relevant features (robotic scenes differ from ImageNet)
2. Universal perturbation may not generalize across VLA tasks
3. Action discretization may provide quantization robustness

**Theory Score**: 6/10 (more uncertainty than frequency approach)

### 4.4 Implementation Analysis

**Required Components**:
1. Jigsaw image generator
2. Visual encoder access (for feature computation)
3. UAP optimization loop
4. VLA model for final evaluation

**Code Availability**:
- Data-free UAP (ICCV 2021): Multiple PyTorch implementations
- https://github.com/ChaoningZhang/Awesome-Universal-Adversarial-Perturbations
- https://github.com/kenny-co/sgd-uap-torch

**Implementation Sketch**:
```python
import torch
import numpy as np
from torchvision import transforms

class DataFreeUAPVLA:
    def __init__(self, visual_encoder, epsilon=10/255, num_jigsaw=1000):
        self.encoder = visual_encoder
        self.epsilon = epsilon
        self.num_jigsaw = num_jigsaw

    def generate_jigsaw(self, size=224, grid=4):
        """Generate random Jigsaw image"""
        # Create random colored patches
        patch_size = size // grid
        patches = []
        for _ in range(grid * grid):
            color = np.random.rand(3)
            patch = np.ones((patch_size, patch_size, 3)) * color
            patches.append(patch)

        # Random permutation
        np.random.shuffle(patches)

        # Reassemble
        image = np.zeros((size, size, 3))
        for i, patch in enumerate(patches):
            row, col = i // grid, i % grid
            image[row*patch_size:(row+1)*patch_size,
                  col*patch_size:(col+1)*patch_size] = patch

        return torch.tensor(image).permute(2, 0, 1).float()

    def compute_uap(self, iterations=10):
        """Compute data-free UAP using Jigsaw images"""
        delta = torch.zeros(3, 224, 224, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.01)

        jigsaw_images = [self.generate_jigsaw() for _ in range(self.num_jigsaw)]

        for iteration in range(iterations):
            total_loss = 0
            for jigsaw in jigsaw_images:
                jigsaw = jigsaw.unsqueeze(0)

                # Clean features
                with torch.no_grad():
                    clean_features = self.encoder(jigsaw)

                # Adversarial features
                adv_image = torch.clamp(jigsaw + delta, 0, 1)
                adv_features = self.encoder(adv_image)

                # Maximize feature disruption
                loss = -torch.norm(adv_features - clean_features, p=2)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

        return delta.detach()

    def evaluate_on_vla(self, vla_model, test_images, instructions):
        """Evaluate UAP on actual VLA tasks"""
        delta = self.compute_uap()

        results = []
        for image, instruction in zip(test_images, instructions):
            clean_action = vla_model(image, instruction)
            adv_action = vla_model(image + delta, instruction)

            deviation = torch.norm(clean_action - adv_action)
            results.append(deviation.item())

        return np.mean(results)
```

**Engineering Challenges**:
1. Need to extract visual encoder from VLA (may require model surgery)
2. Jigsaw generation hyperparameters need tuning
3. Evaluation requires full VLA pipeline

**Implementation Score**: 7/10

### 4.5 Benchmarking Strategy

**Baselines**:
1. Random noise (sanity check)
2. Data-dependent UAP (upper bound)
3. UPA-RFAS (state-of-the-art universal patch)
4. PGD per-image attack (non-universal baseline)

**Key Comparisons**:
```
Experiment 1: Attack success vs data requirements
- Data-free UAP (our method): 0 robotic images
- UPA-RFAS: 60k+ trajectories
- Question: How much does data-free sacrifice in effectiveness?

Experiment 2: Transferability across VLA models
- Train UAP on OpenVLA encoder
- Test on OpenVLA, SpatialVLA, pi0
- Question: Does data-free improve or hurt transferability?

Experiment 3: Task generalization
- Train UAP without task knowledge
- Test on LIBERO-Spatial, Object, Goal, Long-horizon
- Question: Does single UAP work across diverse tasks?
```

**Benchmarking Concerns**:
1. May underperform data-dependent methods significantly
2. Need to show practical value despite potential performance gap
3. Comparison to UPA-RFAS must be fair (different perturbation types)

**Benchmarking Score**: 7/10

### 4.6 Publication Viability

**Target Venues**: Same as Direction 1

**Contribution Framing**:
1. "First data-free adversarial attack on VLA models"
2. "Practical threat model: Attacker needs zero access to robotic data"
3. "Analysis of VLA visual encoder universal vulnerabilities"

**Potential Reviewer Concerns**:
1. "Performance worse than data-dependent methods - what's the point?"
   - Counter: Data-free is more realistic threat model; analyze trade-off
2. "Why not just collect some robotic data?"
   - Counter: Robotic data is expensive; data-free enables rapid attack deployment
3. "Jigsaw images too different from robot scenes"
   - Counter: Empirically validate; analyze which Jigsaw properties transfer

**Publication Score**: 6/10 (higher risk due to potential negative results)

### 4.7 Direction 2 Summary

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 9/10 | Strong novelty, clear distinction from UPA-RFAS |
| Theory | 6/10 | Some uncertainty in Jigsaw-VLA transfer |
| Implementation | 7/10 | Straightforward but needs encoder extraction |
| Benchmarking | 7/10 | Clear setup but may show performance gap |
| Publication | 6/10 | Risk of negative results |
| **Overall** | **7.0/10** | High novelty but higher risk |

---

## 5. Direction 3: Adversarial Reprogramming of VLA

### 5.1 Precise Definition

**Claim**: Adversarial reprogramming (making a model perform an unintended task via input perturbation) has not been applied to VLA models.

**Distinction**:
- Existing VLA attacks cause failure or specific wrong actions
- Reprogramming hijacks the model to perform a completely different task
- This is a fundamentally different threat model

### 5.2 Novelty Validation

**Evidence of Gap**:
1. Searched "VLA + reprogramming + task hijacking" - zero results
2. No VLA paper discusses repurposing for unintended tasks
3. Adversarial reprogramming (ICLR 2019) never applied to action models

**Novelty Strength**:
- Entirely new threat model for VLA
- Opens discussion on VLA-as-a-Service security
- Paradigm shift from "cause failure" to "hijack functionality"

**Novelty Concerns**:
- May be difficult to define meaningful reprogramming tasks for VLA
- Reviewer may question practical relevance

**Novelty Score**: 10/10

### 5.3 Theoretical Analysis

**Foundation Paper**: Elsayed et al., "Adversarial Reprogramming of Neural Networks," ICLR 2019

**Core Result**:
```
Theorem (Elsayed et al.): A neural network f trained on task A can be reprogrammed
to perform task B by learning:
1. Input transformation h_f: maps task B inputs to task A input space
2. Universal perturbation W: added to all transformed inputs
3. Output mapping M: maps task A outputs to task B labels

such that: M(f(h_f(x_B) + W)) approx g_B(x_B)
```

**Application to VLA**:

```
VLA performs: Task A = "Follow language instruction to manipulate objects"
Reprogrammed task B options:
1. "Always drop held object"
2. "Move to fixed position regardless of instruction"
3. "Execute attacker-specified trajectory"
4. "Respond only to specific objects, ignore others"

Formulation:
Let f_VLA: (Image, Instruction) -> Action
Let W be the reprogramming perturbation
Let s_fixed be a fixed "dummy" instruction

Reprogramming objective:
W* = argmin_W E_{x,y ~ D_target} [L(y, M(f_VLA(h(x) + W, s_fixed)))]

where:
- D_target is the attacker's target task distribution
- M maps VLA 7-DoF actions to target task labels
- h embeds target task inputs into VLA image space
```

**Theoretical Concerns**:
1. VLA action space is very different from ImageNet classes
2. Mapping from continuous 7-DoF actions to discrete task labels is non-trivial
3. Instruction modality complicates reprogramming

**Theory Score**: 5/10 (significant theoretical gaps)

### 5.4 Implementation Analysis

**Challenges**:
1. Defining meaningful target tasks for VLA reprogramming
2. Designing output mapping M for action space
3. Handling instruction modality

**Proposed Approach**:
```python
class VLAReprogramming:
    def __init__(self, vla_model, target_task='binary_classification'):
        self.model = vla_model
        self.target_task = target_task

    def embed_input(self, target_input, image_size=224):
        """Embed target task input into VLA image space"""
        # For MNIST digit classification as target task:
        # Resize digit to small patch, place in center of image
        padding = (image_size - 28) // 2
        embedded = torch.zeros(3, image_size, image_size)
        embedded[:, padding:padding+28, padding:padding+28] = target_input.expand(3, -1, -1)
        return embedded

    def action_to_label(self, action):
        """Map VLA 7-DoF action to binary label"""
        # Example: Use sign of gripper action as binary classification
        gripper_action = action[6]  # Last dimension is gripper
        return 1 if gripper_action > 0 else 0

    def train_reprogramming(self, target_dataset, epochs=100):
        W = torch.zeros(3, 224, 224, requires_grad=True)
        s_fixed = "pick up the object"  # Fixed dummy instruction

        optimizer = torch.optim.Adam([W], lr=0.01)

        for epoch in range(epochs):
            for x, y in target_dataset:
                embedded = self.embed_input(x)
                adv_input = embedded + W

                action = self.model(adv_input, s_fixed)
                predicted_label = self.action_to_label(action)

                loss = F.cross_entropy(predicted_label, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return W
```

**Engineering Challenges**:
1. Gradient flow through action discretization
2. Defining sensible output mapping
3. Large search space for reprogramming perturbation

**Implementation Score**: 4/10 (significant engineering challenges)

### 5.5 Benchmarking Strategy

**Target Tasks for Reprogramming**:
1. MNIST digit classification (standard reprogramming benchmark)
2. Binary object presence detection
3. Trajectory direction classification

**Metrics**:
- Reprogramming accuracy on target task
- Original task degradation
- Perturbation magnitude

**Benchmarking Concerns**:
1. No established baselines for VLA reprogramming
2. Unclear what success looks like
3. May need to define new metrics

**Benchmarking Score**: 4/10

### 5.6 Publication Viability

**Potential Contribution**:
1. "First adversarial reprogramming attack on VLA models"
2. "New threat model: VLA task hijacking"
3. "Security implications for VLA-as-a-Service"

**Reviewer Concerns**:
1. "Practical relevance unclear"
2. "Reprogramming target tasks seem artificial"
3. "Technical contribution limited if reprogramming doesn't work well"

**Publication Score**: 5/10 (high risk, high reward)

### 5.7 Direction 3 Summary

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 10/10 | Completely unexplored paradigm |
| Theory | 5/10 | Significant gaps in VLA transfer |
| Implementation | 4/10 | Major engineering challenges |
| Benchmarking | 4/10 | No established baselines |
| Publication | 5/10 | High risk, high reward |
| **Overall** | **5.6/10** | Exploratory direction, needs pilot study |

---

## 6. Comparative Analysis

### 6.1 Side-by-Side Comparison

| Criterion | Frequency-Domain | Data-Free UAP | Reprogramming |
|-----------|------------------|---------------|---------------|
| Novelty | High | Very High | Highest |
| Theoretical Foundation | Strong | Moderate | Weak |
| Implementation Complexity | Medium | Low-Medium | High |
| Code Availability | Excellent | Good | Limited |
| Risk of Negative Results | Low | Medium | High |
| Time to First Results | 2-3 weeks | 1-2 weeks | 4-6 weeks |
| Top Venue Suitability | High | Medium-High | Medium |

### 6.2 Risk-Reward Analysis

```
                    High Reward
                         |
                         |   Reprogramming (3)
                         |       *
                         |
                         |
        Data-Free UAP (2)|
              *          |
                         |
Frequency-Domain (1)     |
         *               |
                         |
    Low Risk ------------|------------ High Risk
                         |
                         |
                    Low Reward
```

### 6.3 Recommendation Matrix

| Researcher Profile | Recommended Direction |
|-------------------|----------------------|
| Risk-averse, needs publication | Direction 1 (Frequency-Domain) |
| Moderate risk tolerance | Direction 2 (Data-Free UAP) |
| High risk tolerance, exploratory | Direction 3 (Reprogramming) |
| Limited time (<4 weeks) | Direction 2 (Data-Free UAP) |
| Strong theory background | Direction 1 (Frequency-Domain) |

---

## 7. Quick Validation Experiments

### 7.1 Direction 1: Frequency-Domain Attack Validation

**Experiment 1A: Encoder Frequency Sensitivity (1-2 days)**
```
Objective: Verify SigLIP+DINOv2 has exploitable frequency sensitivity

Setup:
1. Load OpenVLA visual encoder
2. Generate test images from LIBERO
3. Apply DCT, perturb different frequency bands
4. Measure feature deviation

Expected signal:
- High-frequency perturbations should cause larger feature deviations
- If not, frequency attack may not be effective

Code snippet:
def test_frequency_sensitivity(encoder, image):
    dct = dct_2d(image)
    results = {}
    for freq_band in ['low', 'mid', 'high']:
        perturbed_dct = perturb_band(dct, freq_band, epsilon=8/255)
        perturbed_image = idct_2d(perturbed_dct)

        clean_feat = encoder(image)
        adv_feat = encoder(perturbed_image)

        results[freq_band] = torch.norm(adv_feat - clean_feat).item()
    return results
```

**Experiment 1B: Action Sensitivity (2-3 days)**
```
Objective: Check if frequency perturbations affect VLA actions

Setup:
1. Run OpenVLA on 10 LIBERO tasks
2. Apply frequency perturbations to input frames
3. Compare action outputs

Success criteria:
- Action deviation > baseline (random noise)
- Task failure rate increase > 20%
```

### 7.2 Direction 2: Data-Free UAP Validation

**Experiment 2A: Jigsaw Feature Disruption (1 day)**
```
Objective: Verify Jigsaw images can train effective perturbations

Setup:
1. Generate 1000 Jigsaw images
2. Train UAP on visual encoder only
3. Test on actual LIBERO images

Expected signal:
- UAP trained on Jigsaw should disrupt LIBERO image features
- If feature disruption < 50% of data-dependent UAP, approach may fail

Code snippet:
# Train on Jigsaw
uap = train_uap_on_jigsaw(encoder, num_jigsaw=1000)

# Test on real images
disruption_jigsaw = []
disruption_real = []
for real_image in libero_images:
    clean = encoder(real_image)
    adv = encoder(real_image + uap)
    disruption_real.append(torch.norm(adv - clean))

print(f"Mean disruption: {np.mean(disruption_real)}")
```

**Experiment 2B: Transfer to Actions (2 days)**
```
Objective: Check if encoder-level UAP affects actions

Setup:
1. Use UAP from 2A
2. Apply to VLA inputs
3. Measure action changes

Success criteria:
- Action deviation significantly above random noise
- Some task failures induced
```

### 7.3 Direction 3: Reprogramming Validation

**Experiment 3A: Action Space Mapping (2-3 days)**
```
Objective: Find meaningful mapping from actions to classification labels

Setup:
1. Collect VLA action outputs for various inputs
2. Analyze action space structure
3. Design label mapping

Questions to answer:
- Is action space structured enough for reprogramming?
- Can we find consistent mappings?
```

**Experiment 3B: Simple Reprogramming (3-4 days)**
```
Objective: Attempt binary classification reprogramming

Setup:
1. Target task: "Is gripper open or closed?"
2. Train reprogramming perturbation
3. Measure classification accuracy

Success criteria:
- Accuracy > 70% on target task
- If < 60%, reprogramming may not be viable for VLA
```

### 7.4 Validation Decision Tree

```
Week 1: Run Experiments 1A, 2A
        |
        +-- Frequency sensitivity confirmed?
        |   YES -> Continue with Direction 1
        |   NO  -> Deprioritize Direction 1
        |
        +-- Jigsaw disruption works?
            YES -> Continue with Direction 2
            NO  -> Deprioritize Direction 2

Week 2: Run Experiments 1B, 2B based on Week 1 results
        |
        +-- Frequency attacks affect actions?
        |   YES -> Direction 1 is viable, proceed
        |   NO  -> Direction 1 needs revision
        |
        +-- Data-free UAP affects actions?
            YES -> Direction 2 is viable, proceed
            NO  -> Direction 2 may need different approach

Week 3-4: If both viable, choose based on results
          If only one viable, focus on that
          If neither viable, consider Direction 3 or new directions
```

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Direction 1 | Direction 2 | Direction 3 |
|------|-------------|-------------|-------------|
| Theory doesn't transfer | Low | Medium | High |
| Implementation blockers | Low | Low | High |
| Negative results | Low | Medium | High |
| Insufficient novelty | Medium | Low | Low |

### 8.2 Publication Risks

| Risk | Direction 1 | Direction 2 | Direction 3 |
|------|-------------|-------------|-------------|
| Scooped before submission | Medium | Low | Very Low |
| Rejected as incremental | Medium | Low | Low |
| Insufficient experiments | Low | Medium | High |
| Missing baselines | Low | Medium | High |

### 8.3 Mitigation Strategies

**Direction 1 Risks**:
- Incrementality: Add VLA-specific analysis (action discretization effects, temporal dynamics)
- Being scooped: Move quickly, submit to robotics venue if ML venues slow

**Direction 2 Risks**:
- Negative results: Frame as "analysis of data-free limitations" if UAP underperforms
- Missing baselines: Implement UPA-RFAS baseline or clearly justify comparison

**Direction 3 Risks**:
- High failure risk: Start with pilot study before committing
- Unclear metrics: Propose new evaluation framework as contribution

---

## 9. Final Recommendations

### 9.1 Primary Recommendation: Direction 1 (Frequency-Domain)

**Rationale**:
1. Best risk-reward balance
2. Strongest theoretical foundation
3. Most straightforward path to publication
4. Clear novelty claim with manageable incrementality concerns

**Action Plan**:
- Week 1: Validation experiments 1A, 1B
- Week 2-3: Full implementation and LIBERO experiments
- Week 4-5: Analysis and paper writing
- Week 6: Internal review and submission

### 9.2 Secondary Recommendation: Direction 2 (Data-Free UAP)

**Rationale**:
1. Highest novelty among feasible options
2. Fastest to initial results
3. Strong practical motivation
4. Acceptable risk level

**Action Plan**:
- Week 1: Validation experiments 2A, 2B
- Week 2: Full UAP training and evaluation
- Week 3-4: Comparison with UPA-RFAS, analysis
- Week 5: Paper writing and submission

### 9.3 Exploratory: Direction 3 (Reprogramming)

**Rationale**:
1. Highest novelty but highest risk
2. Recommend as side project or future work
3. Start with pilot study only

**Action Plan**:
- Allocate 1 week maximum for pilot
- Only continue if Experiment 3B shows >70% accuracy
- Consider as second paper if successful

### 9.4 Parallel Strategy (If Resources Allow)

```
Week 1-2: Run validation for all three directions in parallel
Week 3: Evaluate results, commit to 1-2 directions
Week 4-6: Full development of chosen direction(s)
Week 7-8: Paper writing
```

---

## Appendix A: Code Repositories

| Resource | URL | Relevance |
|----------|-----|-----------|
| SSA (Frequency Attack) | https://github.com/yuyang-long/SSA | Direction 1 |
| SGD-UAP | https://github.com/kenny-co/sgd-uap-torch | Direction 2 |
| Data-Free UAP | https://github.com/ChaoningZhang/Awesome-Universal-Adversarial-Perturbations | Direction 2 |
| Adversarial Reprogramming | https://github.com/Prinsphield/Adversarial_Reprogramming | Direction 3 |
| OpenVLA | https://github.com/openvla/openvla | All |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | All |
| torchattacks | https://github.com/Harry24k/adversarial-attacks-pytorch | Baseline |

---

## Appendix B: Key Paper References

1. Moosavi-Dezfooli et al., "Universal Adversarial Perturbations," CVPR 2017
2. Zhang et al., "Data-Free Universal Adversarial Perturbation and Black-Box Attack," ICCV 2021
3. Long et al., "Frequency Domain Model Augmentation for Adversarial Attack," ECCV 2022
4. Elsayed et al., "Adversarial Reprogramming of Neural Networks," ICLR 2019
5. Yin et al., "A Fourier Perspective on Model Robustness in Computer Vision," NeurIPS 2019
6. Wang et al., "Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics," ICCV 2025
7. Lu et al., "When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models," arXiv 2025

---

*Document prepared for research planning. Recommendations based on current literature as of January 2026.*
