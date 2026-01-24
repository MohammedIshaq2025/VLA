# Deep Dive: Frequency-Domain Adversarial Attacks on VLA

## Document Purpose
This document provides a comprehensive understanding of frequency-domain adversarial attacks and a concrete validation plan for applying them to Vision-Language-Action models.

---

## Table of Contents
1. [Fundamentals: What is a Frequency-Domain Attack?](#1-fundamentals-what-is-a-frequency-domain-attack)
2. [Key Insight: ViT vs CNN Frequency Sensitivity](#2-key-insight-vit-vs-cnn-frequency-sensitivity)
3. [Taxonomy of Frequency-Domain Attacks](#3-taxonomy-of-frequency-domain-attacks)
4. [Top Frequency Attacks with Transferability](#4-top-frequency-attacks-with-transferability)
5. [VLA-Specific Considerations](#5-vla-specific-considerations)
6. [Available Code Repositories](#6-available-code-repositories)
7. [Step-by-Step Validation Plan](#7-step-by-step-validation-plan)
8. [Visual Explanations](#8-visual-explanations)

---

## 1. Fundamentals: What is a Frequency-Domain Attack?

### 1.1 The Core Concept

Traditional adversarial attacks (FGSM, PGD) add perturbations directly to pixel values:

```
Spatial Domain Attack:
X_adv = X + delta    (where delta is computed via gradients)
```

Frequency-domain attacks instead operate on the frequency representation:

```
Frequency Domain Attack:
1. F = Transform(X)           # Convert to frequency domain (DCT/FFT)
2. F_adv = F + delta_f        # Perturb frequency coefficients
3. X_adv = InverseTransform(F_adv)  # Convert back to spatial domain
```

### 1.2 Why Frequency Domain?

```
VISUAL: Image Decomposition

Original Image
     |
     v
+------------------+
|   DCT/FFT        |
|   Transform      |
+------------------+
     |
     v
+------------------+------------------+
|  Low Frequency   |  High Frequency  |
|  (Smooth regions,|  (Edges, textures|
|   overall shape) |   fine details)  |
+------------------+------------------+
```

**Key Properties**:
1. **Low-frequency components**: Contain overall shape, color gradients, smooth regions
2. **High-frequency components**: Contain edges, textures, fine details
3. **Different neural networks have different frequency sensitivities**

### 1.3 DCT vs FFT

| Transform | Output | Advantage | Common Use |
|-----------|--------|-----------|------------|
| DCT (Discrete Cosine Transform) | Real coefficients | Simpler, JPEG-compatible | Most attack papers |
| FFT (Fast Fourier Transform) | Complex (magnitude + phase) | More expressive | Phase attacks |
| DWT (Discrete Wavelet Transform) | Multi-resolution | Localized frequency | Less common |

**For VLA attacks, DCT is recommended** due to simpler implementation and proven effectiveness.

---

## 2. Key Insight: ViT vs CNN Frequency Sensitivity

### 2.1 Critical Finding for VLA

**VLA models use ViT-based encoders (SigLIP + DINOv2), NOT CNNs.**

This fundamentally changes the attack strategy:

```
ARCHITECTURE SENSITIVITY COMPARISON

+------------------+------------------+------------------+
|                  |      CNNs        |      ViTs        |
+------------------+------------------+------------------+
| High-Freq Attack | EFFECTIVE        | WEAK             |
| (edges, texture) | (main weakness)  | (robust)         |
+------------------+------------------+------------------+
| Low-Freq Attack  | WEAK             | EFFECTIVE        |
| (shape, color)   | (robust)         | (main weakness)  |
+------------------+------------------+------------------+
| Phase Attack     | MODERATE         | VERY EFFECTIVE   |
| (low-freq phase) |                  | (critical vuln.) |
+------------------+------------------+------------------+

Source: Kim et al., WACV 2024
```

### 2.2 Why This Matters for VLA

```
VLA Pipeline:

Image --> [SigLIP (ViT)] --> Features --> LLM --> Actions
      --> [DINOv2 (ViT)] --/

Both encoders are ViT-based!
Therefore: LOW-FREQUENCY attacks are more effective than HIGH-FREQUENCY attacks
```

### 2.3 Empirical Evidence

From Kim et al. (WACV 2024):
- ViT-B/16 with low-freq attack: 71.9% --> 55.8% accuracy (16.1% drop)
- ViT-B/16 with high-freq attack: 66.3% --> 33.4% accuracy (32.9% drop at much higher perturbation)
- **Low-frequency phase perturbations are most effective against ViTs**

---

## 3. Taxonomy of Frequency-Domain Attacks

### 3.1 By Perturbation Target

```
FREQUENCY ATTACK TAXONOMY

1. MAGNITUDE ATTACKS
   - Perturb |F| (amplitude of frequency coefficients)
   - Formula: F'_mag = F_mag * (1 + delta)
   - Effect: Changes intensity of frequency components

2. PHASE ATTACKS
   - Perturb angle(F) (phase of frequency coefficients)
   - Formula: F'_phase = F_phase + delta
   - Effect: Shifts spatial location of features
   - MOST EFFECTIVE AGAINST ViTs

3. COMBINED ATTACKS
   - Perturb both magnitude and phase
   - Formula: F' = (F_mag * delta_m) * exp(j * (F_phase + delta_p))

4. BAND-SELECTIVE ATTACKS
   - Only perturb specific frequency bands
   - Low-band: target ViTs
   - High-band: target CNNs
```

### 3.2 By Attack Objective

```
OBJECTIVE TYPES

1. UNTARGETED
   - Goal: Cause any misclassification
   - Loss: -CrossEntropy(f(X_adv), y_true)

2. TARGETED
   - Goal: Force specific output
   - Loss: CrossEntropy(f(X_adv), y_target)

3. UNIVERSAL
   - Goal: Single perturbation works for all images
   - Loss: E_x[-CrossEntropy(f(x + delta), y)]
```

---

## 4. Top Frequency Attacks with Transferability

### 4.1 Spectrum Simulation Attack (SSA) - ECCV 2022 Oral

**Paper**: Long et al., "Frequency Domain Model Augmentation for Adversarial Attack"

**Key Innovation**: Uses DCT to create diverse "spectrum saliency maps" that simulate attacking multiple models simultaneously.

**Algorithm**:
```
SSA Attack (Simplified):
Input: Image X, Model f, iterations T, samples N, rho, sigma

1. Initialize: delta = 0
2. For t = 1 to T:
     noise = 0
     For n = 1 to N:
         # Add Gaussian noise
         gauss = Normal(0, sigma)

         # Transform to frequency domain
         X_dct = DCT(X + gauss)

         # Apply random frequency mask
         mask = Uniform(1-rho, 1+rho)
         X_masked = X_dct * mask

         # Transform back
         X_idct = IDCT(X_masked)

         # Compute gradient
         grad = gradient(Loss(f(X_idct), y), X_idct)
         noise += grad

     # Average and update
     noise = noise / N
     delta = delta + alpha * sign(noise)
     delta = clip(delta, -epsilon, epsilon)

3. Return X + delta
```

**Transferability**: 95.4% success rate against 9 defense models

**Code**: https://github.com/yuyang-long/SSA

### 4.2 Spectral Adversarial Attack - WACV 2024

**Paper**: Kim et al., "Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective"

**Key Innovation**: Targets phase spectrum in low-frequency band specifically for ViTs.

**Algorithm**:
```
Spectral Attack (Phase-focused):
Input: Image X, Model f, lambda (balance param)

1. Compute FFT: F = FFT(X)
2. Separate: Magnitude = |F|, Phase = angle(F)
3. Initialize: delta_phase = 0
4. For t = 1 to T:
     # Reconstruct with perturbed phase
     F_adv = Magnitude * exp(j * (Phase + delta_phase))
     X_adv = IFFT(F_adv)

     # Loss balances attack strength and image quality
     Loss = lambda * ||X_adv - X||^2 - CrossEntropy(f(X_adv), y)

     # Update phase perturbation
     delta_phase = delta_phase - lr * gradient(Loss, delta_phase)

     # Mask to low-frequency only
     delta_phase = LowFreqMask(delta_phase)

5. Return X_adv
```

**Key Finding**: Phase attacks on low frequencies are most effective against ViTs.

**Code**: Not publicly available (must implement from paper)

### 4.3 Frequency-Aware Perturbation - IEEE TIP 2024

**Paper**: "Boosting the Transferability of Adversarial Attacks With Frequency-Aware Perturbation"

**Key Innovation**: Concentrates perturbations on frequency components that contribute most to model inference.

**Transferability**: Improved over SSA on both CNNs and ViTs.

### 4.4 Comparison Table

| Attack | Venue | Target Arch | Transferability | Code | VLA Suitability |
|--------|-------|-------------|-----------------|------|-----------------|
| SSA | ECCV 2022 | CNN (mainly) | 95.4% | Yes | Medium |
| Spectral Attack | WACV 2024 | ViT | High | No | HIGH |
| Freq-Aware | TIP 2024 | Both | Higher | No | HIGH |
| Low-Freq PGD | Custom | ViT | Medium | DIY | HIGH |

---

## 5. VLA-Specific Considerations

### 5.1 VLA Architecture Recap

```
VLA INPUT/OUTPUT FLOW

Input Image (224x224x3)
        |
        v
+-------------------+
| SigLIP Encoder    |---> Semantic Features
| (ViT-based)       |     (language-aligned)
+-------------------+
        |
+-------------------+
| DINOv2 Encoder    |---> Spatial Features
| (ViT-based)       |     (fine-grained)
+-------------------+
        |
        v
+-------------------+
| Feature Fusion    |
| (Concatenate)     |
+-------------------+
        |
        v
+-------------------+
| MLP Projector     |
+-------------------+
        |
        v
+-------------------+     +-------------------+
| Llama 2 7B        | <-- | Language Instruct |
+-------------------+     +-------------------+
        |
        v
+-------------------+
| Action Tokens     |
| (256 bins x 7 DoF)|
+-------------------+
        |
        v
7-DoF Robot Action
```

### 5.2 Attack Surface Analysis

```
WHERE TO ATTACK?

Option A: Attack Visual Input (OUR FOCUS)
- Perturb image in frequency domain
- Affects both SigLIP and DINOv2
- Propagates through entire pipeline
- MOST PRACTICAL for VLA

Option B: Attack Language Input
- Already done by RobotGCG
- Not frequency-based
- Out of scope

Option C: Attack Intermediate Features
- Done by EDPA
- Not strictly frequency-domain
- More complex to implement
```

### 5.3 VLA-Specific Challenges

1. **Action Discretization**: VLA discretizes continuous actions into 256 bins
   - Frequency perturbation must be strong enough to cross bin boundaries
   - Small perturbations may have no effect due to quantization

2. **Temporal Dynamics**: VLA runs at 5-10 Hz
   - Single-frame attack may not persist
   - May need temporally consistent perturbation

3. **Instruction Dependence**: VLA output depends on language instruction
   - Attack should work across different instructions
   - Or target specific instruction types

### 5.4 Proposed VLA Frequency Attack

```
VLA-FREQ ATTACK (Proposed)

Key Design Choices:
1. Use DCT (simpler than FFT)
2. Focus on LOW-FREQUENCY perturbation (ViT vulnerability)
3. Target action deviation loss (not classification)

Algorithm:
Input: VLA model, Image I, Instruction s, epsilon, iterations T

1. Compute DCT: F = DCT_2D(I)

2. Create low-frequency mask:
   M_low = zeros(H, W)
   M_low[0:H//4, 0:W//4] = 1  # Keep only low 25% frequencies

3. Initialize: delta_f = zeros_like(F)

4. For t = 1 to T:
     # Apply perturbation only to low frequencies
     F_adv = F + delta_f * M_low

     # Convert back to spatial
     I_adv = IDCT_2D(F_adv)
     I_adv = clip(I_adv, 0, 1)

     # Get VLA action prediction
     action = VLA(I_adv, s)
     action_clean = VLA(I, s)

     # Loss: maximize action deviation
     Loss = -||action - action_clean||^2

     # Or for targeted attack:
     # Loss = ||action - target_action||^2

     # Gradient update
     grad = gradient(Loss, delta_f)
     delta_f = delta_f + alpha * sign(grad)

     # Project to epsilon ball (in spatial domain)
     spatial_pert = IDCT_2D(delta_f * M_low)
     spatial_pert = clip(spatial_pert, -epsilon, epsilon)
     delta_f = DCT_2D(spatial_pert)

5. Return IDCT_2D(F + delta_f * M_low)
```

---

## 6. Available Code Repositories

### 6.1 Primary Repository: SSA

**URL**: https://github.com/yuyang-long/SSA

**Structure**:
```
SSA/
├── attack.py          # Main attack script
├── attack_methods.py  # Helper functions (DI, gkern)
├── dct.py            # DCT/IDCT implementation
├── verify.py         # Evaluation script
├── Normalize.py      # Image normalization
├── loader.py         # Data loading
├── dataset/          # Test images
└── torch_nets/       # Pre-trained models
```

**Key File: dct.py**
```python
# This file contains DCT/IDCT implementations
# You will need to extract and adapt these functions
```

**Dependencies**:
```
python==3.8
pytorch==1.8
pretrainedmodels==0.7
numpy==1.19
pandas==1.2
```

### 6.2 Secondary Repository: TransferAttack

**URL**: https://github.com/Trustworthy-AI-Group/TransferAttack

**Relevance**: Contains ViT-specific attacks and frequency-aware methods

**Structure**:
```
TransferAttack/
├── transferattack/
│   ├── input_transformation/
│   │   ├── ssa.py         # Spectrum Simulation Attack
│   │   └── ...
│   ├── model_related/
│   │   ├── tgr.py         # Token Gradient Regularization (ViT)
│   │   └── ...
│   └── ...
└── ...
```

### 6.3 Utility: torchattacks

**URL**: https://github.com/Harry24k/adversarial-attacks-pytorch

**Usage**: Baseline attacks (PGD, FGSM) for comparison
```python
pip install torchattacks

import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
adv_images = atk(images, labels)
```

### 6.4 Repository Comparison

| Repo | SSA | TransferAttack | torchattacks |
|------|-----|----------------|--------------|
| Frequency Attack | Yes | Yes | No |
| ViT Support | Limited | Yes | Yes |
| Easy to Modify | Medium | Medium | High |
| Documentation | Basic | Good | Excellent |
| Our Use | DCT code | ViT attacks | Baselines |

---

## 7. Step-by-Step Validation Plan

### 7.1 Overview Timeline

```
VALIDATION TIMELINE (2 WEEKS)

Week 1:
├── Day 1-2: Environment Setup
├── Day 3-4: Encoder Sensitivity Test
├── Day 5-6: Basic Frequency Attack Implementation
└── Day 7: Analysis & Decision Point

Week 2 (if Week 1 positive):
├── Day 1-3: Full VLA Attack Implementation
├── Day 4-5: LIBERO Evaluation
└── Day 6-7: Results Analysis
```

### 7.2 Day 1-2: Environment Setup

**Step 1: Clone Repositories**
```bash
# Create project directory
mkdir vla_freq_attack && cd vla_freq_attack

# Clone SSA for DCT code
git clone https://github.com/yuyang-long/SSA.git

# Clone OpenVLA
git clone https://github.com/openvla/openvla.git

# Clone LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

**Step 2: Create Environment**
```bash
conda create -n vla_freq python=3.10
conda activate vla_freq

# Install PyTorch
pip install torch torchvision

# Install OpenVLA dependencies
cd openvla && pip install -e . && cd ..

# Install LIBERO dependencies
cd LIBERO && pip install -e . && cd ..

# Install additional tools
pip install torchattacks matplotlib scipy
```

**Step 3: Download Models**
```python
# Download OpenVLA checkpoint
from openvla import load_openvla
model = load_openvla("openvla/openvla-7b")
```

**Step 4: Verify Setup**
```python
# Test that everything works
import torch
from openvla import OpenVLA

# Load model
model = OpenVLA.from_pretrained("openvla/openvla-7b")

# Test inference
dummy_image = torch.randn(1, 3, 224, 224)
dummy_instruction = "pick up the red block"
action = model(dummy_image, dummy_instruction)
print(f"Action shape: {action.shape}")  # Should be (1, 7)
```

### 7.3 Day 3-4: Encoder Sensitivity Test

**This is the critical validation experiment!**

**Objective**: Verify that SigLIP+DINOv2 encoders are sensitive to low-frequency perturbations.

**Step 1: Extract Visual Encoder**
```python
import torch
import torch.nn.functional as F

def extract_visual_encoder(openvla_model):
    """Extract the visual encoder from OpenVLA"""
    # OpenVLA uses Prismatic VLM which has SigLIP + DINOv2
    visual_encoder = openvla_model.vision_backbone
    return visual_encoder

encoder = extract_visual_encoder(model)
```

**Step 2: Implement DCT Functions**
```python
import torch
import numpy as np
from scipy.fftpack import dct, idct

def dct_2d(x):
    """2D DCT using scipy (for initial testing)"""
    # x: (B, C, H, W) tensor
    x_np = x.detach().cpu().numpy()
    result = np.zeros_like(x_np)
    for b in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            result[b, c] = dct(dct(x_np[b, c], axis=0, norm='ortho'), axis=1, norm='ortho')
    return torch.tensor(result, device=x.device, dtype=x.dtype)

def idct_2d(x):
    """2D IDCT using scipy"""
    x_np = x.detach().cpu().numpy()
    result = np.zeros_like(x_np)
    for b in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            result[b, c] = idct(idct(x_np[b, c], axis=0, norm='ortho'), axis=1, norm='ortho')
    return torch.tensor(result, device=x.device, dtype=x.dtype)

# Faster PyTorch implementation (use this for actual experiments)
def dct_2d_torch(x):
    """2D DCT using PyTorch FFT"""
    # Type-II DCT via FFT
    N = x.shape[-1]
    x_reorder = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    X = torch.fft.fft(x_reorder, dim=-1)
    # Apply DCT weights
    n = torch.arange(N, device=x.device)
    weights = 2 * torch.exp(-1j * np.pi * n / (2 * N))
    weights[0] = weights[0] / np.sqrt(2)
    X = X * weights
    return X.real
```

**Step 3: Create Frequency Band Masks**
```python
def create_frequency_mask(H, W, band='low', ratio=0.25):
    """
    Create mask for different frequency bands

    Args:
        H, W: Image dimensions
        band: 'low', 'mid', 'high', or 'all'
        ratio: What fraction of frequencies to include
    """
    mask = torch.zeros(H, W)

    if band == 'low':
        # Low frequencies are in top-left corner of DCT
        h_cutoff = int(H * ratio)
        w_cutoff = int(W * ratio)
        mask[:h_cutoff, :w_cutoff] = 1

    elif band == 'high':
        # High frequencies are away from top-left
        h_cutoff = int(H * (1 - ratio))
        w_cutoff = int(W * (1 - ratio))
        mask[h_cutoff:, w_cutoff:] = 1

    elif band == 'mid':
        # Middle frequencies
        h_low = int(H * 0.25)
        w_low = int(W * 0.25)
        h_high = int(H * 0.75)
        w_high = int(W * 0.75)
        mask[h_low:h_high, w_low:w_high] = 1
        mask[:h_low, :w_low] = 0

    elif band == 'all':
        mask = torch.ones(H, W)

    return mask
```

**Step 4: Run Sensitivity Experiment**
```python
def test_frequency_sensitivity(encoder, images, epsilon=16/255):
    """
    Test encoder sensitivity to different frequency perturbations

    Args:
        encoder: Visual encoder (SigLIP+DINOv2)
        images: Batch of test images (B, 3, H, W)
        epsilon: Perturbation magnitude

    Returns:
        Dictionary with feature deviations for each frequency band
    """
    results = {}
    H, W = images.shape[2], images.shape[3]

    # Get clean features
    with torch.no_grad():
        clean_features = encoder(images)

    for band in ['low', 'high', 'mid', 'all']:
        # Create frequency mask
        mask = create_frequency_mask(H, W, band=band).to(images.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Transform to DCT domain
        images_dct = dct_2d(images)

        # Add random perturbation only to selected frequencies
        noise = torch.randn_like(images_dct) * epsilon
        perturbed_dct = images_dct + noise * mask

        # Transform back
        perturbed_images = idct_2d(perturbed_dct)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Get perturbed features
        with torch.no_grad():
            perturbed_features = encoder(perturbed_images)

        # Compute feature deviation
        deviation = torch.norm(perturbed_features - clean_features, dim=-1).mean()
        results[band] = deviation.item()

        print(f"Band: {band:5s} | Feature Deviation: {deviation.item():.4f}")

    return results

# RUN THE EXPERIMENT
# Load some test images (you can use LIBERO images or any images)
test_images = torch.randn(10, 3, 224, 224).cuda()  # Replace with real images

results = test_frequency_sensitivity(encoder, test_images)
```

**Step 5: Interpret Results**

```
EXPECTED RESULTS (if hypothesis is correct):

Band: low   | Feature Deviation: 0.8532  <-- HIGHEST (ViT vulnerable)
Band: mid   | Feature Deviation: 0.4217
Band: high  | Feature Deviation: 0.2103  <-- LOWEST (ViT robust)
Band: all   | Feature Deviation: 0.9876

SUCCESS CRITERIA:
- Low-frequency deviation > High-frequency deviation
- Ratio (low/high) > 2.0 suggests strong exploitability

FAILURE CRITERIA:
- High-frequency deviation > Low-frequency deviation
- This would mean ViT vulnerability assumption is wrong for VLA encoders
```

### 7.4 Day 5-6: Basic Frequency Attack Implementation

**Only proceed if Day 3-4 shows positive results!**

**Step 1: Implement Low-Frequency Attack**
```python
class LowFrequencyVLAAttack:
    def __init__(self, model, epsilon=8/255, steps=50, freq_ratio=0.25):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.freq_ratio = freq_ratio

    def attack(self, image, instruction, targeted=False, target_action=None):
        """
        Low-frequency adversarial attack on VLA

        Args:
            image: Input image (1, 3, H, W)
            instruction: Language instruction string
            targeted: Whether to perform targeted attack
            target_action: Target action for targeted attack

        Returns:
            Adversarial image
        """
        H, W = image.shape[2], image.shape[3]

        # Create low-frequency mask
        mask = create_frequency_mask(H, W, band='low', ratio=self.freq_ratio)
        mask = mask.unsqueeze(0).unsqueeze(0).to(image.device)

        # Initialize
        image_dct = dct_2d(image)
        delta_f = torch.zeros_like(image_dct, requires_grad=True)

        alpha = self.epsilon / self.steps

        # Get clean action for reference
        with torch.no_grad():
            clean_action = self.model(image, instruction)

        for step in range(self.steps):
            # Apply perturbation (only low frequencies)
            perturbed_dct = image_dct + delta_f * mask
            adv_image = idct_2d(perturbed_dct)
            adv_image = torch.clamp(adv_image, 0, 1)

            # Make it differentiable
            adv_image.requires_grad_(True)

            # Get action prediction
            action = self.model(adv_image, instruction)

            # Compute loss
            if targeted:
                # Targeted: minimize distance to target action
                loss = F.mse_loss(action, target_action)
            else:
                # Untargeted: maximize distance from clean action
                loss = -F.mse_loss(action, clean_action)

            # Backward pass
            loss.backward()

            # Get gradient in frequency domain
            grad_spatial = adv_image.grad
            grad_dct = dct_2d(grad_spatial)

            # Update perturbation
            with torch.no_grad():
                delta_f = delta_f - alpha * torch.sign(grad_dct)

                # Project to epsilon ball in spatial domain
                spatial_pert = idct_2d(delta_f * mask)
                spatial_pert = torch.clamp(spatial_pert, -self.epsilon, self.epsilon)
                delta_f = dct_2d(spatial_pert)

            delta_f.requires_grad_(True)

        # Generate final adversarial image
        final_dct = image_dct + delta_f.detach() * mask
        adv_image = idct_2d(final_dct)
        adv_image = torch.clamp(adv_image, 0, 1)

        return adv_image

# Test the attack
attack = LowFrequencyVLAAttack(model, epsilon=8/255, steps=50)
adv_image = attack.attack(test_image, "pick up the red block")

# Measure effect
clean_action = model(test_image, "pick up the red block")
adv_action = model(adv_image, "pick up the red block")
print(f"Action deviation: {torch.norm(adv_action - clean_action).item():.4f}")
```

### 7.5 Day 7: Analysis & Decision Point

**Metrics to Compute**:
```python
def evaluate_attack(model, attack, test_images, instructions):
    """Comprehensive attack evaluation"""

    results = {
        'action_deviation': [],
        'success_rate': [],
        'psnr': [],
        'ssim': []
    }

    for image, instruction in zip(test_images, instructions):
        # Get clean action
        clean_action = model(image, instruction)

        # Generate adversarial
        adv_image = attack.attack(image, instruction)
        adv_action = model(adv_image, instruction)

        # Action deviation
        deviation = torch.norm(adv_action - clean_action).item()
        results['action_deviation'].append(deviation)

        # Success: deviation > threshold (you define based on task)
        success = deviation > 0.1  # Adjust threshold
        results['success_rate'].append(success)

        # Image quality
        psnr = compute_psnr(image, adv_image)
        ssim = compute_ssim(image, adv_image)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

    # Print summary
    print(f"Mean Action Deviation: {np.mean(results['action_deviation']):.4f}")
    print(f"Attack Success Rate: {np.mean(results['success_rate'])*100:.1f}%")
    print(f"Mean PSNR: {np.mean(results['psnr']):.2f} dB")
    print(f"Mean SSIM: {np.mean(results['ssim']):.4f}")

    return results
```

**Decision Criteria**:
```
GO CRITERIA (proceed to Week 2):
- Encoder sensitivity test shows low-freq > high-freq deviation
- Attack achieves > 50% success rate on test images
- Action deviation is meaningful (> 0.1 normalized)

NO-GO CRITERIA (pivot or abandon):
- Encoder shows opposite sensitivity (high-freq > low-freq)
- Attack success rate < 20%
- No meaningful action deviation despite visible perturbation

PIVOT OPTIONS:
- If high-freq more effective: Switch to high-frequency attack
- If neither works: Investigate phase attacks instead
- If attack works but weak: Increase perturbation budget
```

---

## 8. Visual Explanations

### 8.1 DCT Frequency Layout

```
DCT COEFFICIENT LAYOUT (224x224 image)

+------------------+------------------+
|                  |                  |
|   LOW FREQ       |    MID FREQ      |
|   (0,0)-(56,56)  |   (0,56)-(56,224)|
|                  |                  |
|   - Overall      |   - Medium       |
|     brightness   |     details      |
|   - Large shapes |   - Edges        |
|                  |                  |
+------------------+------------------+
|                  |                  |
|   MID FREQ       |   HIGH FREQ      |
|  (56,0)-(224,56) | (56,56)-(224,224)|
|                  |                  |
|   - Edges        |   - Fine texture |
|   - Medium       |   - Noise        |
|     details      |   - Sharp edges  |
|                  |                  |
+------------------+------------------+

DC Component (0,0): Average image intensity
Moving away from (0,0): Increasing frequency
```

### 8.2 Attack Pipeline Visualization

```
FREQUENCY DOMAIN ATTACK PIPELINE

Step 1: Original Image
+------------------+
|                  |
|    [Robot Arm    |
|     Scene]       |
|                  |
+------------------+
        |
        v (DCT Transform)

Step 2: DCT Coefficients
+------------------+
| XX               |  XX = Low freq (perturb these)
| XX     ...       |  .. = Mid freq
|     .........    |  ## = High freq (skip)
|   ...........### |
|   .........##### |
|   ......######## |
+------------------+
        |
        v (Add perturbation to low-freq only)

Step 3: Perturbed DCT
+------------------+
| XX+delta         |  <-- Only low-freq modified
| XX+delta  ...    |
|     .........    |  <-- Mid/high unchanged
|   ...........### |
|   .........##### |
|   ......######## |
+------------------+
        |
        v (IDCT Transform)

Step 4: Adversarial Image
+------------------+
|                  |
|   [Subtly        |
|    Different     |
|    Robot Scene]  |
|                  |
+------------------+
        |
        v (Feed to VLA)

Step 5: Wrong Action
Original: [0.1, 0.2, 0.3, 0.1, 0.0, 0.2, 1.0]  (pick up)
Adversarial: [0.5, -0.3, 0.8, 0.4, 0.2, -0.1, 0.0]  (wrong motion)
```

### 8.3 ViT vs CNN Sensitivity Visualization

```
FREQUENCY SENSITIVITY COMPARISON

        CNN Sensitivity              ViT Sensitivity

High    ████████████████████       Low     ████
        ████████████████████               ████
        ████████████████████               ████
        ████████████████████               ████████
        ████████████████                   ████████████
        ████████████                       ████████████████
        ████████                           ████████████████████
Low     ████                       High    ████████████████████

        Low     High                       Low     High
        Frequency                          Frequency

CNN: High-frequency perturbations cause more damage
ViT: Low-frequency perturbations cause more damage

VLA uses ViT --> Attack LOW frequencies!
```

### 8.4 Validation Experiment Flow

```
VALIDATION EXPERIMENT FLOWCHART

                    START
                      |
                      v
            +-------------------+
            | Setup Environment |
            | (Day 1-2)         |
            +-------------------+
                      |
                      v
            +-------------------+
            | Extract Visual    |
            | Encoder from VLA  |
            +-------------------+
                      |
                      v
            +-------------------+
            | Test Frequency    |
            | Sensitivity       |
            | (Day 3-4)         |
            +-------------------+
                      |
                      v
            +-------------------+
            | Low-freq >        |
            | High-freq?        |
            +-------------------+
                /           \
               /             \
          YES /               \ NO
             /                 \
            v                   v
    +-------------+      +--------------+
    | Implement   |      | PIVOT:       |
    | Low-Freq    |      | Try high-freq|
    | Attack      |      | or phase     |
    | (Day 5-6)   |      | attack       |
    +-------------+      +--------------+
            |
            v
    +-------------------+
    | Test on VLA       |
    | Action Deviation  |
    +-------------------+
            |
            v
    +-------------------+
    | Success Rate      |
    | > 50%?            |
    +-------------------+
           /           \
          /             \
     YES /               \ NO
        /                 \
       v                   v
+-------------+     +--------------+
| PROCEED to  |     | Analyze why  |
| Week 2:     |     | - Epsilon?   |
| Full LIBERO |     | - Steps?     |
| Evaluation  |     | - Mask size? |
+-------------+     +--------------+
```

---

## Summary: Quick Reference Card

```
+===========================================================+
|           VLA FREQUENCY ATTACK QUICK REFERENCE            |
+===========================================================+

WHAT TO ATTACK:
- Visual input (images)
- Focus on LOW-frequency components
- Use DCT transform

KEY PARAMETERS:
- Epsilon: 8/255 to 16/255
- Steps: 50-100
- Frequency ratio: 0.25 (bottom-left 25% of DCT)

CODE RESOURCES:
- DCT: github.com/yuyang-long/SSA/blob/master/dct.py
- VLA: github.com/openvla/openvla
- Baselines: pip install torchattacks

VALIDATION CHECKLIST:
[ ] Environment setup complete
[ ] Visual encoder extracted
[ ] Frequency sensitivity test shows low > high
[ ] Basic attack achieves > 50% success
[ ] Action deviation is meaningful

SUCCESS SIGNAL:
- Low-freq perturbation causes 2x+ more feature deviation than high-freq
- Attack causes significant action changes with imperceptible perturbation

FAILURE SIGNAL:
- High-freq more effective than low-freq (theory mismatch)
- No action deviation despite large perturbation (VLA robust)
+===========================================================+
```

---

*Document prepared for research validation. Adapt code snippets to your specific environment.*
