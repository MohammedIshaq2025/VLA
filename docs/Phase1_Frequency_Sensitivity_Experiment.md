# Phase 1: Frequency Sensitivity Experiment - Complete Guide

## Overview

This phase validates the core hypothesis: **VLA models (using ViT-based encoders) are more vulnerable to low-frequency perturbations than high-frequency ones.**

**Timeline**: This is a validation experiment before full attack implementation.
**Goal**: Determine if frequency-domain attacks are viable for VLA.

---

## 1. Theoretical Foundation

### Why Low-Frequency for ViTs?

Unlike CNNs that use small convolutional kernels (sensitive to local high-frequency patterns), Vision Transformers:

1. **Use patch embeddings** (16×16 or 14×14 pixels) - averages out high-frequency noise within patches
2. **Global self-attention** - captures long-range dependencies (low-frequency structure)
3. **Positional encodings** - encode spatial relationships at image-level, not pixel-level

**Key Papers Supporting This**:
- Yin et al. (NeurIPS 2019): "Fourier Perspective on Model Robustness" - ViTs rely more on low-frequency components
- Kim et al. (WACV 2024): "Low-Frequency Adversarial Attack" - explicitly targets ViT vulnerability
- SSA (ECCV 2022): Shows spectrum-based attacks transfer better across architectures

### DCT Frequency Bands

```
DCT Coefficient Matrix (8×8 example):
┌─────────────────────────────────┐
│ DC  L   L   M   M   H   H   H  │  DC = (0,0) - mean/average
│ L   L   M   M   H   H   H   H  │  L  = Low frequency (smooth variations)
│ L   M   M   H   H   H   H   H  │  M  = Mid frequency (edges, structure)
│ M   M   H   H   H   H   H   H  │  H  = High frequency (fine details, noise)
│ M   H   H   H   H   H   H   H  │
│ H   H   H   H   H   H   H   H  │  Distance from (0,0) = frequency
│ H   H   H   H   H   H   H   H  │
│ H   H   H   H   H   H   H   H  │
└─────────────────────────────────┘
```

---

## 2. Project Structure

Create this structure in your project directory:

```bash
cd /data1/ma1/Ishaq/VLA_Frequency_Attack/

mkdir -p src/utils src/models experiments scripts results/phase1
touch src/__init__.py src/utils/__init__.py src/models/__init__.py
```

Final structure:
```
VLA_Frequency_Attack/
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── dct.py              # DCT/IDCT implementation
│   └── models/
│       ├── __init__.py
│       └── vla_wrapper.py      # OpenVLA wrapper
├── experiments/
│   └── phase1_sensitivity.py   # Main experiment
├── scripts/
│   └── run_phase1.slurm        # SLURM job script
└── results/
    └── phase1/                 # Output directory
```

---

## 3. Core Implementation Files

### 3.1 DCT Utilities (`src/utils/dct.py`)

This is a **corrected** implementation - the original had incomplete 2D DCT.

```python
"""
Discrete Cosine Transform utilities for frequency-domain attacks.
Corrected implementation with proper 2D DCT and orthonormal scaling.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def create_dct_matrix(n: int, device: torch.device = None) -> torch.Tensor:
    """
    Create orthonormal DCT-II matrix.

    The DCT-II matrix C satisfies: C @ C.T = I (orthonormal)
    DCT coefficients: X = C @ x @ C.T
    Inverse: x = C.T @ X @ C

    Args:
        n: Size of the DCT matrix (n x n)
        device: Target device for tensor

    Returns:
        DCT matrix of shape (n, n)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create index grids
    i = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)  # (n, 1)
    j = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0)  # (1, n)

    # DCT-II basis: cos(pi * i * (2j + 1) / (2n))
    C = torch.cos(np.pi * i * (2 * j + 1) / (2 * n))

    # Orthonormal scaling
    C[0, :] *= np.sqrt(1 / n)
    C[1:, :] *= np.sqrt(2 / n)

    return C


class DCT2D:
    """
    2D Discrete Cosine Transform for image processing.

    Supports batched operations on GPU with gradient tracking.
    """

    def __init__(self, height: int, width: int, device: torch.device = None):
        """
        Initialize DCT matrices for given image dimensions.

        Args:
            height: Image height
            width: Image width
            device: Target device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.height = height
        self.width = width

        # Pre-compute DCT matrices (cached for efficiency)
        self.C_h = create_dct_matrix(height, self.device)  # (H, H)
        self.C_w = create_dct_matrix(width, self.device)   # (W, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT to input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            DCT coefficients of shape (B, C, H, W)
        """
        # Move to correct device if needed
        x = x.to(self.device)

        # 2D DCT: X = C_h @ x @ C_w.T
        # Apply along height: C_h @ x
        x_dct = torch.einsum('hH,bcHW->bchW', self.C_h, x)
        # Apply along width: result @ C_w.T
        x_dct = torch.einsum('bchW,wW->bchw', x_dct, self.C_w)

        return x_dct

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse 2D DCT (IDCT) to frequency coefficients.

        Args:
            X: DCT coefficients of shape (B, C, H, W)

        Returns:
            Reconstructed image of shape (B, C, H, W)
        """
        X = X.to(self.device)

        # 2D IDCT: x = C_h.T @ X @ C_w
        # Apply along height: C_h.T @ X
        x = torch.einsum('Hh,bchW->bcHW', self.C_h, X)
        # Apply along width: result @ C_w
        x = torch.einsum('bcHW,Ww->bcHw', x, self.C_w)

        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(x)


def create_frequency_mask(
    height: int,
    width: int,
    band: str = 'low',
    ratio: float = 0.25,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create a frequency band mask for DCT coefficients.

    Args:
        height: Image height
        width: Image width
        band: 'low', 'mid', or 'high'
        ratio: Fraction of frequencies to include (0.0 to 1.0)
        device: Target device

    Returns:
        Binary mask of shape (1, 1, H, W)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create distance matrix from DC component (top-left corner)
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Normalized distance (0 at DC, 1 at max frequency corner)
    max_dist = np.sqrt(height**2 + width**2)
    dist = torch.sqrt(yy**2 + xx**2) / max_dist

    # Create mask based on band
    if band == 'low':
        mask = (dist <= ratio).float()
    elif band == 'high':
        mask = (dist >= (1 - ratio)).float()
    elif band == 'mid':
        low_thresh = ratio / 2
        high_thresh = 1 - ratio / 2
        mask = ((dist > low_thresh) & (dist < high_thresh)).float()
    else:
        raise ValueError(f"Unknown band: {band}. Use 'low', 'mid', or 'high'")

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def visualize_frequency_mask(mask: torch.Tensor, title: str = "Frequency Mask"):
    """
    Visualize a frequency mask (for debugging).

    Args:
        mask: Mask tensor of shape (1, 1, H, W)
        title: Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Frequency (horizontal)')
    plt.ylabel('Frequency (vertical)')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


# Validation function
def validate_dct_implementation():
    """
    Validate DCT implementation by checking reconstruction error.

    Returns:
        True if implementation is correct
    """
    print("Validating DCT implementation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create random test image
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W, device=device)

    # Initialize DCT
    dct = DCT2D(H, W, device)

    # Forward and inverse
    X = dct.forward(x)
    x_reconstructed = dct.inverse(X)

    # Check reconstruction error
    error = torch.abs(x - x_reconstructed).max().item()
    print(f"Max reconstruction error: {error:.2e}")

    if error < 1e-5:
        print("DCT implementation PASSED")
        return True
    else:
        print("DCT implementation FAILED - reconstruction error too high")
        return False


if __name__ == "__main__":
    validate_dct_implementation()
```

### 3.2 OpenVLA Wrapper (`src/models/vla_wrapper.py`)

**Corrected** - uses proper HuggingFace transformers API.

```python
"""
OpenVLA model wrapper for adversarial attacks.
Corrected to use HuggingFace transformers API.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, Tuple, Optional, List
import numpy as np


class OpenVLAWrapper(nn.Module):
    """
    Wrapper for OpenVLA model that exposes vision encoder and action prediction.

    OpenVLA Architecture:
    - Vision Encoder: SigLIP (ViT-SO400M) + DINOv2 (ViT-L/14) fused
    - Projector: 2-layer MLP
    - LLM: Llama 2 7B
    - Action Head: 256 discrete bins per DoF, 7 DoF total
    """

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize OpenVLA wrapper.

        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Target device
            torch_dtype: Model precision (bfloat16 recommended for H200)
        """
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading OpenVLA from: {model_path}")
        print(f"Device: {device}, Dtype: {torch_dtype}")

        # Load processor (handles image preprocessing and tokenization)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Set to eval mode
        self.model.eval()

        # Get image size from processor
        self.image_size = self._get_image_size()
        print(f"Image size: {self.image_size}")

    def _get_image_size(self) -> int:
        """Extract expected image size from processor."""
        # OpenVLA typically uses 224x224
        try:
            return self.processor.image_processor.size['height']
        except:
            return 224  # Default

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image tensor for model input.

        Args:
            image: Raw image tensor (B, C, H, W) in [0, 1] range

        Returns:
            Preprocessed image tensor
        """
        # OpenVLA expects images normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

        # Normalize
        image_normalized = (image - mean) / std

        return image_normalized.to(self.torch_dtype)

    def get_vision_features(
        self,
        image: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Extract vision encoder features.

        Args:
            image: Preprocessed image tensor (B, C, H, W)
            return_all_layers: If True, return features from all layers

        Returns:
            Vision features tensor
        """
        # Get the vision backbone
        # OpenVLA uses a fused encoder (SigLIP + DINOv2)
        vision_backbone = self.model.vision_backbone if hasattr(self.model, 'vision_backbone') else None

        if vision_backbone is None:
            # Try alternative access patterns
            for name in ['vision_tower', 'visual', 'image_encoder', 'vit']:
                if hasattr(self.model, name):
                    vision_backbone = getattr(self.model, name)
                    break

        if vision_backbone is None:
            raise AttributeError(
                "Could not find vision backbone. "
                "Model structure may have changed. "
                "Available attributes: " + str([n for n in dir(self.model) if not n.startswith('_')])
            )

        with torch.no_grad():
            features = vision_backbone(image)

        return features

    def predict_action(
        self,
        image: torch.Tensor,
        instruction: str,
        unnorm_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict robot action from image and instruction.

        Args:
            image: Image tensor (B, C, H, W) in [0, 1] range or PIL Image
            instruction: Natural language instruction
            unnorm_key: Key for action unnormalization (dataset-specific)

        Returns:
            Predicted action array (7,) - [x, y, z, roll, pitch, yaw, gripper]
        """
        # Process inputs
        inputs = self.processor(
            text=instruction,
            images=image,
            return_tensors="pt"
        ).to(self.device, dtype=self.torch_dtype)

        # Generate action tokens
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False
            )

        return action

    def forward_with_grad(
        self,
        image: torch.Tensor,
        instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient tracking for adversarial attacks.

        Args:
            image: Image tensor (B, C, H, W) requiring grad
            instruction: Natural language instruction

        Returns:
            Tuple of (logits, action_tokens)
        """
        # Ensure gradient tracking
        assert image.requires_grad, "Image must require gradients"

        # Process text (no gradients needed)
        text_inputs = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get image features with gradients
        image_processed = self.preprocess_image(image)

        # Forward through model
        outputs = self.model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=image_processed,
            return_dict=True
        )

        return outputs.logits, outputs

    def compute_action_loss(
        self,
        image: torch.Tensor,
        instruction: str,
        target_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for adversarial optimization.

        Args:
            image: Image tensor requiring gradients
            instruction: Natural language instruction
            target_action: Target action for targeted attack (optional)

        Returns:
            Loss tensor (scalar)
        """
        logits, _ = self.forward_with_grad(image, instruction)

        if target_action is not None:
            # Targeted attack: minimize distance to target
            # Convert target action to token indices
            # This is simplified - actual implementation needs proper tokenization
            loss = -torch.mean(logits)  # Placeholder
        else:
            # Untargeted attack: maximize prediction uncertainty
            # Use negative entropy or similar
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            loss = -torch.mean(entropy)  # Maximize entropy

        return loss


def test_wrapper():
    """Test the OpenVLA wrapper."""
    import sys

    # Use your local checkpoint path
    model_path = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/"

    print("Testing OpenVLA wrapper...")

    try:
        wrapper = OpenVLAWrapper(model_path=model_path)
        print("Model loaded successfully!")

        # Test with dummy input
        dummy_image = torch.randn(1, 3, 224, 224, device='cuda')
        instruction = "pick up the red block"

        # Test action prediction
        action = wrapper.predict_action(dummy_image, instruction)
        print(f"Predicted action shape: {action.shape}")
        print(f"Predicted action: {action}")

        print("Wrapper test PASSED")
        return True

    except Exception as e:
        print(f"Wrapper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_wrapper()
```

### 3.3 Phase 1 Experiment (`experiments/phase1_sensitivity.py`)

This is the main experiment script that tests the hypothesis.

```python
"""
Phase 1: Frequency Sensitivity Experiment

Tests whether OpenVLA is more vulnerable to low-frequency or high-frequency perturbations.
This validates our core hypothesis before full attack implementation.

Expected outcome: Low-frequency perturbations cause larger action deviations.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dct import DCT2D, create_frequency_mask, validate_dct_implementation
from src.models.vla_wrapper import OpenVLAWrapper


class FrequencySensitivityExperiment:
    """
    Measures VLA sensitivity to perturbations in different frequency bands.

    Methodology:
    1. Take clean images
    2. Add perturbations ONLY in specific frequency bands (low/mid/high)
    3. Measure action deviation from clean prediction
    4. Compare deviations across frequency bands
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        epsilon: float = 8/255,  # L-inf bound (standard adversarial)
        num_samples: int = 100,
        seed: int = 42
    ):
        """
        Initialize experiment.

        Args:
            model_path: Path to OpenVLA checkpoint
            output_dir: Directory for results
            epsilon: Perturbation magnitude (L-inf norm)
            num_samples: Number of random samples to test
            seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epsilon = epsilon
        self.num_samples = num_samples
        self.seed = seed

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Image dimensions (OpenVLA standard)
        self.img_size = 224

        # Initialize DCT
        self.dct = DCT2D(self.img_size, self.img_size, self.device)

        # Frequency bands to test
        self.bands = ['low', 'mid', 'high']
        self.band_ratios = {
            'low': 0.15,   # Bottom 15% of frequencies
            'mid': 0.70,   # Middle 70%
            'high': 0.15   # Top 15% of frequencies
        }

        # Pre-create masks
        self.masks = {
            band: create_frequency_mask(
                self.img_size, self.img_size,
                band=band,
                ratio=self.band_ratios[band],
                device=self.device
            )
            for band in self.bands
        }

        # Test instructions (variety of tasks)
        self.instructions = [
            "pick up the red block",
            "move the arm to the left",
            "place the object on the table",
            "push the blue cube forward",
            "grasp the handle and pull"
        ]

        # Results storage
        self.results = {band: [] for band in self.bands}

    def load_model(self):
        """Load the VLA model."""
        print("Loading OpenVLA model...")
        self.model = OpenVLAWrapper(
            model_path=self.model_path,
            device=str(self.device),
            torch_dtype=torch.bfloat16
        )
        print("Model loaded successfully!")

    def generate_random_image(self) -> torch.Tensor:
        """
        Generate a random test image.

        In practice, you'd load real images from LIBERO dataset.
        For initial validation, random images test the mechanism.

        Returns:
            Image tensor (1, 3, 224, 224) in [0, 1] range
        """
        # Random image in valid range
        image = torch.rand(1, 3, self.img_size, self.img_size, device=self.device)
        return image

    def add_frequency_perturbation(
        self,
        image: torch.Tensor,
        band: str
    ) -> torch.Tensor:
        """
        Add perturbation only in specified frequency band.

        Args:
            image: Clean image (B, C, H, W)
            band: Frequency band ('low', 'mid', 'high')

        Returns:
            Perturbed image (B, C, H, W)
        """
        # Transform to frequency domain
        image_dct = self.dct.forward(image)

        # Generate random perturbation in frequency domain
        perturbation_dct = torch.randn_like(image_dct) * self.epsilon

        # Apply band mask - only perturb selected frequencies
        mask = self.masks[band]
        perturbation_dct = perturbation_dct * mask

        # Add perturbation in frequency domain
        perturbed_dct = image_dct + perturbation_dct

        # Transform back to spatial domain
        perturbed_image = self.dct.inverse(perturbed_dct)

        # Clamp to valid range [0, 1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    def compute_action_deviation(
        self,
        action_clean: np.ndarray,
        action_perturbed: np.ndarray
    ) -> dict:
        """
        Compute deviation metrics between clean and perturbed actions.

        Args:
            action_clean: Clean action prediction (7,)
            action_perturbed: Perturbed action prediction (7,)

        Returns:
            Dictionary of deviation metrics
        """
        diff = action_perturbed - action_clean

        return {
            'l2_norm': float(np.linalg.norm(diff)),
            'l1_norm': float(np.sum(np.abs(diff))),
            'linf_norm': float(np.max(np.abs(diff))),
            'mean_abs_diff': float(np.mean(np.abs(diff))),
            'per_dim_diff': diff.tolist()
        }

    def run_single_sample(self, sample_idx: int) -> dict:
        """
        Run experiment on a single sample.

        Args:
            sample_idx: Sample index

        Returns:
            Results dictionary for this sample
        """
        # Generate random image
        image = self.generate_random_image()

        # Select instruction
        instruction = self.instructions[sample_idx % len(self.instructions)]

        # Get clean action prediction
        try:
            action_clean = self.model.predict_action(image, instruction)
        except Exception as e:
            print(f"Error getting clean action: {e}")
            return None

        sample_results = {'clean_action': action_clean.tolist()}

        # Test each frequency band
        for band in self.bands:
            # Add frequency-specific perturbation
            perturbed_image = self.add_frequency_perturbation(image, band)

            # Get perturbed action
            try:
                action_perturbed = self.model.predict_action(perturbed_image, instruction)
            except Exception as e:
                print(f"Error getting {band} action: {e}")
                continue

            # Compute deviation
            deviation = self.compute_action_deviation(action_clean, action_perturbed)

            sample_results[band] = {
                'perturbed_action': action_perturbed.tolist(),
                'deviation': deviation
            }

            # Store L2 deviation for summary statistics
            self.results[band].append(deviation['l2_norm'])

        return sample_results

    def run_experiment(self):
        """Run the full experiment."""
        print(f"\n{'='*60}")
        print("Phase 1: Frequency Sensitivity Experiment")
        print(f"{'='*60}")
        print(f"Epsilon: {self.epsilon:.4f}")
        print(f"Num samples: {self.num_samples}")
        print(f"Frequency bands: {self.bands}")
        print(f"{'='*60}\n")

        # Validate DCT first
        if not validate_dct_implementation():
            raise RuntimeError("DCT validation failed!")

        # Load model
        self.load_model()

        # Visualize masks
        self._visualize_masks()

        # Run samples
        all_results = []
        for i in tqdm(range(self.num_samples), desc="Running samples"):
            result = self.run_single_sample(i)
            if result is not None:
                all_results.append(result)

        # Compute summary statistics
        summary = self._compute_summary()

        # Save results
        self._save_results(all_results, summary)

        # Generate plots
        self._generate_plots(summary)

        # Print decision
        self._print_decision(summary)

        return summary

    def _visualize_masks(self):
        """Visualize the frequency masks."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, band in zip(axes, self.bands):
            mask = self.masks[band].squeeze().cpu().numpy()
            im = ax.imshow(mask, cmap='hot')
            ax.set_title(f'{band.capitalize()} Frequency Mask')
            ax.set_xlabel('Frequency (horizontal)')
            ax.set_ylabel('Frequency (vertical)')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_masks.png', dpi=150)
        plt.close()
        print(f"Saved frequency masks to {self.output_dir / 'frequency_masks.png'}")

    def _compute_summary(self) -> dict:
        """Compute summary statistics."""
        summary = {}

        for band in self.bands:
            deviations = self.results[band]
            if len(deviations) > 0:
                summary[band] = {
                    'mean': float(np.mean(deviations)),
                    'std': float(np.std(deviations)),
                    'median': float(np.median(deviations)),
                    'min': float(np.min(deviations)),
                    'max': float(np.max(deviations)),
                    'count': len(deviations)
                }
            else:
                summary[band] = None

        # Compute ratios
        if summary['low'] and summary['high']:
            summary['low_to_high_ratio'] = summary['low']['mean'] / (summary['high']['mean'] + 1e-8)

        return summary

    def _save_results(self, all_results: list, summary: dict):
        """Save results to files."""
        # Save detailed results
        results_file = self.output_dir / 'detailed_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved detailed results to {results_file}")

        # Save summary
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")

    def _generate_plots(self, summary: dict):
        """Generate visualization plots."""
        # Bar plot of mean deviations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Mean deviation by band
        bands = self.bands
        means = [summary[b]['mean'] if summary[b] else 0 for b in bands]
        stds = [summary[b]['std'] if summary[b] else 0 for b in bands]

        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
        axes[0].bar(bands, means, yerr=stds, color=colors, capsize=5)
        axes[0].set_ylabel('Mean L2 Action Deviation')
        axes[0].set_xlabel('Frequency Band')
        axes[0].set_title('VLA Sensitivity by Frequency Band')
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Distribution box plot
        data = [self.results[b] for b in bands]
        bp = axes[1].boxplot(data, labels=bands, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('L2 Action Deviation')
        axes[1].set_xlabel('Frequency Band')
        axes[1].set_title('Distribution of Deviations')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_comparison.png', dpi=150)
        plt.close()
        print(f"Saved comparison plot to {self.output_dir / 'sensitivity_comparison.png'}")

    def _print_decision(self, summary: dict):
        """Print decision based on results."""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)

        for band in self.bands:
            if summary[band]:
                print(f"{band.upper():8s}: mean={summary[band]['mean']:.4f}, std={summary[band]['std']:.4f}")

        if 'low_to_high_ratio' in summary:
            ratio = summary['low_to_high_ratio']
            print(f"\nLow/High Ratio: {ratio:.2f}")

            print("\n" + "-"*60)
            print("DECISION:")
            print("-"*60)

            if ratio > 1.5:
                print("GO_LOW_FREQ: Low-frequency attacks are significantly more effective!")
                print("-> Proceed with DCT-based low-frequency attack implementation")
                print("-> Focus perturbation budget on low-frequency bands")
                print("-> Expected to achieve better transferability")
            elif ratio < 0.67:
                print("PIVOT_HIGH_FREQ: High-frequency attacks are more effective")
                print("-> Consider traditional high-frequency perturbation methods")
                print("-> VLA may behave more like CNN than typical ViT")
                print("-> Investigate architecture details")
            else:
                print("INVESTIGATE: No clear frequency preference detected")
                print("-> Try different band ratios")
                print("-> Test with real LIBERO images instead of random")
                print("-> Consider hybrid frequency approach")

        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Frequency Sensitivity Experiment')

    parser.add_argument(
        '--model_path',
        type=str,
        default='/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/',
        help='Path to OpenVLA checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/phase1',
        help='Output directory for results'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=8/255,
        help='Perturbation magnitude (L-inf)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Run experiment
    experiment = FrequencySensitivityExperiment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        num_samples=args.num_samples,
        seed=args.seed
    )

    summary = experiment.run_experiment()

    print("\nExperiment complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
```

---

## 4. SLURM Job Script

### `scripts/run_phase1.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=phase1_freq
#SBATCH --output=results/phase1/slurm_%j.out
#SBATCH --error=results/phase1/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# =============================================================================
# Phase 1: Frequency Sensitivity Experiment
# Expected runtime: 30-60 minutes depending on num_samples
# =============================================================================

echo "============================================"
echo "Starting Phase 1: Frequency Sensitivity"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Activate environment
source /data1/ma1/envs/upa-vla/bin/activate

# Set project root
PROJECT_ROOT="/data1/ma1/Ishaq/VLA_Frequency_Attack"
cd $PROJECT_ROOT

# Create output directory
mkdir -p results/phase1

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run DCT validation first
echo "Running DCT validation..."
python -c "from src.utils.dct import validate_dct_implementation; validate_dct_implementation()"

if [ $? -ne 0 ]; then
    echo "DCT validation failed! Exiting."
    exit 1
fi

echo ""
echo "Starting main experiment..."
echo ""

# Run experiment
python experiments/phase1_sensitivity.py \
    --model_path /data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/ \
    --output_dir results/phase1 \
    --epsilon 0.031372549  \
    --num_samples 100 \
    --seed 42

echo ""
echo "============================================"
echo "Experiment complete!"
echo "Date: $(date)"
echo "============================================"

# Display results
echo ""
echo "Summary:"
cat results/phase1/summary.json
```

---

## 5. Quick Start Commands

After creating all files, run these commands:

```bash
# 1. Navigate to project
cd /data1/ma1/Ishaq/VLA_Frequency_Attack

# 2. Activate environment
source /data1/ma1/envs/upa-vla/bin/activate

# 3. Verify structure
ls -la src/utils/ src/models/ experiments/ scripts/

# 4. Test DCT implementation
python src/utils/dct.py

# 5. Test model wrapper (optional, uses GPU)
python src/models/vla_wrapper.py

# 6. Submit job
sbatch scripts/run_phase1.slurm

# 7. Monitor job
squeue -u $USER
tail -f results/phase1/slurm_*.out
```

---

## 6. Expected Outputs

After the experiment completes, you should have:

```
results/phase1/
├── slurm_XXXXX.out          # Job output log
├── slurm_XXXXX.err          # Job error log
├── frequency_masks.png      # Visualization of frequency bands
├── sensitivity_comparison.png  # Main results plot
├── detailed_results.json    # Per-sample results
└── summary.json             # Summary statistics
```

### Sample `summary.json`:
```json
{
  "low": {
    "mean": 0.245,
    "std": 0.089,
    "median": 0.231,
    "min": 0.067,
    "max": 0.512,
    "count": 100
  },
  "mid": {
    "mean": 0.156,
    "std": 0.072,
    "median": 0.148,
    "min": 0.034,
    "max": 0.398,
    "count": 100
  },
  "high": {
    "mean": 0.089,
    "std": 0.045,
    "median": 0.082,
    "min": 0.012,
    "max": 0.234,
    "count": 100
  },
  "low_to_high_ratio": 2.75
}
```

---

## 7. Decision Matrix

Based on the `low_to_high_ratio` in results:

| Ratio | Decision | Next Step |
|-------|----------|-----------|
| > 1.5 | **GO_LOW_FREQ** | Proceed to Phase 2: Implement DCT-based attack |
| 0.67 - 1.5 | **INVESTIGATE** | Try different band ratios, use real images |
| < 0.67 | **PIVOT_HIGH_FREQ** | Re-evaluate, consider traditional attacks |

---

## 8. Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
```python
# Reduce batch processing, use smaller num_samples first
python experiments/phase1_sensitivity.py --num_samples 10
```

**2. "Could not find vision backbone"**
```python
# The model structure may differ. Debug with:
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
print([n for n in dir(model) if not n.startswith('_')])
```

**3. "DCT validation failed"**
```python
# Check for NaN/Inf issues
python -c "
import torch
x = torch.randn(1, 3, 224, 224, device='cuda')
print(f'Input has NaN: {torch.isnan(x).any()}')
print(f'Input has Inf: {torch.isinf(x).any()}')
"
```

**4. "ModuleNotFoundError: No module named 'src'"**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/data1/ma1/Ishaq/VLA_Frequency_Attack:$PYTHONPATH
```

---

## 9. Next Steps (Phase 2 Preview)

If Phase 1 shows **GO_LOW_FREQ** (ratio > 1.5):

1. **Phase 2**: Implement gradient-based DCT attack
   - Optimize perturbation in frequency domain
   - Use PGD with DCT projection
   - Compare with baseline spatial attacks

2. **Phase 3**: Test transferability
   - Attack OpenVLA, test on RT-2, Octo
   - Measure transfer success rate

3. **Phase 4**: Full evaluation on LIBERO
   - Task success rate degradation
   - Compare with existing VLA attacks

---

## 10. References

1. **SSA (ECCV 2022)**: "Frequency Domain Model Augmentation for Adversarial Attack" - https://github.com/yuyang-long/SSA
2. **Kim et al. (WACV 2024)**: "Low-Frequency Adversarial Attack against Vision Transformers"
3. **Yin et al. (NeurIPS 2019)**: "A Fourier Perspective on Model Robustness in Computer Vision"
4. **OpenVLA**: https://github.com/openvla/openvla
