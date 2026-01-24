"""
Discrete Cosine Transform utilities for frequency-domain attacks.

This module provides a high-level wrapper around the existing DCT implementation
from SSA (dct_utils.py) with additional utilities for frequency band manipulation.

Note: DCT operations require float32 precision. BFloat16 tensors are automatically
converted to float32 for DCT/IDCT operations and converted back afterward.
"""

import torch
import numpy as np
from typing import Tuple, Optional

# Import existing DCT functions from dct_utils
from src.utils.dct_utils import dct_2d, idct_2d


class DCT2D:
    """
    2D Discrete Cosine Transform wrapper for image processing.

    Wraps the existing dct_2d/idct_2d functions from SSA with a clean interface
    that supports batched operations on GPU with gradient tracking.

    Note: Automatically handles dtype conversion - bfloat16 inputs are converted
    to float32 for FFT operations and converted back afterward.
    """

    def __init__(self, height: int, width: int, device: torch.device = None):
        """
        Initialize DCT for given image dimensions.

        Args:
            height: Image height
            width: Image width
            device: Target device (defaults to CUDA if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.height = height
        self.width = width

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

        # Store original dtype for later conversion
        original_dtype = x.dtype

        # Convert to float32 if needed (FFT doesn't support bfloat16)
        if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
            x = x.float()

        # Apply 2D DCT using existing implementation
        x_dct = dct_2d(x, norm='ortho')

        # Convert back to original dtype
        if original_dtype != x_dct.dtype:
            x_dct = x_dct.to(original_dtype)

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

        # Store original dtype for later conversion
        original_dtype = X.dtype

        # Convert to float32 if needed (FFT doesn't support bfloat16)
        if X.dtype == torch.bfloat16 or X.dtype == torch.float16:
            X = X.float()

        # Apply 2D IDCT using existing implementation
        x = idct_2d(X, norm='ortho')

        # Convert back to original dtype
        if original_dtype != x.dtype:
            x = x.to(original_dtype)

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

    The mask is based on the distance from the DC component (top-left corner)
    in the frequency domain. Different bands capture different image features:
    - Low frequency: Overall shape, color gradients (effective for ViTs)
    - Mid frequency: Edges, structure
    - High frequency: Fine details, texture (effective for CNNs)

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
    # DC component is at (0, 0) in DCT coefficient matrix
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Normalized distance (0 at DC, 1 at max frequency corner)
    max_dist = np.sqrt(height**2 + width**2)
    dist = torch.sqrt(yy**2 + xx**2) / max_dist

    # Create mask based on band
    if band == 'low':
        # Keep only low frequencies (close to DC)
        mask = (dist <= ratio).float()
    elif band == 'high':
        # Keep only high frequencies (far from DC)
        mask = (dist >= (1 - ratio)).float()
    elif band == 'mid':
        # Keep middle frequencies
        low_thresh = ratio / 2
        high_thresh = 1 - ratio / 2
        mask = ((dist > low_thresh) & (dist < high_thresh)).float()
    else:
        raise ValueError(f"Unknown band: {band}. Use 'low', 'mid', or 'high'")

    # Add batch and channel dimensions: (1, 1, H, W)
    return mask.unsqueeze(0).unsqueeze(0)


def visualize_frequency_mask(mask: torch.Tensor, title: str = "Frequency Mask", save_path: str = None):
    """
    Visualize a frequency mask (for debugging and documentation).

    Args:
        mask: Mask tensor of shape (1, 1, H, W)
        title: Plot title
        save_path: Path to save plot (if None, uses title as filename)
    """
    try:
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = f'{title.lower().replace(" ", "_")}.png'

        plt.figure(figsize=(6, 6))
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Frequency (horizontal)')
        plt.ylabel('Frequency (vertical)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[OK] Saved frequency mask visualization to {save_path}")
    except ImportError:
        print("[WARNING] matplotlib not available, skipping visualization")


def validate_dct_implementation() -> bool:
    """
    Validate DCT implementation by checking reconstruction error.

    Tests that IDCT(DCT(x)) == x within numerical precision.
    Also tests bfloat16 conversion handling.

    Returns:
        True if implementation is correct (error < 1e-5)
    """
    print("="*70)
    print("Validating DCT implementation...")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create random test image
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W, device=device)

    # Initialize DCT
    dct = DCT2D(H, W, device)

    # Test 1: float32 roundtrip
    print(f"\nTesting DCT/IDCT roundtrip on {(B, C, H, W)} tensor (float32)...")
    X = dct.forward(x)
    x_reconstructed = dct.inverse(X)

    error = torch.abs(x - x_reconstructed).max().item()
    print(f"Max reconstruction error (float32): {error:.2e}")

    # Verify dimensions preserved
    assert x_reconstructed.shape == x.shape, f"Shape mismatch: {x_reconstructed.shape} != {x.shape}"
    print(f"Shape preserved: {x.shape}")

    # Verify device preserved
    assert x_reconstructed.device == x.device, f"Device mismatch: {x_reconstructed.device} != {x.device}"
    print(f"Device preserved: {device}")

    # Test 2: bfloat16 roundtrip (important for VLA attacks)
    print(f"\nTesting DCT/IDCT roundtrip with bfloat16...")
    x_bf16 = x.to(torch.bfloat16)
    X_bf16 = dct.forward(x_bf16)
    x_rec_bf16 = dct.inverse(X_bf16)

    # Note: bfloat16 has lower precision, so we use a higher tolerance
    error_bf16 = torch.abs(x_bf16 - x_rec_bf16).max().item()
    print(f"Max reconstruction error (bfloat16): {error_bf16:.2e}")
    print(f"Output dtype: {x_rec_bf16.dtype}")

    # Check error threshold
    if error < 1e-5 and error_bf16 < 1e-2:  # bfloat16 has lower precision
        print("\n" + "="*70)
        print("[SUCCESS] DCT implementation PASSED")
        print("="*70)
        print("DCT/IDCT roundtrip reconstruction is accurate")
        print(f"Float32 error {error:.2e} < threshold 1e-5")
        print(f"BFloat16 error {error_bf16:.2e} < threshold 1e-2")
        return True
    else:
        print("\n" + "="*70)
        print("[FAILED] DCT implementation FAILED")
        print("="*70)
        print(f"Float32 reconstruction error: {error:.2e}")
        print(f"BFloat16 reconstruction error: {error_bf16:.2e}")
        return False


if __name__ == "__main__":
    # Run validation when script is executed directly
    success = validate_dct_implementation()

    if success:
        print("\nTesting frequency mask generation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Test each band type
        for band in ['low', 'mid', 'high']:
            mask = create_frequency_mask(224, 224, band=band, ratio=0.25, device=device)
            print(f"[OK] {band.upper()} band mask: {mask.shape}, {mask.sum().item():.0f} active coefficients")

        print("\n" + "="*70)
        print("[SUCCESS] All DCT utilities validated successfully")
        print("="*70)
    else:
        import sys
        sys.exit(1)
