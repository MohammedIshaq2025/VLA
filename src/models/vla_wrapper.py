"""
OpenVLA model wrapper for adversarial attacks.

This module provides a clean interface to the OpenVLA model with support for
gradient-based attacks following the literature (Kim et al. WACV 2024, SSA ECCV 2022).

Key Features:
- Differentiable forward pass for gradient computation
- Support for action token logits (not just generated actions)
- Visual feature extraction for analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, Tuple, Optional, Union, List
from PIL import Image
import numpy as np


class OpenVLAWrapper(nn.Module):
    """
    Wrapper for OpenVLA model that exposes vision encoder and action prediction.

    OpenVLA Architecture:
    - Vision Encoder: SigLIP (ViT-SO400M) + DINOv2 (ViT-L/14) fused
    - Projector: 2-layer MLP
    - LLM: Llama 2 7B
    - Action Head: 256 discrete bins per DoF, 7 DoF total

    This wrapper provides BOTH:
    1. predict_action() - Non-differentiable action inference (uses generate())
    2. forward_for_attack() - Differentiable forward pass for gradient-based attacks
    """

    def __init__(
        self,
        model_path: str = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize OpenVLA wrapper.

        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Target device ('cuda' or 'cpu')
            torch_dtype: Model precision (bfloat16 recommended for H200)
        """
        super().__init__()

        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading OpenVLA from: {model_path}")
        print(f"Device: {device}, Dtype: {torch_dtype}")

        # Load processor (handles image preprocessing and tokenization)
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("[OK] Processor loaded")

        # Load model using AutoModelForVision2Seq (verified API)
        print("Loading model (this may take 30-60 seconds)...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            print("Moving model to GPU...")
            self.model = self.model.to(device)
        elif device == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, using CPU")
            self.device = 'cpu'

        # Set to eval mode
        self.model.eval()

        # Get image size from processor
        self.image_size = self._get_image_size()

        # Get vocab size and action dimension for loss computation
        self.vocab_size = self.model.config.text_config.vocab_size
        self.action_dim = 7  # 7 DoF for robot actions

        # ImageNet normalization stats (used by processor internally)
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        print(f"[OK] Model loaded successfully")
        print(f"Image size: {self.image_size}")
        print(f"Vocab size: {self.vocab_size}")

        # Report memory usage if on GPU
        if device == 'cuda' and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

    def _get_image_size(self) -> int:
        """Extract expected image size from processor."""
        try:
            return self.processor.image_processor.size['height']
        except:
            return 224

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to image tensor.

        Args:
            image: Tensor of shape (B, C, H, W) in [0, 1]

        Returns:
            Normalized tensor
        """
        mean = self.image_mean.to(image.device, dtype=image.dtype)
        std = self.image_std.to(image.device, dtype=image.dtype)
        return (image - mean) / std

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Remove ImageNet normalization from image tensor.

        Args:
            image: Normalized tensor of shape (B, C, H, W)

        Returns:
            Tensor in [0, 1] range
        """
        mean = self.image_mean.to(image.device, dtype=image.dtype)
        std = self.image_std.to(image.device, dtype=image.dtype)
        return image * std + mean

    def preprocess_image(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert image to PIL format for processor.

        Args:
            image: Image in various formats

        Returns:
            PIL Image
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image_np = image.permute(1, 2, 0).detach().cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            image = Image.fromarray(image_np)
        elif isinstance(image, np.ndarray):
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        return image

    def get_pixel_values(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Get normalized pixel values ready for model forward pass.

        This uses the processor to ensure correct preprocessing.

        Args:
            image: Input image

        Returns:
            Pixel values tensor of shape (1, 6, H, W) for fused backbone
        """
        image_pil = self.preprocess_image(image)

        # Use processor's image_processor directly
        image_inputs = self.processor.image_processor(image_pil, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(self.device, dtype=self.torch_dtype)

        return pixel_values

    def predict_action(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        instruction: str,
        unnorm_key: str = "bridge_orig"
    ) -> np.ndarray:
        """
        Predict robot action from image and instruction (non-differentiable).

        Args:
            image: Input image
            instruction: Natural language instruction
            unnorm_key: Key for action unnormalization

        Returns:
            Predicted action array (7,)
        """
        image_pil = self.preprocess_image(image)
        inputs = self.processor(instruction, image_pil)

        if self.device == 'cuda':
            inputs = inputs.to(self.device, dtype=self.torch_dtype)
        else:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False
            )

        if torch.is_tensor(action):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze()

        return action

    def forward_for_attack(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable forward pass for gradient-based attacks.

        This is the key method for frequency-band sensitivity testing following
        Kim et al. (WACV 2024) methodology.

        Args:
            pixel_values: Image tensor (B, C, H, W) - should have requires_grad=True
            input_ids: Tokenized instruction
            attention_mask: Attention mask for instruction
            labels: Optional labels for loss computation

        Returns:
            Tuple of (logits, loss) where:
            - logits: Language model logits (B, seq_len, vocab_size)
            - loss: Cross-entropy loss if labels provided, else None
        """
        # Ensure pixel_values has gradients
        if not pixel_values.requires_grad:
            pixel_values.requires_grad_(True)

        # Forward through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True
        )

        return outputs.logits, outputs.loss

    def get_text_inputs(self, instruction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokenized instruction for forward pass.

        Args:
            instruction: Natural language instruction

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Tokenize instruction
        text_inputs = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        return input_ids, attention_mask

    def compute_feature_loss(
        self,
        pixel_values_clean: torch.Tensor,
        pixel_values_perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute visual feature divergence loss.

        This is an alternative loss function for gradient computation when
        action logits are not directly accessible.

        Args:
            pixel_values_clean: Clean image pixel values
            pixel_values_perturbed: Perturbed image pixel values

        Returns:
            Negative cosine similarity (to maximize divergence)
        """
        # Get visual features from the vision backbone
        with torch.no_grad():
            features_clean = self.model.vision_backbone(pixel_values_clean)

        features_perturbed = self.model.vision_backbone(pixel_values_perturbed)

        # Flatten features
        feat_clean = features_clean.view(features_clean.size(0), -1)
        feat_perturbed = features_perturbed.view(features_perturbed.size(0), -1)

        # Negative cosine similarity (minimize to maximize divergence)
        cos_sim = F.cosine_similarity(feat_clean, feat_perturbed, dim=1)

        return -cos_sim.mean()  # Negative because we want to maximize divergence

    def get_vision_features(
        self,
        image: Union[torch.Tensor, Image.Image]
    ) -> torch.Tensor:
        """
        Extract vision encoder features (for analysis).

        Args:
            image: Input image

        Returns:
            Vision features tensor
        """
        pixel_values = self.get_pixel_values(image)

        with torch.no_grad():
            features = self.model.vision_backbone(pixel_values)

        return features

    def prepare_attack_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare all inputs needed for gradient-based attack.

        This is a convenience method that returns properly formatted inputs.

        Args:
            image: Input image
            instruction: Natural language instruction

        Returns:
            Dictionary containing pixel_values, input_ids, attention_mask
        """
        # Get pixel values
        pixel_values = self.get_pixel_values(image)

        # Get text inputs
        input_ids, attention_mask = self.get_text_inputs(instruction)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def test_wrapper():
    """Test the OpenVLA wrapper with gradient computation."""
    import sys

    model_path = "/data1/ma1/Ishaq/ump-vla/checkpoints/openvla-7b/"

    print("="*70)
    print("Testing OpenVLA Wrapper with Gradient Support")
    print("="*70)

    try:
        # Initialize wrapper
        wrapper = OpenVLAWrapper(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        print("\n[OK] Model loaded successfully")

        # Test with dummy input
        print("\nTesting action prediction (non-differentiable)...")
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        instruction = "pick up the red block"

        action = wrapper.predict_action(dummy_image, instruction)
        print(f"[OK] Predicted action shape: {action.shape}")
        print(f"[OK] Predicted action: {action}")

        # Test gradient computation
        print("\nTesting gradient computation (differentiable)...")
        inputs = wrapper.prepare_attack_inputs(dummy_image, instruction)

        # Enable gradients on pixel values
        pixel_values = inputs['pixel_values'].clone().requires_grad_(True)

        # Forward pass
        logits, _ = wrapper.forward_for_attack(
            pixel_values=pixel_values,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        print(f"[OK] Logits shape: {logits.shape}")

        # Compute loss and gradient
        # Use mean of logits as simple loss for testing
        loss = logits.mean()
        loss.backward()

        print(f"[OK] Loss value: {loss.item():.4f}")
        print(f"[OK] Gradient shape: {pixel_values.grad.shape}")
        print(f"[OK] Gradient max: {pixel_values.grad.abs().max().item():.6f}")
        print(f"[OK] Gradient mean: {pixel_values.grad.abs().mean().item():.6f}")

        # Verify gradient is non-zero
        assert pixel_values.grad is not None, "Gradient is None"
        assert pixel_values.grad.abs().max() > 0, "Gradient is all zeros"

        print("\n" + "="*70)
        print("[SUCCESS] VLA wrapper test with gradient support PASSED")
        print("="*70)
        print("\nWrapper capabilities verified:")
        print("  - Model loads correctly on GPU")
        print("  - Non-differentiable action prediction works")
        print("  - Differentiable forward pass works")
        print("  - Gradient computation verified")
        print("  - Gradient is non-zero (model is sensitive to input)")

        return True

    except Exception as e:
        print(f"\n[FAILED] VLA wrapper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_wrapper()
    import sys
    sys.exit(0 if success else 1)
