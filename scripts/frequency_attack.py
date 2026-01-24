#!/usr/bin/env python3
"""
FrequencyConstrainedCWAttack: Carlini-Wagner attack with frequency constraints
Implements the attack described in VLA_Frequency_Attack_CONTEXT.md
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from PIL import Image

# Critical constants from Phase 0
ACTION_TOKEN_OFFSET = 31808
VOCAB_SIZE = 32064


class FrequencyConstrainedCWAttack:
    """
    Carlini-Wagner attack with frequency-domain constraints for OpenVLA.

    The attack minimizes CW loss to flip action bins while constraining
    perturbations to a specific frequency band (low/mid/high).

    Args:
        model: OpenVLA model (AutoModelForVision2Seq)
        processor: OpenVLA processor (AutoProcessor)
        epsilon: L∞ perturbation bound (default: 16/255)
        alpha: Step size for PGD (default: 2/255)
        num_iterations: Number of attack iterations (default: 50)
        freq_ratio: Frequency mask ratio (default: 0.25 for low-freq)
        kappa: CW loss margin (default: 5.0)
        device: Device to run on (default: "cuda:0")
    """

    def __init__(
        self,
        model,
        processor,
        epsilon: float = 16/255,
        alpha: float = 2/255,
        num_iterations: int = 100,
        freq_ratio: float = 0.25,
        kappa: float = 10.0,
        device: str = "cuda:0"
    ):
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.freq_ratio = freq_ratio
        self.kappa = kappa
        self.device = device

        # Pre-compute frequency mask
        self.freq_mask = self._create_freq_mask(224, 224, freq_ratio).to(device)

    def _create_freq_mask(self, H: int, W: int, ratio: float) -> torch.Tensor:
        """
        Create circular frequency mask (centered after fftshift).

        Args:
            H: Height of image
            W: Width of image
            ratio: Frequency ratio
                   Positive (e.g., 0.25, 0.5): Low-pass (keep central frequencies)
                   Negative (e.g., -0.25): High-pass (keep outer frequencies)

        Returns:
            Binary mask (H, W) with 1 in target frequency region
        """
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((y - cy).float()**2 + (x - cx).float()**2)

        # Handle negative ratio for high-pass filter
        if ratio < 0:
            # High-pass: keep frequencies OUTSIDE central radius
            radius = abs(ratio) * min(H, W) / 2
            mask = (dist > radius).float()
        else:
            # Low-pass: keep frequencies INSIDE central radius
            radius = ratio * min(H, W) / 2
            mask = (dist <= radius).float()

        return mask

    def _project_to_freq_band(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Project perturbation to target frequency band.

        Args:
            delta: Perturbation (B, C, H, W)

        Returns:
            Filtered perturbation (B, C, H, W)
        """
        # FFT per channel
        delta_freq = torch.fft.fftshift(
            torch.fft.fft2(delta, dim=(-2, -1)),
            dim=(-2, -1)
        )

        # Apply mask (broadcast over batch and channels)
        mask = self.freq_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        delta_freq_filtered = delta_freq * mask

        # IFFT back to spatial domain
        delta_filtered = torch.fft.ifft2(
            torch.fft.ifftshift(delta_freq_filtered, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        return delta_filtered

    def _compute_purity(self, delta: torch.Tensor) -> float:
        """
        Compute frequency purity (energy in band / total energy).

        Args:
            delta: Perturbation (B, C, H, W)

        Returns:
            Purity ratio in [0, 1]
        """
        delta_freq = torch.fft.fftshift(
            torch.fft.fft2(delta, dim=(-2, -1)),
            dim=(-2, -1)
        )

        mask = self.freq_mask.unsqueeze(0).unsqueeze(0)
        energy_in_band = (mask * delta_freq.abs()**2).sum()
        total_energy = (delta_freq.abs()**2).sum()

        return (energy_in_band / (total_energy + 1e-10)).item()

    def _get_clean_bins(
        self,
        image: torch.Tensor,
        instruction: str
    ) -> torch.Tensor:
        """
        Get clean action bins from model generation.

        Args:
            image: Input image (B, C, H, W) in [0, 1]
            instruction: Text instruction

        Returns:
            Clean bins (7,) in [0, 255]
        """
        # Process inputs
        pil_img = Image.fromarray(
            (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        inputs = self.processor(instruction, pil_img, return_tensors="pt")

        # Move to device with correct dtypes: bfloat16 for images, long for tokens
        inputs = {
            k: v.to(self.device, dtype=torch.bfloat16) if 'pixel' in k
            else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate action tokens
        with torch.no_grad():
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        action_tokens = gen_output[0, -7:]  # Last 7 tokens
        clean_bins = action_tokens - ACTION_TOKEN_OFFSET  # Convert to [0, 255]

        # Clamp to valid range
        clean_bins = torch.clamp(clean_bins, 0, 255)

        return clean_bins.cpu()

    def _compute_cw_loss(
        self,
        action_logits: torch.Tensor,
        clean_bins: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Carlini-Wagner loss for action bins.

        Args:
            action_logits: Logits (1, 7, 256)
            clean_bins: Clean bins (7,) in [0, 255]

        Returns:
            CW loss (scalar)
        """
        cw_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        for i in range(7):
            z = action_logits[0, i, :].float()  # (256,)
            correct_bin = int(clean_bins[i].item())
            correct_bin = max(0, min(255, correct_bin))

            # Get correct bin logit
            z_correct = z[correct_bin]

            # Get max logit from other bins
            mask = torch.ones(256, dtype=torch.bool, device=self.device)
            mask[correct_bin] = False
            z_other_max = z[mask].max()

            # CW loss for this dimension
            loss_i = torch.clamp(z_correct - z_other_max + self.kappa, min=0)
            cw_loss = cw_loss + loss_i

        return cw_loss

    def attack(
        self,
        image: torch.Tensor,
        instruction: str,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Execute frequency-constrained CW attack on image.

        Args:
            image: Clean image (1, C, H, W) in [0, 1]
            instruction: Text instruction
            verbose: Print per-iteration stats

        Returns:
            - Adversarial image (1, C, H, W) in [0, 1]
            - Attack info dict with metrics
        """
        # Get clean bins
        clean_bins = self._get_clean_bins(image, instruction).to(self.device)

        if verbose:
            print(f"[Attack] Clean bins: {clean_bins.tolist()}")

        # Process inputs to get normalized pixel_values (1, 6, 224, 224)
        pil_img = Image.fromarray(
            (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        inputs = self.processor(instruction, pil_img, return_tensors="pt")
        text_tokens = inputs["input_ids"].to(self.device)
        clean_pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)

        # Initialize perturbation in normalized space (6 channels)
        delta = torch.zeros_like(clean_pixel_values, requires_grad=True, device=self.device)

        # Attack loop
        best_loss = float('inf')
        best_delta = delta.clone().detach()

        for iteration in range(self.num_iterations):
            # Zero gradients from previous iteration
            if delta.grad is not None:
                delta.grad.zero_()

            # Apply perturbation (no clamping - normalized space)
            pixel_values_adv = clean_pixel_values + delta

            # Convert to bfloat16 for model
            pixel_values_adv_bf16 = pixel_values_adv.to(torch.bfloat16)

            # Forward pass with teacher forcing
            # Input: text + first 6 action tokens (to predict all 7)
            action_tokens_full = (clean_bins + ACTION_TOKEN_OFFSET).long()
            input_ids = torch.cat([
                text_tokens,
                action_tokens_full[:-1].unsqueeze(0).to(self.device)
            ], dim=1).long()

            outputs = self.model(
                pixel_values=pixel_values_adv_bf16,
                input_ids=input_ids
            )

            # Extract action logits
            T = text_tokens.shape[1]
            action_logits = outputs.logits[:, T-1:T+6, -256:]  # (1, 7, 256)

            # Compute CW loss
            cw_loss = self._compute_cw_loss(action_logits, clean_bins)

            # Backward pass
            cw_loss.backward()

            # Get gradient
            grad = delta.grad

            if grad is None:
                if verbose:
                    print(f"[Iter {iteration}] WARNING: No gradient")
                break

            # Update perturbation (gradient DESCENT to minimize CW loss)
            with torch.no_grad():
                delta_new = delta.data - self.alpha * grad.sign()

                # Frequency projection (apply to each 3-channel half separately)
                delta_new_siglip = self._project_to_freq_band(delta_new[:, :3, :, :])
                delta_new_dino = self._project_to_freq_band(delta_new[:, 3:, :, :])
                delta_new = torch.cat([delta_new_siglip, delta_new_dino], dim=1)

                # L∞ projection (scale of perturbation in normalized space)
                # In normalized space, epsilon is different for each encoder
                # For simplicity, use same epsilon (can be tuned)
                delta_new = torch.clamp(delta_new, -self.epsilon, self.epsilon)

                # Update delta.data in-place
                delta.data = delta_new

            # Track best
            if cw_loss.item() < best_loss:
                best_loss = cw_loss.item()
                best_delta = delta.clone().detach()

            # Check bin flips during teacher forcing (for logging only)
            with torch.no_grad():
                predicted_bins = action_logits.argmax(dim=-1).squeeze(0)
                num_flipped_tf = (predicted_bins.cpu() != clean_bins.cpu()).sum().item()

            if verbose and iteration % 10 == 0:
                print(f"[Iter {iteration}] Loss: {cw_loss.item():.4f}, "
                      f"FlippedTF: {num_flipped_tf}/7")

            # Early stopping based on CW loss (not bin flips during teacher forcing)
            # CW loss near 0 means attack succeeded
            if cw_loss.item() < 0.5:
                if verbose:
                    print(f"[Iter {iteration}] Early stop: CW loss < 0.5")
                break

        # Create final adversarial pixel_values
        pixel_values_adv_final = clean_pixel_values + best_delta

        # Compute final metrics
        with torch.no_grad():
            # Get adversarial bins by running model.generate
            gen_output_adv = self.model.generate(
                pixel_values=pixel_values_adv_final.to(torch.bfloat16),
                input_ids=text_tokens,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

            adv_bins = (gen_output_adv[0, -7:] - ACTION_TOKEN_OFFSET).cpu()
            adv_bins = torch.clamp(adv_bins, 0, 255)

            num_flipped = (adv_bins != clean_bins.cpu()).sum().item()
            bin_flip_rate = num_flipped / 7.0

        # Compute purity (average across both encoder channels)
        purity_siglip = self._compute_purity(best_delta[:, :3, :, :])
        purity_dino = self._compute_purity(best_delta[:, 3:, :, :])
        purity = (purity_siglip + purity_dino) / 2.0

        # Compute perturbation stats
        delta_norm_l2 = best_delta.norm().item()
        delta_norm_linf = best_delta.abs().max().item()

        info = {
            'clean_bins': clean_bins.cpu().numpy(),
            'adv_bins': adv_bins.cpu().numpy(),
            'num_bins_flipped': num_flipped,
            'bin_flip_rate': bin_flip_rate,
            'final_cw_loss': best_loss,
            'frequency_purity': purity,
            'perturbation_l2': delta_norm_l2,
            'perturbation_linf': delta_norm_linf,
            'iterations': iteration + 1
        }

        if verbose:
            print(f"\n[Attack Complete]")
            print(f"  Clean bins: {clean_bins.tolist()}")
            print(f"  Adv bins:   {adv_bins.tolist()}")
            print(f"  Bin flip rate: {bin_flip_rate*100:.1f}% ({num_flipped}/7)")
            print(f"  Frequency purity: {purity*100:.1f}%")
            print(f"  Perturbation L∞: {delta_norm_linf:.6f} (ε={self.epsilon:.6f})")

        return pixel_values_adv_final, info


def create_attack(
    model,
    processor,
    freq_ratio: float = 0.25,
    epsilon: float = 16/255,
    device: str = "cuda:0"
) -> FrequencyConstrainedCWAttack:
    """
    Factory function to create attack with standard settings.

    Args:
        model: OpenVLA model
        processor: OpenVLA processor
        freq_ratio: Frequency ratio (0.25=low, 0.5=mid, 1.0=full)
        epsilon: L∞ bound
        device: Device

    Returns:
        Configured attack instance
    """
    return FrequencyConstrainedCWAttack(
        model=model,
        processor=processor,
        epsilon=epsilon,
        alpha=2/255,
        num_iterations=50,
        freq_ratio=freq_ratio,
        kappa=5.0,
        device=device
    )
