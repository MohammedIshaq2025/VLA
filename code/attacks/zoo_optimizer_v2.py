# File: /data1/ma1/Ishaq/ump-vla/code/attacks/zoo_optimizer_v2.py
"""
Zero-Order Optimization V2: MAXIMIZE DEVIATION FROM CLEAN PREDICTION

This is the PRIMARY optimizer for Direction 2 (Closed-Loop Trajectory Attack).

Goal: Create a patch that causes MAXIMUM CONSISTENT deviation from the model's
clean predictions. In closed-loop execution, these deviations accumulate and
cause task failure.

Key Design Decisions:
1. NO adversarial target - we maximize deviation from clean, period
2. Mini-batch gradient estimation - reduces variance, improves consistency
3. Best patch tracked by DEVIATION metric, not loss
4. Weighted components: position (for trajectory drift) + gripper (for task failure)

Loss Function:
    L = -w_pos * ||pos_patched - pos_clean||
        - w_rot * ||rot_patched - rot_clean||
        - w_grip * |grip_patched - grip_clean|

    Minimizing L maximizes deviation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import time
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openvla_action_extractor import OpenVLAActionExtractor
from utils.se3_distance import se3_distance


class ZOOSOptimizerV2:
    """
    Zero-Order Optimization V2: Maximize deviation from clean prediction.

    Designed for Direction 2: Closed-Loop Trajectory Attack

    Key Features:
    - Mini-batch gradient estimation (reduces variance)
    - Best patch tracked by average deviation (not loss)
    - Configurable component weights (position vs rotation vs gripper)
    - Gradient retry on zero (don't waste queries)
    """

    def __init__(self,
                 model: OpenVLAActionExtractor,
                 patch_size: int = 32,
                 learning_rate: float = 0.01,
                 perturbation_scale: float = 0.1,
                 query_budget: int = 200,
                 mini_batch_size: int = 3,
                 early_stop_threshold: float = 85.0,
                 early_stop_patience: int = 20,
                 deviation_threshold: float = 0.3,
                 position_weight: float = 1.0,
                 rotation_weight: float = 1.0,
                 gripper_weight: float = 5.0,
                 seed: int = 42):
        """
        Args:
            model: OpenVLA action extractor
            patch_size: Size of adversarial patch (default: 32x32)
            learning_rate: Step size for patch updates
            perturbation_scale: Scale of random perturbations for gradient estimation
            query_budget: Maximum number of optimization steps
            mini_batch_size: Number of frames to average for gradient estimation
            early_stop_threshold: Deviation rate threshold for early stopping (%)
            early_stop_patience: Steps above threshold before stopping
            deviation_threshold: SE(3) distance threshold for "significant" deviation
            position_weight: Weight on position deviation in loss
            rotation_weight: Weight on rotation deviation in loss
            gripper_weight: Weight on gripper change in loss
            seed: Random seed for reproducibility
        """
        self.model = model
        self.patch_size = patch_size
        self.lr = learning_rate
        self.sigma = perturbation_scale
        self.query_budget = query_budget
        self.mini_batch_size = mini_batch_size
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.deviation_threshold = deviation_threshold
        self.patience_counter = 0

        # Component weights for loss function
        self.w_pos = position_weight
        self.w_rot = rotation_weight
        self.w_grip = gripper_weight

        # Use seeded random state for reproducibility
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)

        # Initialize patch with small random values (seeded)
        self.patch = self.rng.uniform(0.3, 0.7, (patch_size, patch_size, 3))

        # History tracking
        self.query_history = []

        # Best patch tracking - BY DEVIATION, not loss
        self.best_patch = self.patch.copy()
        self.best_avg_deviation = 0.0
        self.validation_window = []  # Rolling window of deviations
        self.validation_window_size = 20

    def apply_patch(self, image: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Apply patch to image at specified position."""
        x, y = position
        patched = image.copy()

        h, w = self.patch_size, self.patch_size
        img_h, img_w = image.shape[:2]

        # Ensure patch fits within image bounds
        max_x = img_w - w
        max_y = img_h - h
        x = np.clip(x, 0, max(0, max_x))
        y = np.clip(y, 0, max(0, max_y))

        # Handle edge case where patch is larger than image
        if h > img_h or w > img_w:
            # Resize patch to fit
            scale = min(img_h / h, img_w / w) * 0.9
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple resize by slicing
            patch_resized = self.patch[:new_h, :new_w]
            patched[y:y+new_h, x:x+new_w] = (patch_resized * 255).astype(np.uint8)
        else:
            patched[y:y+h, x:x+w] = (self.patch * 255).astype(np.uint8)

        return patched

    def compute_deviation(self, patched_pred: np.ndarray, clean_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute component-wise deviation between patched and clean predictions.

        Returns:
            Dict with pos_dev, rot_dev, grip_dev, total_dev
        """
        pos_dev = np.linalg.norm(patched_pred[:3] - clean_pred[:3])
        rot_dev = np.linalg.norm(patched_pred[3:6] - clean_pred[3:6])
        grip_dev = np.abs(patched_pred[6] - clean_pred[6])

        # Weighted total (same as SE3 distance but with configurable weights)
        total_dev = self.w_pos * pos_dev + self.w_rot * rot_dev + self.w_grip * grip_dev

        return {
            "pos_dev": pos_dev,
            "rot_dev": rot_dev,
            "grip_dev": grip_dev,
            "total_dev": total_dev,
            "se3_dev": pos_dev + rot_dev + grip_dev  # Standard SE3 for comparison
        }

    def compute_loss(self, patched_pred: np.ndarray, clean_pred: np.ndarray) -> float:
        """
        Compute loss for MAXIMIZE DEVIATION objective.

        Loss = -weighted_deviation

        Minimizing this loss maximizes deviation.
        """
        dev = self.compute_deviation(patched_pred, clean_pred)
        # Negative because we minimize loss but want to maximize deviation
        return -dev["total_dev"]

    def is_significant_deviation(self, patched_pred: np.ndarray, clean_pred: np.ndarray) -> bool:
        """Check if deviation exceeds threshold (for success rate calculation)."""
        se3_dev = se3_distance(patched_pred, clean_pred)
        return se3_dev > self.deviation_threshold

    def query_single_frame(self, image: np.ndarray, instruction: str,
                           patch_values: np.ndarray,
                           patch_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Query model on a single frame with given patch values.

        Args:
            image: Clean image (no patch)
            instruction: Task instruction
            patch_values: Patch to apply (may be perturbed)
            patch_position: Where to place patch

        Returns:
            Dict with patched_pred, clean_pred, deviation metrics
        """
        # Store current patch, apply test patch
        original_patch = self.patch
        self.patch = patch_values

        # Get clean prediction (no patch)
        clean_pred = self.model.get_action_vector(image, instruction)
        if isinstance(clean_pred, torch.Tensor):
            clean_pred = clean_pred.cpu().numpy()
        clean_pred = np.array(clean_pred).flatten()

        # Get patched prediction
        patched_image = self.apply_patch(image, patch_position)
        patched_pred = self.model.get_action_vector(patched_image, instruction)
        if isinstance(patched_pred, torch.Tensor):
            patched_pred = patched_pred.cpu().numpy()
        patched_pred = np.array(patched_pred).flatten()

        # Restore original patch
        self.patch = original_patch

        # Compute deviation
        dev = self.compute_deviation(patched_pred, clean_pred)
        loss = -dev["total_dev"]
        is_significant = self.is_significant_deviation(patched_pred, clean_pred)

        return {
            "clean_pred": clean_pred,
            "patched_pred": patched_pred,
            "loss": loss,
            "deviation": dev,
            "is_significant": is_significant
        }

    def optimize_step(self, episodes: List[Dict], loader,
                      patch_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        One optimization step using mini-batch gradient estimation.

        Key improvement: Average gradients across multiple frames to reduce variance.
        """
        original_patch = self.patch.copy()

        # Sample random perturbation direction (normalized)
        delta = self.rng.randn(*self.patch.shape)
        delta = delta / (np.linalg.norm(delta) + 1e-8)

        # Prepare perturbed patches
        patch_pos = np.clip(original_patch + self.sigma * delta, 0, 1)
        patch_neg = np.clip(original_patch - self.sigma * delta, 0, 1)

        # Mini-batch: sample multiple frames and average
        losses_pos = []
        losses_neg = []
        deviations = []
        significant_count = 0

        for _ in range(self.mini_batch_size):
            # Sample random frame
            episode = self.py_rng.choice(episodes)
            image, _, instruction = loader.sample_random_frame(episode, self.py_rng)

            # Query with positive perturbation
            result_pos = self.query_single_frame(image, instruction, patch_pos, patch_position)
            losses_pos.append(result_pos["loss"])

            # Query with negative perturbation
            result_neg = self.query_single_frame(image, instruction, patch_neg, patch_position)
            losses_neg.append(result_neg["loss"])

            # Track deviation (average of pos and neg for this frame)
            avg_dev = (result_pos["deviation"]["se3_dev"] + result_neg["deviation"]["se3_dev"]) / 2
            deviations.append(avg_dev)

            if result_pos["is_significant"] or result_neg["is_significant"]:
                significant_count += 1

        # Average losses across mini-batch
        avg_loss_pos = np.mean(losses_pos)
        avg_loss_neg = np.mean(losses_neg)

        # Gradient estimate
        grad_estimate = (avg_loss_pos - avg_loss_neg) / (2 * self.sigma)

        # Check for zero gradient (both perturbations gave same result)
        if abs(grad_estimate) < 1e-8:
            # Retry with different direction (don't waste this query)
            delta = self.rng.randn(*self.patch.shape)
            delta = delta / (np.linalg.norm(delta) + 1e-8)
            patch_pos = np.clip(original_patch + self.sigma * delta, 0, 1)
            patch_neg = np.clip(original_patch - self.sigma * delta, 0, 1)

            # Re-query on first frame only (efficiency)
            episode = self.py_rng.choice(episodes)
            image, _, instruction = loader.sample_random_frame(episode, self.py_rng)
            result_pos = self.query_single_frame(image, instruction, patch_pos, patch_position)
            result_neg = self.query_single_frame(image, instruction, patch_neg, patch_position)

            grad_estimate = (result_pos["loss"] - result_neg["loss"]) / (2 * self.sigma)

        # Gradient descent update (minimize loss = maximize deviation)
        self.patch = np.clip(original_patch - self.lr * grad_estimate * delta, 0, 1)

        # Compute metrics for this step
        avg_deviation = np.mean(deviations)
        deviation_rate = significant_count / self.mini_batch_size
        avg_loss = (avg_loss_pos + avg_loss_neg) / 2

        # Update validation window (for best patch tracking)
        self.validation_window.append(avg_deviation)
        if len(self.validation_window) > self.validation_window_size:
            self.validation_window.pop(0)

        # Update best patch based on ROLLING AVERAGE DEVIATION (not single-step loss)
        current_avg_deviation = np.mean(self.validation_window)
        if current_avg_deviation > self.best_avg_deviation:
            self.best_avg_deviation = current_avg_deviation
            self.best_patch = self.patch.copy()

        # Record history (convert numpy types to Python types for JSON serialization)
        self.query_history.append({
            "query_id": len(self.query_history),
            "loss": float(avg_loss),
            "avg_deviation": float(avg_deviation),
            "deviation_rate": float(deviation_rate),
            "grad_magnitude": float(abs(grad_estimate)),
            "rolling_avg_deviation": float(current_avg_deviation),
            "is_best": bool(current_avg_deviation >= self.best_avg_deviation)
        })

        return {
            "loss": avg_loss,
            "avg_deviation": avg_deviation,
            "deviation_rate": deviation_rate,
            "grad_magnitude": abs(grad_estimate),
            "rolling_avg_deviation": current_avg_deviation
        }

    def train(self, episodes: List[Dict], patch_position: Tuple[int, int] = (48, 48)) -> Dict[str, Any]:
        """
        Train adversarial patch using ZOO V2 (maximize deviation).

        Args:
            episodes: List of training episodes from LIBERO
            patch_position: Where to place patch on image (default: center for 128x128)

        Returns:
            Dict with best_patch, training history, metrics
        """
        from utils.libero_loader import LIBEROLoader
        loader = LIBEROLoader()

        print(f"\n{'='*70}")
        print("ZOO V2: MAXIMIZE DEVIATION ATTACK (Direction 2 Aligned)")
        print(f"{'='*70}")
        print(f"Goal: Maximize deviation from clean predictions")
        print(f"Metric: Average SE(3) deviation across frames")
        print(f"{'='*70}")
        print(f"Query budget:     {self.query_budget}")
        print(f"Mini-batch size:  {self.mini_batch_size}")
        print(f"Learning rate:    {self.lr}")
        print(f"Perturbation σ:   {self.sigma}")
        print(f"Patch size:       {self.patch_size}x{self.patch_size}")
        print(f"Patch position:   {patch_position}")
        print(f"Weights:          pos={self.w_pos}, rot={self.w_rot}, grip={self.w_grip}")
        print(f"Deviation thresh: {self.deviation_threshold}")
        print(f"Training episodes:{len(episodes)}")
        print(f"Seed:             {self.seed}")
        print(f"{'='*70}\n")

        start_time = time.time()

        # Running averages for logging
        recent_deviations = []
        recent_deviation_rates = []

        for query in range(self.query_budget):
            # Optimization step
            result = self.optimize_step(episodes, loader, patch_position)

            # Update running averages
            recent_deviations.append(result['avg_deviation'])
            recent_deviation_rates.append(result['deviation_rate'])
            if len(recent_deviations) > 20:
                recent_deviations.pop(0)
                recent_deviation_rates.pop(0)

            # Log progress every 10 queries
            if query % 10 == 0 or query == self.query_budget - 1:
                avg_dev = np.mean(recent_deviations)
                avg_rate = np.mean(recent_deviation_rates)
                print(f"[Q {query:4d}/{self.query_budget}] "
                      f"Dev: {result['avg_deviation']:.4f} (avg: {avg_dev:.4f}) | "
                      f"Rate: {result['deviation_rate']*100:5.1f}% (avg: {avg_rate*100:.1f}%) | "
                      f"Best: {self.best_avg_deviation:.4f} | "
                      f"Grad: {result['grad_magnitude']:.4f}")

            # Early stopping check (based on deviation rate)
            if np.mean(recent_deviation_rates) >= self.early_stop_threshold / 100.0:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n[EARLY STOP] Deviation rate >= {self.early_stop_threshold}% "
                          f"for {self.early_stop_patience} steps")
                    break
            else:
                self.patience_counter = 0

            # Clear GPU cache periodically
            if query % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        training_time = time.time() - start_time

        # Final statistics
        final_deviations = [h["avg_deviation"] for h in self.query_history[-20:]]
        final_deviation_rates = [h["deviation_rate"] for h in self.query_history[-20:]]

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total queries:        {len(self.query_history)}")
        print(f"Training time:        {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"Best avg deviation:   {self.best_avg_deviation:.4f}")
        print(f"Final avg deviation:  {np.mean(final_deviations):.4f}")
        print(f"Final deviation rate: {np.mean(final_deviation_rates)*100:.1f}%")
        print(f"{'='*70}\n")

        return {
            "best_patch": self.best_patch,
            "final_patch": self.patch.copy(),
            "training_history": self.query_history,
            "best_avg_deviation": self.best_avg_deviation,
            "total_queries": len(self.query_history),
            "training_time": training_time
        }
