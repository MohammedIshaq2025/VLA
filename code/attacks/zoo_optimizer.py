# File: /data1/ma1/Ishaq/ump-vla/code/attacks/zoo_optimizer.py
"""
Zero-Order Optimization on SE(3) manifold for adversarial patches.

Attack Goal: Make the model output an adversarial action (close to target) 
that differs significantly from the clean/correct action.

Loss Function: Minimize distance(prediction, adversarial_target)
ASR Metric: Prediction deviates significantly from clean prediction

FIXED BUGS:
- Uses consistent random state via rng parameter
- Target is computed once per sample (deterministic target generator)
"""

import numpy as np
import torch
from typing import Callable, Dict, List, Tuple, Any
import time
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openvla_action_extractor import OpenVLAActionExtractor
from utils.se3_distance import se3_distance


class ZOOSOptimizer:
    """
    Zero-Order Optimization on SE(3) manifold for adversarial patches.
    Uses symmetric sampling (antithetic) to reduce variance.
    
    Attack Strategy:
    - Apply a learned patch to images
    - Optimize patch to make model predictions close to adversarial target
    - Adversarial target is designed to cause task failure (inverted gripper, wrong position)
    """
    
    def __init__(self, 
                 model: OpenVLAActionExtractor,
                 patch_size: int = 32,
                 learning_rate: float = 0.01,
                 perturbation_scale: float = 0.1,
                 query_budget: int = 200,
                 early_stop_threshold: float = 85.0,
                 early_stop_patience: int = 10,
                 asr_threshold: float = 0.5,
                 seed: int = 42):
        """
        Args:
            model: OpenVLA action extractor
            patch_size: Size of adversarial patch (default: 32x32)
            learning_rate: Step size for patch updates
            perturbation_scale: Scale of random perturbations for gradient estimation
            query_budget: Maximum number of optimization steps
            early_stop_threshold: ASR threshold for early stopping (%)
            early_stop_patience: Steps above threshold before stopping
            asr_threshold: SE(3) distance threshold for counting as attack success
            seed: Random seed for reproducibility
        """
        self.model = model
        self.patch_size = patch_size
        self.lr = learning_rate
        self.sigma = perturbation_scale  # Perturbation scale for ZOO
        self.query_budget = query_budget
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.asr_threshold = asr_threshold
        self.patience_counter = 0
        
        # Use seeded random state for reproducibility
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        
        # Initialize patch with small random values (seeded)
        self.patch = self.rng.uniform(0.3, 0.7, (patch_size, patch_size, 3))
        
        # History tracking
        self.query_history = []
        
    def apply_patch(self, image: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Apply patch to image at specified position.
        
        Args:
            image: (H, W, 3) RGB image uint8
            position: (x, y) top-left corner of patch
            
        Returns:
            Patched image (uint8)
        """
        x, y = position
        patched = image.copy()
        
        # Ensure patch fits within image bounds
        h, w = self.patch_size, self.patch_size
        max_x = image.shape[1] - w
        max_y = image.shape[0] - h
        x = np.clip(x, 0, max_x)
        y = np.clip(y, 0, max_y)
        
        # Apply patch (convert from [0,1] to [0,255])
        patched[y:y+h, x:x+w] = (self.patch * 255).astype(np.uint8)
        return patched
    
    def compute_loss(self, action_pred: np.ndarray, action_target: np.ndarray) -> float:
        """
        Compute loss for adversarial attack.
        
        We want the model to output the adversarial target, so:
        Loss = distance(prediction, target)
        Minimize loss → prediction gets closer to adversarial target
        
        Args:
            action_pred: Predicted action from patched image
            action_target: Adversarial target action (designed to cause failure)
            
        Returns:
            Loss value (lower = prediction closer to adversarial target = attack working)
        """
        return se3_distance(action_pred, action_target)
    
    def compute_asr(self, action_pred: np.ndarray, action_clean: np.ndarray) -> float:
        """
        Compute Attack Success Rate.
        
        ASR = 1 if the prediction deviates significantly from clean prediction.
        This measures whether the attack successfully changed model behavior.
        
        Args:
            action_pred: Predicted action from patched image
            action_clean: Clean prediction (what model outputs WITHOUT patch)
            
        Returns:
            1.0 if attack successful, 0.0 otherwise
        """
        # Check gripper flip (most important for task failure)
        gripper_change = np.abs(action_pred[6] - action_clean[6])
        gripper_flipped = gripper_change > 1.0  # Requires significant change
        
        # Check overall SE(3) deviation
        distance_from_clean = se3_distance(action_pred, action_clean)
        significant_deviation = distance_from_clean > self.asr_threshold
        
        # Attack is successful if either condition is met
        return 1.0 if (gripper_flipped or significant_deviation) else 0.0
    
    def query_with_patch(self, image: np.ndarray, instruction: str, 
                         action_target: np.ndarray, clean_pred: np.ndarray,
                         patch_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Single query: apply patch, get prediction, compute loss and ASR.
        
        Args:
            image: Clean image (no patch)
            instruction: Task instruction
            action_target: Adversarial target action
            clean_pred: Model's prediction WITHOUT patch (baseline)
            patch_position: Where to place patch
            
        Returns:
            Dict with loss, asr, action_pred, metrics
        """
        start = time.time()
        
        # Apply patch to image
        patched_image = self.apply_patch(image, patch_position)
        
        # Get model prediction on patched image
        action_pred = self.model.get_action_vector(patched_image, instruction)
        
        # Ensure numpy array with shape (7,)
        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.cpu().numpy()
        action_pred = np.array(action_pred).flatten()
        
        if len(action_pred) != 7:
            raise ValueError(f"Expected action shape (7,), got {action_pred.shape}")
        
        # Compute loss (distance to adversarial target - lower is better)
        loss = self.compute_loss(action_pred, action_target)
        
        # Compute ASR (deviation from clean prediction)
        asr = self.compute_asr(action_pred, clean_pred)
        
        # Distance metrics for logging
        dist_to_target = se3_distance(action_pred, action_target)
        dist_to_clean = se3_distance(action_pred, clean_pred)
        
        # Component-wise changes
        pos_change = np.linalg.norm(action_pred[:3] - clean_pred[:3])
        rot_change = np.linalg.norm(action_pred[3:6] - clean_pred[3:6])
        gripper_change = np.abs(action_pred[6] - clean_pred[6])
        
        # GPU memory
        gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        return {
            "loss": loss,
            "asr": asr,
            "action_pred": action_pred,
            "dist_to_target": dist_to_target,
            "dist_to_clean": dist_to_clean,
            "pos_change": pos_change,
            "rot_change": rot_change,
            "gripper_change": gripper_change,
            "query_time": time.time() - start,
            "gpu_memory": gpu_memory
        }
    
    def optimize_step(self, image: np.ndarray, instruction: str, 
                      action_target: np.ndarray, clean_pred: np.ndarray,
                      patch_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        One optimization step using antithetic sampling (ZOO).
        
        ZOO Gradient Estimation:
        - Sample random direction δ
        - Query f(x + σδ) and f(x - σδ)  
        - Estimate gradient: ∇f ≈ (f(x+σδ) - f(x-σδ)) / (2σ) * δ
        - Update: x ← x - lr * ∇f (gradient descent to minimize loss)
        """
        # Save original patch
        original_patch = self.patch.copy()
        
        # Sample random perturbation direction (normalized) using seeded RNG
        delta = self.rng.randn(*self.patch.shape)
        delta = delta / (np.linalg.norm(delta) + 1e-8)  # Normalize
        
        # Query with positive perturbation: x + σδ
        self.patch = np.clip(original_patch + self.sigma * delta, 0, 1)
        result_pos = self.query_with_patch(image, instruction, action_target, clean_pred, patch_position)
        
        # Query with negative perturbation: x - σδ
        self.patch = np.clip(original_patch - self.sigma * delta, 0, 1)
        result_neg = self.query_with_patch(image, instruction, action_target, clean_pred, patch_position)
        
        # Estimate directional derivative: (f(x+σδ) - f(x-σδ)) / (2σ)
        grad_estimate = (result_pos["loss"] - result_neg["loss"]) / (2 * self.sigma)
        
        # Gradient descent update: x ← x - lr * grad * δ
        # We minimize loss, so we subtract the gradient
        self.patch = np.clip(original_patch - self.lr * grad_estimate * delta, 0, 1)
        
        # Compute metrics (average of both queries)
        avg_loss = (result_pos["loss"] + result_neg["loss"]) / 2
        avg_asr = (result_pos["asr"] + result_neg["asr"]) / 2
        avg_dist_to_target = (result_pos["dist_to_target"] + result_neg["dist_to_target"]) / 2
        avg_dist_to_clean = (result_pos["dist_to_clean"] + result_neg["dist_to_clean"]) / 2
        avg_pos_change = (result_pos["pos_change"] + result_neg["pos_change"]) / 2
        avg_rot_change = (result_pos["rot_change"] + result_neg["rot_change"]) / 2
        avg_gripper_change = (result_pos["gripper_change"] + result_neg["gripper_change"]) / 2
        
        # Record history
        self.query_history.append({
            "query_id": len(self.query_history),
            "loss": avg_loss,
            "asr": avg_asr,
            "dist_to_target": avg_dist_to_target,
            "dist_to_clean": avg_dist_to_clean,
            "pos_change": avg_pos_change,
            "rot_change": avg_rot_change,
            "gripper_change": avg_gripper_change,
            "grad_magnitude": abs(grad_estimate),
            "gpu_memory": max(result_pos["gpu_memory"], result_neg["gpu_memory"]),
            "query_time": result_pos["query_time"] + result_neg["query_time"]
        })
        
        return {
            "loss": avg_loss,
            "asr": avg_asr,
            "dist_to_target": avg_dist_to_target,
            "dist_to_clean": avg_dist_to_clean,
            "pos_change": avg_pos_change,
            "rot_change": avg_rot_change,
            "gripper_change": avg_gripper_change,
            "grad_magnitude": abs(grad_estimate),
            "gpu_memory": max(result_pos["gpu_memory"], result_neg["gpu_memory"])
        }
    
    def train(self, episodes: List[Dict], action_target_fn: Callable) -> Dict[str, Any]:
        """
        Train adversarial patch using ZOO.
        
        Args:
            episodes: List of training episodes from LIBERO
            action_target_fn: Function(clean_action) -> adversarial_target_action
            
        Returns:
            Dict with final patch, training history, metrics
        """
        from utils.libero_loader import LIBEROLoader
        loader = LIBEROLoader()
        
        print(f"\n[ZOO] Starting SE(3) Manifold Attack")
        print(f"[ZOO] Query budget: {self.query_budget}")
        print(f"[ZOO] Learning rate: {self.lr}, Perturbation scale: {self.sigma}")
        print(f"[ZOO] Early stop: {self.early_stop_threshold}% ASR for {self.early_stop_patience} steps")
        print(f"[ZOO] ASR threshold (SE3 distance): {self.asr_threshold}")
        print(f"[ZOO] Training on {len(episodes)} episodes\n")
        
        best_asr = 0.0
        best_patch = self.patch.copy()
        best_loss = float('inf')
        
        # Running averages for smoother logging
        recent_losses = []
        recent_asrs = []
        recent_gripper_changes = []
        
        for query in range(self.query_budget):
            # Sample random episode and frame using seeded RNG
            episode = self.py_rng.choice(episodes)
            image, clean_action, instruction = loader.sample_random_frame(episode, self.py_rng)
            
            # Generate adversarial target from clean action (DETERMINISTIC)
            target_action = action_target_fn(clean_action)
            
            # Get clean prediction (model output WITHOUT patch) - this is our baseline
            clean_pred = self.model.get_action_vector(image, instruction)
            if isinstance(clean_pred, torch.Tensor):
                clean_pred = clean_pred.cpu().numpy()
            clean_pred = np.array(clean_pred).flatten()
            
            # Patch position (center for 128x128 images)
            position = (48, 48)
            
            # Optimization step
            result = self.optimize_step(image, instruction, target_action, clean_pred, position)
            
            # Update running averages
            recent_losses.append(result['loss'])
            recent_asrs.append(result['asr'])
            recent_gripper_changes.append(result['gripper_change'])
            if len(recent_losses) > 20:
                recent_losses.pop(0)
                recent_asrs.pop(0)
                recent_gripper_changes.pop(0)
            
            # Log progress every 10 queries
            if query % 10 == 0:
                avg_loss = np.mean(recent_losses)
                avg_asr = np.mean(recent_asrs)
                avg_grip = np.mean(recent_gripper_changes)
                print(f"[ZOO] Q {query:4d}/{self.query_budget} | "
                      f"Loss: {result['loss']:.3f} (avg:{avg_loss:.3f}) | "
                      f"ASR: {result['asr']*100:.0f}% (avg:{avg_asr*100:.0f}%) | "
                      f"Δgrip: {result['gripper_change']:.3f} (avg:{avg_grip:.3f}) | "
                      f"GPU: {result['gpu_memory']:.1f}GB")
            
            # Track best patch (by loss - lower is better)
            if result["loss"] < best_loss:
                best_loss = result["loss"]
                best_patch = self.patch.copy()
            
            # Track best ASR
            if result["asr"] > best_asr:
                best_asr = result["asr"]
            
            # Early stopping check
            if np.mean(recent_asrs) >= self.early_stop_threshold / 100.0:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n[ZOO] Early stopping: Avg ASR >= {self.early_stop_threshold}% "
                          f"for {self.early_stop_patience} consecutive steps")
                    break
            else:
                self.patience_counter = 0
            
            # Clear GPU cache periodically
            if query % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final summary
        final_losses = [h["loss"] for h in self.query_history[-20:]] if len(self.query_history) >= 20 else [h["loss"] for h in self.query_history]
        final_asrs = [h["asr"] for h in self.query_history[-20:]] if len(self.query_history) >= 20 else [h["asr"] for h in self.query_history]
        final_gripper_changes = [h["gripper_change"] for h in self.query_history[-20:]] if len(self.query_history) >= 20 else [h["gripper_change"] for h in self.query_history]
        
        print(f"\n[ZOO] Training Complete!")
        print(f"[ZOO] Total queries: {len(self.query_history)}")
        print(f"[ZOO] Best loss: {best_loss:.4f}")
        print(f"[ZOO] Best ASR: {best_asr*100:.1f}%")
        print(f"[ZOO] Final avg loss (last 20): {np.mean(final_losses):.4f}")
        print(f"[ZOO] Final avg ASR (last 20): {np.mean(final_asrs)*100:.1f}%")
        print(f"[ZOO] Final avg gripper change: {np.mean(final_gripper_changes):.4f}")
        
        return {
            "best_patch": best_patch,
            "final_patch": self.patch.copy(),
            "training_history": self.query_history,
            "best_loss": best_loss,
            "best_asr": best_asr,
            "total_queries": len(self.query_history)
        }
