# File: /data1/ma1/Ishaq/ump-vla/code/utils/target_generator.py
"""
Adversarial Target Generation for VLA Models.

IMPORTANT: All functions must be DETERMINISTIC (no randomness).
This ensures stable optimization during training.
"""

import numpy as np


def generate_generic_failure_target(clean_action: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Create adversarial target that causes generic task failure.
    Strategy: Invert gripper state and add DETERMINISTIC perturbation.
    
    Args:
        clean_action: np.ndarray (7,) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
        seed: Optional seed for reproducibility (if None, uses deterministic perturbation)
        
    Returns:
        np.ndarray (7,) - Target adversarial action
    """
    target = clean_action.copy()
    
    # Invert gripper state (most critical for task failure)
    # Gripper: -1 = closed, +1 = open
    if clean_action[6] > 0:
        target[6] = -1.0  # Was open, now closed
    else:
        target[6] = 1.0   # Was closed, now open
    
    # DETERMINISTIC perturbation based on clean_action values
    # This ensures same clean_action always produces same target
    # Use clean_action itself to generate consistent perturbation
    
    # Position perturbation: offset by 5cm in direction opposite to current motion
    # If robot is moving right (+x), target moves left (-x)
    target[0] = clean_action[0] - 0.05 * np.sign(clean_action[0] + 1e-8)
    target[1] = clean_action[1] - 0.05 * np.sign(clean_action[1] + 1e-8)
    target[2] = clean_action[2] - 0.05 * np.sign(clean_action[2] + 1e-8)
    
    # Rotation perturbation: small fixed offset
    target[3] = clean_action[3] + 0.05
    target[4] = clean_action[4] - 0.05
    target[5] = clean_action[5] + 0.05
    
    return target


def generate_drop_object_target(clean_action: np.ndarray) -> np.ndarray:
    """
    Create adversarial target that causes premature object dropping.
    Strategy: Set gripper to open, move downward.
    
    This is DETERMINISTIC - no random elements.
    
    Args:
        clean_action: np.ndarray (7,)
        
    Returns:
        np.ndarray (7,)
    """
    target = clean_action.copy()
    
    # Force gripper open (drops object)
    target[6] = 1.0
    
    # Move downward (negative Z)
    target[2] = -0.05  # Fixed downward motion
    
    # Reduce XY movement (stay in place)
    target[0] = 0.0
    target[1] = 0.0
    
    # Zero rotation
    target[3:6] = 0.0
    
    return target


def generate_opposite_direction_target(clean_action: np.ndarray) -> np.ndarray:
    """
    Create target that is exactly opposite to clean action.
    Simple and deterministic.
    
    Args:
        clean_action: np.ndarray (7,)
        
    Returns:
        np.ndarray (7,) - Negated action with inverted gripper
    """
    target = -clean_action.copy()
    
    # For gripper, invert the sign (since we already negated)
    # Actually just ensure it's flipped from original
    if clean_action[6] > 0:
        target[6] = -1.0
    else:
        target[6] = 1.0
    
    return target


def generate_zero_action_target(clean_action: np.ndarray) -> np.ndarray:
    """
    Create target of zero motion (robot freezes).
    Simplest deterministic target.
    
    Args:
        clean_action: np.ndarray (7,)
        
    Returns:
        np.ndarray (7,) - All zeros except gripper (open)
    """
    target = np.zeros(7)
    target[6] = 1.0  # Gripper open (likely causes drop)
    return target
