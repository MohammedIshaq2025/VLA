"""
SE(3) Distance Function
Geometric distance computation for 7D robot actions on SE(3) manifold.

V2 Update: Added normalized_se3_distance() for scale-invariant comparison.
"""

import numpy as np

# ============================================================================
# Default scale factors based on typical OpenVLA action ranges
# These normalize each component to roughly [0, 1] range
# ============================================================================
DEFAULT_POS_SCALE = 0.1    # Position actions typically in [-0.1, 0.1] meters
DEFAULT_ROT_SCALE = 0.3    # Rotation actions typically in [-0.3, 0.3] radians
DEFAULT_GRIP_SCALE = 2.0   # Gripper in [-1, 1], max diff = 2


def se3_distance(action1: np.ndarray, action2: np.ndarray) -> float:
    """
    Compute geometric distance between two 7D actions on SE(3) manifold.

    WARNING: This function has a scale imbalance problem!
    Gripper (range 0-2) dominates position (range 0-0.1).
    Use normalized_se3_distance() for balanced comparison.

    SE(3) = R^3 × SO(3) (translation × rotation)
    We use weighted L2 for translation and angular distance for rotation.

    Args:
        action1: np.ndarray (7,) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action2: np.ndarray (7,) - same format

    Returns:
        float: Distance in action space (non-negative)
               Units: meters for position, radians for rotation, normalized for gripper
    """
    # Position component (3D Euclidean distance)
    pos1, pos2 = action1[:3], action2[:3]
    pos_dist = np.linalg.norm(pos1 - pos2)

    # Rotation component (SO(3) geodesic distance approximation)
    # For small rotations (Euler angles), Euclidean norm approximates geodesic distance
    rot1, rot2 = action1[3:6], action2[3:6]
    rot_dist = np.linalg.norm(rot1 - rot2)

    # Gripper component (binary/continuous)
    grip1, grip2 = action1[6], action2[6]
    grip_dist = np.abs(grip1 - grip2)

    # Combined metric with equal weighting
    # Typical ranges:
    #   - Position: 0-0.1 meters (end-effector delta)
    #   - Rotation: 0-0.2 radians (orientation delta)
    #   - Gripper: 0-2 (open=-1, close=+1, so max diff=2)
    total_dist = pos_dist + rot_dist + grip_dist

    return total_dist


def normalized_se3_distance(action1: np.ndarray, action2: np.ndarray,
                            pos_scale: float = DEFAULT_POS_SCALE,
                            rot_scale: float = DEFAULT_ROT_SCALE,
                            grip_scale: float = DEFAULT_GRIP_SCALE,
                            w_pos: float = 1.0,
                            w_rot: float = 0.5,
                            w_grip: float = 0.1) -> dict:
    """
    Compute NORMALIZED geometric distance between two 7D actions.

    This function solves the scale imbalance problem in se3_distance() by:
    1. Normalizing each component to [0, 1] range using typical action scales
    2. Applying configurable weights to each component

    Mathematical basis:
        pos_norm = ||pos1 - pos2|| / pos_scale  → roughly [0, 1]
        rot_norm = ||rot1 - rot2|| / rot_scale  → roughly [0, 1]
        grip_norm = |grip1 - grip2| / grip_scale → [0, 1]

        total = w_pos * pos_norm + w_rot * rot_norm + w_grip * grip_norm

    Default weights (w_pos=1.0, w_rot=0.5, w_grip=0.1) prioritize position
    for trajectory drift attacks while de-emphasizing gripper flips.

    Args:
        action1: np.ndarray (7,) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action2: np.ndarray (7,) - same format
        pos_scale: Normalization scale for position (default: 0.1m)
        rot_scale: Normalization scale for rotation (default: 0.3 rad)
        grip_scale: Normalization scale for gripper (default: 2.0)
        w_pos: Weight for position component (default: 1.0)
        w_rot: Weight for rotation component (default: 0.5)
        w_grip: Weight for gripper component (default: 0.1)

    Returns:
        dict: Contains:
            - 'total': Weighted normalized total distance
            - 'pos_dist': Raw position distance (meters)
            - 'rot_dist': Raw rotation distance (radians)
            - 'grip_dist': Raw gripper distance (0-2)
            - 'pos_norm': Normalized position distance
            - 'rot_norm': Normalized rotation distance
            - 'grip_norm': Normalized gripper distance
    """
    # Raw distances
    pos_dist = np.linalg.norm(action1[:3] - action2[:3])
    rot_dist = np.linalg.norm(action1[3:6] - action2[3:6])
    grip_dist = np.abs(action1[6] - action2[6])

    # Normalized distances (clipped to [0, 1] for safety)
    pos_norm = min(pos_dist / pos_scale, 1.0) if pos_scale > 0 else 0.0
    rot_norm = min(rot_dist / rot_scale, 1.0) if rot_scale > 0 else 0.0
    grip_norm = min(grip_dist / grip_scale, 1.0) if grip_scale > 0 else 0.0

    # Weighted total
    total = w_pos * pos_norm + w_rot * rot_norm + w_grip * grip_norm

    return {
        'total': float(total),
        'pos_dist': float(pos_dist),
        'rot_dist': float(rot_dist),
        'grip_dist': float(grip_dist),
        'pos_norm': float(pos_norm),
        'rot_norm': float(rot_norm),
        'grip_norm': float(grip_norm)
    }


def position_only_distance(action1: np.ndarray, action2: np.ndarray) -> float:
    """
    Compute position-only distance between two actions.

    For trajectory drift attacks, position is the primary component that
    causes task failure through cumulative drift.

    Args:
        action1: np.ndarray (7,) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action2: np.ndarray (7,) - same format

    Returns:
        float: Position distance in meters
    """
    return float(np.linalg.norm(action1[:3] - action2[:3]))




