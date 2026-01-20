"""
SE(3) Distance Function
Geometric distance computation for 7D robot actions on SE(3) manifold.
"""

import numpy as np


def se3_distance(action1: np.ndarray, action2: np.ndarray) -> float:
    """
    Compute geometric distance between two 7D actions on SE(3) manifold.
    
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




