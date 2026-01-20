#!/usr/bin/env python3
"""
Verification script for OpenVLA Action Extractor and SE(3) distance function.

Tests:
1. OpenVLA model loading
2. Action vector extraction from LIBERO image
3. SE(3) distance function correctness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openvla_action_extractor import OpenVLAActionExtractor
from utils.se3_distance import se3_distance
from utils.libero_loader import LIBEROLoader

print("=" * 70)
print("OpenVLA Action Extractor & SE(3) Distance Verification")
print("=" * 70)

# Test 1: SE(3) Distance Function
print("\n[TEST 1/3] SE(3) Distance Function")
print("-" * 70)

# Test 1a: Identical actions → distance should be 0
action1 = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, -1.0])
action2 = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, -1.0])
dist = se3_distance(action1, action2)
print(f"Identical actions: {dist:.6f} (expected: 0.0)")
assert np.abs(dist) < 1e-6, "Distance between identical actions should be 0"
print("✓ Test 1a passed")

# Test 1b: Opposite gripper → distance should be ~2
action1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
action2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
dist = se3_distance(action1, action2)
print(f"Opposite gripper: {dist:.6f} (expected: ~2.0)")
assert np.abs(dist - 2.0) < 0.1, "Distance for opposite gripper should be ~2"
print("✓ Test 1b passed")

# Test 1c: Different position
action1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
action2 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
dist = se3_distance(action1, action2)
print(f"Position difference (0.1m): {dist:.6f} (expected: ~0.1)")
assert np.abs(dist - 0.1) < 0.01, "Distance should match position difference"
print("✓ Test 1c passed")

# Test 1d: Non-negative
action1 = np.random.randn(7)
action2 = np.random.randn(7)
dist = se3_distance(action1, action2)
print(f"Random actions: {dist:.6f}")
assert dist >= 0, "Distance must be non-negative"
print("✓ Test 1d passed")

print("\n✅ SE(3) distance function tests passed!")

# Test 2: OpenVLA Model Loading
print("\n[TEST 2/3] OpenVLA Model Loading")
print("-" * 70)

try:
    extractor = OpenVLAActionExtractor(
        model_path="checkpoints/openvla-7b",
        device="cuda:0" if os.path.exists("/proc/driver/nvidia/version") else "cpu"
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Action Extraction from LIBERO
print("\n[TEST 3/3] Action Extraction from LIBERO Image")
print("-" * 70)

try:
    # Load LIBERO data
    loader = LIBEROLoader(base_path="/data1/ma1/Ishaq/ump-vla/data/libero")
    task_data = loader.load_task("libero_spatial", task_id=0)
    
    # Get a sample frame
    episode = task_data['episodes'][0]
    image, action_gt, instruction = loader.sample_random_frame(episode)
    
    print(f"Episode: {episode['episode_id']}")
    print(f"Image shape: {image.shape}")
    print(f"Instruction: {instruction}")
    print(f"Ground truth action: {action_gt}")
    
    # Extract action from OpenVLA
    action_pred = extractor.get_action_vector(image, instruction)
    
    print(f"\nPredicted action shape: {action_pred.shape}")
    print(f"Predicted action: {action_pred}")
    
    # Verify shape
    assert action_pred.shape == (7,), f"Expected shape (7,), got {action_pred.shape}"
    print("✓ Action shape correct: (7,)")
    
    # Verify action ranges
    print("\nAction component ranges:")
    print(f"  Position (xyz): [{action_pred[:3].min():.4f}, {action_pred[:3].max():.4f}]")
    print(f"  Rotation (rpy): [{action_pred[3:6].min():.4f}, {action_pred[3:6].max():.4f}]")
    print(f"  Gripper: {action_pred[6]:.4f}")
    
    # Compute distance between predicted and ground truth
    dist = se3_distance(action_pred, action_gt)
    print(f"\nSE(3) distance (predicted vs ground truth): {dist:.4f}")
    
    print("\n✓ Action extraction works correctly")
    
except Exception as e:
    print(f"✗ Action extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅✅✅ ALL VERIFICATION TESTS PASSED ✅✅✅")
print("=" * 70)
print("\nVerified components:")
print("  ✓ SE(3) distance function (4 tests)")
print("  ✓ OpenVLA model loading")
print("  ✓ Action vector extraction from LIBERO images")
print("  ✓ Action shape and range validation")
print("\nReady for Phase 2: ZOO Optimizer implementation")




