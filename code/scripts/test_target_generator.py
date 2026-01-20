#!/usr/bin/env python3
"""
Test script for Adversarial Target Generation (Phase 3)
Verifies that target functions create meaningful adversarial targets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.target_generator import generate_generic_failure_target, generate_drop_object_target

print("=" * 70)
print("Adversarial Target Generator Test")
print("=" * 70)

# Test samples - typical action vectors from LIBERO
test_actions = [
    # Case 1: Gripper closed (negative), moving forward/up
    np.array([0.05, 0.02, 0.03, 0.01, -0.01, 0.02, -1.0]),
    # Case 2: Gripper open (positive), moving backward/down
    np.array([-0.03, -0.01, -0.02, 0.0, 0.05, -0.03, 1.0]),
    # Case 3: Near-zero action with closed gripper
    np.array([0.001, 0.002, -0.001, 0.0, 0.0, 0.0, -0.8]),
    # Case 4: Large movement with open gripper
    np.array([0.1, 0.08, 0.05, 0.1, -0.1, 0.15, 0.9]),
    # Case 5: Zero position, only gripper
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
]

all_tests_passed = True

print("\n" + "=" * 70)
print("TEST 1: Generic Failure Target")
print("=" * 70)

for i, clean_action in enumerate(test_actions):
    print(f"\n[Test Case {i+1}]")
    print(f"  Clean Action:  {np.array2string(clean_action, precision=4, suppress_small=True)}")
    
    target = generate_generic_failure_target(clean_action)
    print(f"  Target Action: {np.array2string(target, precision=4, suppress_small=True)}")
    
    # Verify gripper inversion
    gripper_inverted = (clean_action[6] > 0 and target[6] == -1.0) or (clean_action[6] <= 0 and target[6] == 1.0)
    print(f"  Gripper: {clean_action[6]:.2f} → {target[6]:.2f} | Inverted: {'✓' if gripper_inverted else '✗'}")
    
    # Verify position perturbation
    pos_diff = np.linalg.norm(target[:3] - clean_action[:3])
    pos_perturbed = pos_diff > 0
    print(f"  Position diff: {pos_diff:.4f} | Perturbed: {'✓' if pos_perturbed else '✗'}")
    
    # Verify rotation perturbation
    rot_diff = np.linalg.norm(target[3:6] - clean_action[3:6])
    rot_perturbed = rot_diff > 0
    print(f"  Rotation diff: {rot_diff:.4f} | Perturbed: {'✓' if rot_perturbed else '✗'}")
    
    # Check overall
    if not gripper_inverted:
        print(f"  ❌ FAILED: Gripper not inverted!")
        all_tests_passed = False
    elif not pos_perturbed:
        print(f"  ❌ FAILED: Position not perturbed!")
        all_tests_passed = False
    elif not rot_perturbed:
        print(f"  ❌ FAILED: Rotation not perturbed!")
        all_tests_passed = False
    else:
        print(f"  ✓ PASSED")

print("\n" + "=" * 70)
print("TEST 2: Drop Object Target")
print("=" * 70)

for i, clean_action in enumerate(test_actions):
    print(f"\n[Test Case {i+1}]")
    print(f"  Clean Action:  {np.array2string(clean_action, precision=4, suppress_small=True)}")
    
    target = generate_drop_object_target(clean_action)
    print(f"  Target Action: {np.array2string(target, precision=4, suppress_small=True)}")
    
    # Verify gripper is open
    gripper_open = target[6] == 1.0
    print(f"  Gripper: {clean_action[6]:.2f} → {target[6]:.2f} | Open: {'✓' if gripper_open else '✗'}")
    
    # Verify Z is negative (downward)
    z_downward = target[2] <= 0
    print(f"  Z movement: {clean_action[2]:.4f} → {target[2]:.4f} | Downward: {'✓' if z_downward else '✗'}")
    
    # Verify XY reduced
    xy_reduced = abs(target[0]) <= abs(clean_action[0]) and abs(target[1]) <= abs(clean_action[1])
    print(f"  XY: [{clean_action[0]:.4f}, {clean_action[1]:.4f}] → [{target[0]:.4f}, {target[1]:.4f}] | Reduced: {'✓' if xy_reduced else '✗'}")
    
    # Check overall
    if not gripper_open:
        print(f"  ❌ FAILED: Gripper not open!")
        all_tests_passed = False
    elif not z_downward:
        print(f"  ❌ FAILED: Z not downward!")
        all_tests_passed = False
    else:
        print(f"  ✓ PASSED")

print("\n" + "=" * 70)
print("TEST 3: Verify Targets Differ from Clean Actions")
print("=" * 70)

print("\nChecking that targets are meaningfully different from clean actions...")
np.random.seed(42)  # For reproducibility

differences = []
for i, clean_action in enumerate(test_actions):
    generic_target = generate_generic_failure_target(clean_action)
    drop_target = generate_drop_object_target(clean_action)
    
    generic_diff = np.linalg.norm(generic_target - clean_action)
    drop_diff = np.linalg.norm(drop_target - clean_action)
    
    differences.append((generic_diff, drop_diff))
    print(f"  Case {i+1}: Generic diff = {generic_diff:.4f}, Drop diff = {drop_diff:.4f}")

avg_generic = np.mean([d[0] for d in differences])
avg_drop = np.mean([d[1] for d in differences])
print(f"\n  Average Generic diff: {avg_generic:.4f}")
print(f"  Average Drop diff: {avg_drop:.4f}")

if avg_generic > 0.1 and avg_drop > 0.1:
    print("  ✓ Targets differ meaningfully from clean actions")
else:
    print("  ⚠️ Targets may be too similar to clean actions")

print("\n" + "=" * 70)
if all_tests_passed:
    print("✅ ALL TESTS PASSED")
else:
    print("❌ SOME TESTS FAILED")
    sys.exit(1)
print("=" * 70)



