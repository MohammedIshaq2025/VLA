#!/usr/bin/env python3
"""
Test script for ZOO Optimizer (Fixed Version)

Tests the optimizer with a small query budget to verify:
1. Loss DECREASES (gets smaller, approaching 0 = prediction close to target)
2. ASR measures actual deviation from clean action
3. GPU memory usage is tracked
4. Logging works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless rendering
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import torch
from openvla_action_extractor import OpenVLAActionExtractor
from attacks.zoo_optimizer import ZOOSOptimizer
from utils.libero_loader import LIBEROLoader
from utils.target_generator import generate_generic_failure_target

print("=" * 70)
print("ZOO Optimizer Test (Fixed Version)")
print("=" * 70)

# Test parameters
QUERY_BUDGET = 50  # Small budget for testing
PATCH_SIZE = 32
LEARNING_RATE = 0.01
PERTURBATION_SCALE = 0.1
ASR_THRESHOLD = 0.3

print(f"\n[SETUP] Test Parameters:")
print(f"  Query budget: {QUERY_BUDGET}")
print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Perturbation scale: {PERTURBATION_SCALE}")
print(f"  ASR threshold: {ASR_THRESHOLD}")

print(f"\n[SETUP] Loading LIBERO data...")
loader = LIBEROLoader()
task_data = loader.load_task("libero_spatial", 0)
train_episodes, _ = loader.split_episodes(task_data["episodes"], train_ratio=0.3)

print(f"[SETUP] Loaded {len(train_episodes)} training episodes")
print(f"[SETUP] Task: {task_data['task_name']}")

print(f"\n[SETUP] Loading OpenVLA model...")
model = OpenVLAActionExtractor(
    model_path="checkpoints/openvla-7b",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
print(f"[SETUP] Model loaded on: {model.device}")

print(f"\n[SETUP] Initializing ZOO Optimizer...")
optimizer = ZOOSOptimizer(
    model=model,
    patch_size=PATCH_SIZE,
    learning_rate=LEARNING_RATE,
    perturbation_scale=PERTURBATION_SCALE,
    query_budget=QUERY_BUDGET,
    early_stop_threshold=85.0,
    early_stop_patience=10,
    asr_threshold=ASR_THRESHOLD
)

print(f"[SETUP] Starting training...")
print("=" * 70)

# Run training
results = optimizer.train(train_episodes, generate_generic_failure_target)

print("\n" + "=" * 70)
print("Training Analysis")
print("=" * 70)

# Analyze results
if len(optimizer.query_history) > 0:
    losses = [h["loss"] for h in optimizer.query_history]
    asrs = [h["asr"] for h in optimizer.query_history]
    dist_to_targets = [h["dist_to_target"] for h in optimizer.query_history]
    dist_to_cleans = [h["dist_to_clean"] for h in optimizer.query_history]
    
    print(f"\n[METRICS] Training Statistics:")
    print(f"  Total queries: {len(optimizer.query_history)}")
    
    print(f"\n  Loss (distance to target - lower is better):")
    print(f"    Initial: {losses[0]:.4f}")
    print(f"    Final:   {losses[-1]:.4f}")
    print(f"    Change:  {losses[-1] - losses[0]:.4f} {'✓ decreased' if losses[-1] < losses[0] else '✗ increased'}")
    print(f"    Best:    {min(losses):.4f}")
    
    print(f"\n  ASR (attack success rate):")
    print(f"    Initial: {asrs[0]*100:.1f}%")
    print(f"    Final:   {asrs[-1]*100:.1f}%")
    print(f"    Best:    {max(asrs)*100:.1f}%")
    print(f"    Average: {np.mean(asrs)*100:.1f}%")
    
    print(f"\n  Distance to Clean Action (deviation caused by patch):")
    print(f"    Initial: {dist_to_cleans[0]:.4f}")
    print(f"    Final:   {dist_to_cleans[-1]:.4f}")
    print(f"    Average: {np.mean(dist_to_cleans):.4f}")
    
    # Validation checks
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)
    
    # Check 1: Loss should decrease over time
    first_half_loss = np.mean(losses[:len(losses)//2])
    second_half_loss = np.mean(losses[len(losses)//2:])
    loss_decreasing = second_half_loss < first_half_loss
    print(f"\n[CHECK 1] Loss trend:")
    print(f"  First half avg:  {first_half_loss:.4f}")
    print(f"  Second half avg: {second_half_loss:.4f}")
    print(f"  Result: {'✓ PASS - Loss decreasing' if loss_decreasing else '⚠ WARN - Loss not decreasing (may need more queries)'}")
    
    # Check 2: ASR should be reasonable (not 100% immediately)
    immediate_100_asr = asrs[0] == 1.0 and all(a == 1.0 for a in asrs[:5])
    print(f"\n[CHECK 2] ASR sanity:")
    print(f"  Initial ASR: {asrs[0]*100:.1f}%")
    print(f"  Result: {'✗ FAIL - Suspiciously high initial ASR' if immediate_100_asr else '✓ PASS - ASR looks reasonable'}")
    
    # Check 3: Patch effect should be measurable
    avg_patch_effect = np.mean(dist_to_cleans)
    print(f"\n[CHECK 3] Patch effect:")
    print(f"  Average deviation from clean: {avg_patch_effect:.4f}")
    print(f"  Result: {'✓ PASS - Patch has effect' if avg_patch_effect > 0.1 else '⚠ WARN - Weak patch effect'}")
    
    # GPU memory stats
    if torch.cuda.is_available():
        gpu_memories = [h["gpu_memory"] for h in optimizer.query_history]
        print(f"\n[GPU] Memory Usage:")
        print(f"  Average: {np.mean(gpu_memories):.2f} GB")
        print(f"  Max: {np.max(gpu_memories):.2f} GB")
    
    print("\n" + "=" * 70)
    all_checks_pass = loss_decreasing and not immediate_100_asr
    if all_checks_pass:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("⚠️  SOME CHECKS NEED ATTENTION (see above)")
    print("=" * 70)
else:
    print("\n❌ ERROR: No queries were executed")
    sys.exit(1)
