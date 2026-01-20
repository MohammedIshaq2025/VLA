#!/usr/bin/env python3
"""
GPU Verification Script for LIBERO Loader
Tests LIBERO data loading on GPU node to ensure everything works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.libero_loader import LIBEROLoader

print("=" * 70)
print("LIBERO Loader GPU Verification")
print("=" * 70)

# Test parameters
SUITE = "libero_spatial"
TASK_ID = 0
NUM_RANDOM_FRAMES = 5

print(f"\n[TEST] Loading task: {SUITE}, task_{TASK_ID}")
print("-" * 70)

try:
    # Initialize loader
    loader = LIBEROLoader()
    
    # Load task
    task_data = loader.load_task(SUITE, TASK_ID)
    
    # Verify episode count
    num_episodes = task_data["num_episodes"]
    print(f"\n[VERIFY] Number of episodes: {num_episodes}")
    assert num_episodes == 50, f"Expected 50 episodes, got {num_episodes}"
    print("✓ Episode count correct (50 episodes)")
    
    # Print episode lengths
    episode_lengths = [ep["length"] for ep in task_data["episodes"]]
    print(f"\n[VERIFY] Episode lengths:")
    print(f"  Min: {min(episode_lengths)}")
    print(f"  Max: {max(episode_lengths)}")
    print(f"  Mean: {np.mean(episode_lengths):.1f}")
    print(f"  Median: {np.median(episode_lengths):.1f}")
    print(f"  Total frames: {sum(episode_lengths)}")
    
    # Sample random frames with non-zero actions
    print(f"\n[VERIFY] Sampling {NUM_RANDOM_FRAMES} random frames with non-zero actions...")
    sampled_frames = []
    episodes = task_data["episodes"]
    
    # Sample frames from random episodes
    import random
    for _ in range(NUM_RANDOM_FRAMES):
        # Pick a random episode
        episode = random.choice(episodes)
        # Sample a random frame from it
        image, action, instruction = loader.sample_random_frame(episode)
        
        # Find the frame index for display
        frame_idx = None
        for i, ep_action in enumerate(episode["actions"]):
            if np.allclose(ep_action, action):
                frame_idx = i
                break
        
        sampled_frames.append({
            "episode_id": episode["episode_id"],
            "frame_idx": frame_idx if frame_idx is not None else "unknown",
            "action": action,
            "image": image,
            "instruction": instruction
        })
    
    print(f"✓ Successfully sampled {len(sampled_frames)} frames")
    
    # Print action ranges for sampled frames
    print(f"\n[VERIFY] Action ranges for sampled frames:")
    all_actions = np.array([frame["action"] for frame in sampled_frames])
    
    print(f"  Shape: {all_actions.shape} (should be ({NUM_RANDOM_FRAMES}, 7))")
    assert all_actions.shape == (NUM_RANDOM_FRAMES, 7), f"Expected shape ({NUM_RANDOM_FRAMES}, 7), got {all_actions.shape}"
    
    print(f"\n  Per-dimension ranges:")
    dim_names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    for i, name in enumerate(dim_names):
        dim_actions = all_actions[:, i]
        print(f"    {name:8s}: min={dim_actions.min():8.4f}, max={dim_actions.max():8.4f}, mean={dim_actions.mean():8.4f}")
    
    # Verify non-zero actions
    print(f"\n[VERIFY] Checking for non-zero actions...")
    action_norms = np.linalg.norm(all_actions[:, :6], axis=1)  # Position + rotation (exclude gripper)
    non_zero_count = np.sum(action_norms > 1e-6)
    print(f"  Frames with non-zero position/rotation: {non_zero_count}/{NUM_RANDOM_FRAMES}")
    
    if non_zero_count < NUM_RANDOM_FRAMES:
        print(f"  ⚠ Warning: Some frames have zero actions (this may be OK if gripper-only actions)")
    
    # Verify gripper values
    gripper_values = all_actions[:, 6]
    print(f"\n  Gripper values: {gripper_values}")
    print(f"  Gripper range: [{gripper_values.min():.2f}, {gripper_values.max():.2f}]")
    
    # Print sample frame details
    print(f"\n[VERIFY] Sample frame details:")
    for i, frame in enumerate(sampled_frames[:3]):  # Show first 3
        print(f"  Frame {i+1}:")
        print(f"    Episode: {frame['episode_id']}, Frame: {frame['frame_idx']}")
        print(f"    Action: {frame['action']}")
        print(f"    Image shape: {frame['image'].shape}")
    
    print("\n" + "=" * 70)
    print("✅ ALL VERIFICATION TESTS PASSED")
    print("=" * 70)
    print(f"✓ Task loaded: {task_data['task_name']}")
    print(f"✓ Episodes: {num_episodes}")
    print(f"✓ Random frames sampled: {len(sampled_frames)}")
    print(f"✓ Action shapes correct")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

