#!/usr/bin/env python3
"""
Test script to verify LIBERO works in headless mode.

This script tests if LIBERO environments can be created and run
without a display, which is required for SLURM jobs.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup headless rendering BEFORE importing robosuite/libero
from setup_headless_rendering import setup_headless_rendering

print("=" * 60)
print("LIBERO Headless Rendering Test")
print("=" * 60)

# Configure rendering
print("\n[1/4] Configuring headless rendering...")
if not setup_headless_rendering('auto'):
    print("✗ Failed to configure rendering")
    sys.exit(1)

# Now safe to import
print("\n[2/4] Importing LIBERO and robosuite...")
try:
    from libero.libero import benchmark
    import robosuite
    print("✓ Successfully imported libero and robosuite")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test creating a LIBERO environment
print("\n[3/4] Creating LIBERO environment...")
try:
    # Get a task from LIBERO-Spatial suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    
    # Create environment (this uses OffScreenRenderEnv)
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), 
        task.problem_folder, 
        task.bddl_file
    )
    
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256
    )
    print(f"✓ Successfully created LIBERO environment: {task.language}")
    
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test running a few steps
print("\n[4/4] Testing environment step()...")
try:
    obs = env.reset()
    print(f"✓ Reset successful. Observation keys: {list(obs.keys())}")
    
    # Check if we have image observations
    if "agentview_image" in obs:
        print(f"✓ Image observation shape: {obs['agentview_image'].shape}")
    
    # Take a few dummy steps
    for i in range(3):
        action = [0, 0, 0, 0, 0, 0, -1]  # Dummy action
        obs, reward, done, info = env.step(action)
        if done:
            print(f"✓ Episode ended. Success: {info.get('success', 'N/A')}")
            break
    
    print("✓ Environment step() works correctly")
    
except Exception as e:
    print(f"✗ Failed to run environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓✓✓ ALL TESTS PASSED - LIBERO works in headless mode! ✓✓✓")
print("=" * 60)
print("\nYou can now run LIBERO evaluations on SLURM clusters.")
print("Make sure to call setup_headless_rendering() before importing libero/robosuite.")




