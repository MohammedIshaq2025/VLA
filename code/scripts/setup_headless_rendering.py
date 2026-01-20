#!/usr/bin/env python3
"""
Setup script for headless rendering on supercomputers/SLURM clusters.

This script configures environment variables for robosuite/LIBERO to work
in headless environments without a display.

Usage:
    # Before importing robosuite or libero, run:
    python setup_headless_rendering.py
    
    # Or source the environment variables:
    source setup_headless_rendering.sh
"""

import os
import sys


def setup_headless_rendering(method='auto'):
    """
    Setup environment variables for headless rendering.
    
    Args:
        method: 'osmesa' (software rendering), 'egl' (GPU rendering), or 'auto' (try both)
    """
    if method == 'auto':
        # Try OSMesa first (more compatible, no GPU driver needed)
        os.environ['MUJOCO_GL'] = 'osmesa'
        if 'PYOPENGL_PLATFORM' in os.environ:
            del os.environ['PYOPENGL_PLATFORM']
        if 'EGL_PLATFORM' in os.environ:
            del os.environ['EGL_PLATFORM']
        print("[INFO] Set MUJOCO_GL=osmesa (software rendering)")
        
        # Try importing to see if it works
        try:
            import robosuite
            print("[SUCCESS] OSMesa rendering configured successfully")
            return True
        except Exception as e:
            print(f"[WARNING] OSMesa failed: {e}")
            print("[INFO] Trying EGL (GPU rendering)...")
            
            # Fallback to EGL
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            os.environ['EGL_PLATFORM'] = 'device'
            
            # Set GPU device if available
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpu_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
                os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_id)
                print(f"[INFO] Set MUJOCO_EGL_DEVICE_ID={gpu_id}")
            
            try:
                import robosuite
                print("[SUCCESS] EGL rendering configured successfully")
                return True
            except Exception as e2:
                print(f"[ERROR] EGL also failed: {e2}")
                return False
    
    elif method == 'osmesa':
        os.environ['MUJOCO_GL'] = 'osmesa'
        if 'PYOPENGL_PLATFORM' in os.environ:
            del os.environ['PYOPENGL_PLATFORM']
        print("[INFO] Set MUJOCO_GL=osmesa")
        return True
    
    elif method == 'egl':
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['EGL_PLATFORM'] = 'device'
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
            os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_id)
        print("[INFO] Set MUJOCO_GL=egl")
        return True
    
    return False


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else 'auto'
    success = setup_headless_rendering(method)
    
    if success:
        print("\n✓ Headless rendering configured. You can now import robosuite/libero.")
        print("\nTo use in your scripts, add this at the top:")
        print("  import sys")
        print("  sys.path.insert(0, 'code/scripts')")
        print("  from setup_headless_rendering import setup_headless_rendering")
        print("  setup_headless_rendering()")
        print("  # Now safe to import robosuite/libero")
    else:
        print("\n✗ Failed to configure headless rendering.")
        print("\nPossible solutions:")
        print("1. Install OSMesa libraries:")
        print("   conda install -c conda-forge mesalib")
        print("   # OR")
        print("   sudo apt-get install libosmesa6-dev")
        print("\n2. For EGL (GPU rendering), ensure:")
        print("   - NVIDIA drivers are installed")
        print("   - libnvidia-egl libraries are available")
        print("   - GPU is accessible (check with nvidia-smi)")
        sys.exit(1)




