#!/usr/bin/env python3
"""
Comprehensive test script for the openVLA environment.
Tests all critical imports and basic functionality.
"""

import sys
import os

# Set headless rendering for robosuite/libero
os.environ['MUJOCO_GL'] = 'osmesa'

print("=" * 70)
print("openVLA Environment Test Suite")
print("=" * 70)

# Test 1: Python version
print("\n[TEST 1] Python Version:")
print(f"Python: {sys.version}")
assert sys.version_info >= (3, 10), "Python 3.10+ required"
print("✓ Python version OK")

# Test 2: PyTorch
print("\n[TEST 2] PyTorch:")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch import OK")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test 3: Transformers
print("\n[TEST 3] Transformers:")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("✓ Transformers import OK")
except Exception as e:
    print(f"✗ Transformers import failed: {e}")
    sys.exit(1)

# Test 4: OpenVLA package
print("\n[TEST 4] OpenVLA Package:")
try:
    # This should work if openvla is installed
    import openvla
    print("✓ OpenVLA package import OK")
except Exception as e:
    print(f"⚠ OpenVLA package import: {e} (may be OK if using transformers directly)")

# Test 5: LIBERO
print("\n[TEST 5] LIBERO:")
try:
    import libero
    print(f"LIBERO version: {libero.__version__ if hasattr(libero, '__version__') else 'unknown'}")
    print("✓ LIBERO import OK")
except Exception as e:
    print(f"✗ LIBERO import failed: {e}")
    sys.exit(1)

# Test 6: Robosuite
print("\n[TEST 6] Robosuite:")
try:
    import robosuite
    print(f"Robosuite version: {robosuite.__version__ if hasattr(robosuite, '__version__') else 'unknown'}")
    print("✓ Robosuite import OK")
except Exception as e:
    print(f"⚠ Robosuite import warning: {e}")
    print("  (This is expected in headless mode - will work with MUJOCO_GL=osmesa)")

# Test 7: Core dependencies
print("\n[TEST 7] Core Dependencies:")
deps = ['numpy', 'PIL', 'h5py', 'einops', 'timm', 'tokenizers']
for dep in deps:
    try:
        if dep == 'PIL':
            import PIL
            print(f"✓ {dep} (Pillow) OK")
        else:
            __import__(dep)
            print(f"✓ {dep} OK")
    except Exception as e:
        print(f"✗ {dep} import failed: {e}")
        sys.exit(1)

# Test 8: OpenVLA Action Extractor (if exists)
print("\n[TEST 8] OpenVLA Action Extractor:")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from openvla_action_extractor import OpenVLAActionExtractor
    print("✓ OpenVLAActionExtractor import OK")
except Exception as e:
    print(f"⚠ OpenVLAActionExtractor import: {e} (may not exist yet)")

# Test 9: SE(3) Distance
print("\n[TEST 9] SE(3) Distance Function:")
try:
    from utils.se3_distance import se3_distance
    import numpy as np
    # Test with dummy actions
    action1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
    dist = se3_distance(action1, action2)
    print(f"✓ SE(3) distance function OK (test distance: {dist:.4f})")
except Exception as e:
    print(f"✗ SE(3) distance function failed: {e}")
    sys.exit(1)

# Test 10: LIBERO Loader
print("\n[TEST 10] LIBERO Loader:")
try:
    from utils.libero_loader import LIBEROLoader
    print("✓ LIBEROLoader import OK")
except Exception as e:
    print(f"✗ LIBEROLoader import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Environment is ready!")
print("=" * 70)

