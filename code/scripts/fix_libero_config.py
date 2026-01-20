#!/usr/bin/env python3
"""
Fix LIBERO configuration file to point to correct installation paths.

This script updates ~/.libero/config.yaml to use the correct paths
from the installed LIBERO package, fixing the BDDL file path issue.
"""

import os
import yaml

def fix_libero_config():
    """Update LIBERO config file with correct paths."""
    from libero.libero import get_default_path_dict
    
    # Get default paths (these point to the installed LIBERO package)
    default_paths = get_default_path_dict()
    
    # Update config file
    config_path = os.path.expanduser("~/.libero")
    config_file = os.path.join(config_path, "config.yaml")
    
    # Ensure config directory exists
    os.makedirs(config_path, exist_ok=True)
    
    print("=" * 70)
    print("Fixing LIBERO Configuration")
    print("=" * 70)
    print(f"\nConfig file: {config_file}")
    
    # Write the correct paths
    with open(config_file, 'w') as f:
        yaml.dump(default_paths, f)
    
    print("\nUpdated config paths:")
    for key, path in default_paths.items():
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {key}: {abs_path}")
    
    # Verify it works
    print("\nVerifying get_libero_path works...")
    from libero.libero import get_libero_path
    try:
        bddl_path = get_libero_path("bddl_files")
        print(f"✓ get_libero_path('bddl_files') = {bddl_path}")
        print(f"✓ Path exists: {os.path.exists(bddl_path)}")
        print("\n✅ LIBERO configuration fixed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_libero_config()
    exit(0 if success else 1)




