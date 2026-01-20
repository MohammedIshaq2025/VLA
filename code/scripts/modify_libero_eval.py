#!/usr/bin/env python3
"""
Modify run_libero_eval.py to add headless rendering setup.

This script patches the LIBERO evaluation script to include
headless rendering configuration at the start.
"""

import os
import sys

eval_script_path = "code/openvla/experiments/robot/libero/run_libero_eval.py"

# Read the original file
with open(eval_script_path, 'r') as f:
    content = f.read()

# Check if already modified
if "setup_headless_rendering" in content:
    print(f"✓ {eval_script_path} already has headless rendering setup")
    sys.exit(0)

# Find the insertion point (after imports, before main code)
# Look for the line with "from libero.libero import benchmark"
insertion_marker = "from libero.libero import benchmark"
setup_code = """
# === HEADLESS RENDERING SETUP (for SLURM/remote supercomputers) ===
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))
from setup_headless_rendering import setup_headless_rendering
setup_headless_rendering('osmesa')  # Use OSMesa for headless rendering
# === END HEADLESS RENDERING SETUP ===

"""

# Insert the setup code after the marker
if insertion_marker in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if insertion_marker in line:
            # Insert after this line
            lines.insert(i + 1, setup_code)
            break
    
    # Write modified content
    modified_content = '\n'.join(lines)
    with open(eval_script_path, 'w') as f:
        f.write(modified_content)
    
    print(f"✓ Successfully modified {eval_script_path}")
    print("  Added headless rendering setup before LIBERO imports")
else:
    print(f"✗ Could not find insertion point in {eval_script_path}")
    sys.exit(1)




