#!/bin/bash
# Shell script to setup headless rendering environment variables
# Source this file before running Python scripts: source setup_headless_rendering.sh

# Method 1: OSMesa (software rendering - recommended for headless systems)
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM

# Method 2: EGL (GPU rendering - uncomment if OSMesa doesn't work)
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# export EGL_PLATFORM=device
# if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
#     export MUJOCO_EGL_DEVICE_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
# fi

echo "Headless rendering configured: MUJOCO_GL=$MUJOCO_GL"


