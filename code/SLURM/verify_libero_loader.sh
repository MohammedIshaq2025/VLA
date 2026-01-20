#!/bin/bash
#SBATCH --job-name=VERIFY_LIBERO_LOADER
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma1@alumni.cmu.edu
#SBATCH --output=logs/verify_libero_loader_%j.out
#SBATCH --error=logs/verify_libero_loader_%j.err

# === Set headless rendering for LIBERO ===
export MUJOCO_GL=osmesa

# === Direct Python path (bypass conda activate) ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Debug info ===
echo "=========================================="
echo "LIBERO Loader GPU Verification"
echo "=========================================="
echo "Job started at: $(date)"
echo "Python path: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === Verify Python exists ===
if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not found or not executable at $PYTHON"
    exit 1
fi

# === Change to project root ===
cd /data1/ma1/Ishaq/ump-vla

# === Run verification script ===
echo ""
echo "Running LIBERO loader verification..."
echo ""
$PYTHON code/scripts/verify_libero_loader_gpu.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ VERIFICATION PASSED"
    echo "✅ LIBERO loader is working correctly on GPU node"
else
    echo "❌ VERIFICATION FAILED - Exit code: $EXIT_CODE"
fi
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

exit $EXIT_CODE



