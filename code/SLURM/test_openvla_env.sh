#!/bin/bash
#SBATCH --job-name=TEST_OPENVLA_ENV
#SBATCH --partition=gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --mcs-label=mcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ma1@alumni.cmu.edu
#SBATCH --output=logs/test_openvla_env_%j.out
#SBATCH --error=logs/test_openvla_env_%j.err

# === Set headless rendering for LIBERO ===
export MUJOCO_GL=osmesa

# === Direct Python path (bypass conda activate) ===
PYTHON="/data1/ma1/envs/upa-vla/bin/python3.10"

# === Debug info ===
echo "=========================================="
echo "openVLA Environment SLURM Test"
echo "=========================================="
echo "Job started at: $(date)"
echo "Python path: $PYTHON"
echo "Working directory: $(pwd)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# === Verify Python exists and is executable ===
if [ ! -f "$PYTHON" ]; then
    echo "❌ ERROR: Python not found at $PYTHON"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "❌ ERROR: Python not executable at $PYTHON"
    exit 1
fi

echo "✓ Python executable found and accessible"

# === Test Python version ===
echo ""
echo "[TEST] Python version:"
$PYTHON --version

# === Test PyTorch import ===
echo ""
echo "[TEST] PyTorch import:"
$PYTHON -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>&1

# === Run comprehensive test script ===
echo ""
echo "=========================================="
echo "Running comprehensive test suite..."
echo "=========================================="
cd /data1/ma1/Ishaq/ump-vla
$PYTHON code/scripts/test_openvla_env.py

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Environment is accessible from SLURM!"
    echo "✅ FINAL GREENLIGHT - Ready for production use"
else
    echo "❌ SOME TESTS FAILED - Exit code: $TEST_EXIT_CODE"
fi
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

exit $TEST_EXIT_CODE



