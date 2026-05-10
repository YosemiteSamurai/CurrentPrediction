#!/bin/bash
# =============================================================================
# test_env.sbatch -- Smoke-test the cluster conda environment
#
# Submits a tiny SLURM job that activates the currentprediction conda
# environment and prints its Python/package versions, so you can verify
# the environment is reachable from the compute nodes before running any
# real training or simulation jobs.
# =============================================================================

#SBATCH --job-name=test_env
#SBATCH --output=logs/test_env.out
#SBATCH --error=logs/test_env.err
#SBATCH --partition=eecs
#SBATCH --account=eecs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:05:00


module load conda
source activate currentprediction
export PATH=$CONDA_PREFIX/bin:$PATH

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Checking for pandas..."
python -c "import pandas; print('pandas is available')" || echo 'pandas NOT available'
echo "Checking for ngspice..."
which ngspice && ngspice -v || echo 'ngspice NOT available in conda env'
