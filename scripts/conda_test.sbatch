#!/bin/bash
#SBATCH --job-name=conda-test
#SBATCH --output=/nfs/stak/users/jonesm25/CurrentPrediction/logs/conda-test-%j.out
#SBATCH --error=/nfs/stak/users/jonesm25/CurrentPrediction/logs/conda-test-%j.err
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --account=eecs
#SBATCH --partition=eecs

# Minimal conda environment test

echo "SLURM script started: $(date)"
module load conda
echo "Conda module loaded: $(date)"
source activate currentprediction
echo "Conda env activated: $(date)"
which python
python -c "import numpy; import torch; print('Numpy:', numpy.__version__, 'Torch:', torch.__version__)"
echo "Test script finished: $(date)"
