#!/bin/bash
# =============================================================================
# setup_env.sh -- One-time environment setup for the CurrentPrediction project
#
# Run this script once on a submit node (submit-a/b/c) to create the conda
# environment before submitting any SLURM jobs.
#
# Usage:
#   ssh jonesm25@submit.hpc.engr.oregonstate.edu
#   cd /nfs/stak/users/jonesm25/CurrentPrediction
#   bash setup_env.sh
# =============================================================================

set -e

PROJECT_DIR="/nfs/stak/users/jonesm25/CurrentPrediction"

echo "Loading conda module..."
module load conda

echo "Creating conda environment from environment.yml..."
conda env create -f "$PROJECT_DIR/environment.yml"

echo ""
echo "Environment 'currentprediction' created successfully."
echo ""
echo "Next steps:"
echo "  1. Activate it and log in to Weights & Biases:"
echo "     conda activate currentprediction"
echo "     wandb login"
echo "     (This stores your API key in ~/.netrc so SLURM jobs can use it.)"
echo ""
echo "  2. Create the logs directory:"
echo "     mkdir -p $PROJECT_DIR/logs"
echo ""
echo "  3. Submit a training job:"
echo "     sbatch $PROJECT_DIR/train.sbatch"
