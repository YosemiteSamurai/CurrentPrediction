#!/bin/bash
#SBATCH --job-name=finalize
#SBATCH --output=logs/finalize-%j.out
#SBATCH --error=logs/finalize-%j.err
#SBATCH --partition=eecs
#SBATCH --account=eecs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00

# This script merges per-task results from the simulation array job and builds the final dataset.

DESIGN=${DESIGN:-2inv}
DATASET=${DATASET:-dataset2}

PYTHON=/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python
PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction

cd $PROJECT_DIR
mkdir -p logs

echo "Merging simulation results for $DESIGN, $DATASET..."
$PYTHON -u python/run_sims.py --design $DESIGN --dataset $DATASET --finalize

echo "Finalization complete. Dataset is ready."
