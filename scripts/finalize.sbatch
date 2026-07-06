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
MODEL=${MODEL:-}

PYTHON=/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python
PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction

# Put the conda env's bin/ on PATH so bare `ngspice` (called by run_sims.py)
# resolves on the compute node, which does not inherit the login-node PATH.
export PATH="$(dirname "$PYTHON"):$PATH"

cd $PROJECT_DIR
mkdir -p logs

echo "Merging simulation results for $DESIGN, $DATASET..."
$PYTHON -u python/run_sims.py --design $DESIGN --dataset $DATASET --model $MODEL --finalize
rc=$?

# On success, tidy the intermediate artifacts now that the dataset is built:
# per-run SPICE logs and per-task array logs. The merged simulation manifest
# (metadata_<design>.json) is moved next to the dataset it describes so it is
# kept but out of results/. (Runs here instead of on the login node since the
# training job is a dependent SLURM job and the pipeline exits after submit.)
if [ $rc -eq 0 ]; then
    echo "Cleaning up per-run artifacts..."
    rm -f sims/* logs/sim-*.out logs/sim-*.err
    mkdir -p datasets
    mv -f "results/metadata_${DESIGN}.json" "datasets/metadata_${DESIGN}.json" 2>/dev/null || true
fi

echo "Finalization complete (exit $rc). Dataset is ready."
# Propagate the exit code so training's afterok dependency only releases
# when the dataset was built successfully.
exit $rc