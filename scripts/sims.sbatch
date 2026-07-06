#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=logs/sim-%A_%a.out
#SBATCH --error=logs/sim-%A_%a.err
#SBATCH --partition=eecs
#SBATCH --account=eecs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Each task runs a bounded SIMS_PER_TASK-sized chunk of the total work
# (~1 sim/sec, default 1200 sims ≈ 20 min), so it always fits in 30 min.
#SBATCH --time=0:30:00
# Array range is set dynamically by submit_sims.sh via --array flag

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

echo "Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX starting on $(hostname)"
$PYTHON -u python/run_sims.py \
    --design $DESIGN \
    --dataset $DATASET \
    --model $MODEL \
    --task-id $SLURM_ARRAY_TASK_ID
rc=$?
echo "Task $SLURM_ARRAY_TASK_ID done (exit $rc)"
# Propagate the Python exit code so a short/failed task fails the array
# task, keeping the afterok dependency from releasing finalize.
exit $rc
