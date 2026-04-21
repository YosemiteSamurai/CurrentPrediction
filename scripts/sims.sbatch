#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=logs/sim-%A_%a.out
#SBATCH --error=logs/sim-%A_%a.err
#SBATCH --partition=eecs
#SBATCH --account=eecs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:30:00
# Array range is set dynamically by submit_sims.sh via --array flag

DESIGN=${DESIGN:-2inv}
DATASET=${DATASET:-dataset2}

PYTHON=/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python
PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction

cd $PROJECT_DIR
mkdir -p logs

echo "Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX starting on $(hostname)"
$PYTHON -u python/run_sims.py \
    --design $DESIGN \
    --dataset $DATASET \
    --task-id $SLURM_ARRAY_TASK_ID
echo "Task $SLURM_ARRAY_TASK_ID done"
