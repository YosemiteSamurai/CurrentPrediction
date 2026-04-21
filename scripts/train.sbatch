#!/bin/bash
# =============================================================================
# train.sbatch -- SLURM batch script for CurrentPrediction GCN training
#
# Submit from a submit node:
#   sbatch /nfs/stak/users/jonesm25/CurrentPrediction/train.sbatch
#
# Check status:
#   squeue -u jonesm25
#
# View available partitions:
#   sinfo
# =============================================================================

#SBATCH --job-name=currentprediction
#SBATCH --output=/nfs/stak/users/jonesm25/CurrentPrediction/logs/slurm-%j.out
#SBATCH --error=/nfs/stak/users/jonesm25/CurrentPrediction/logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# classgpu: class-access GPU partition (1-day limit)
#   nodes: dgx2 (V100 32GB), cn-r (A40 48GB), cn-s (A40 48GB), cn-gpu3/4 (RTX8000 44GB)
# Alternatives if classgpu is unavailable:
#   preempt   -- wider GPU pool but jobs can be preempted
#   ampere    -- A40-only nodes, 2-day limit (requires group membership)
#SBATCH --account=eecs
#SBATCH --partition=eecs

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"

# Load conda module (needed for conda to be available, but we call the
# environment's Python binary directly to avoid conda activate/run issues
# in non-interactive SLURM shells).
module load conda

PROJECT_DIR="/nfs/stak/users/jonesm25/CurrentPrediction"
PYTHON="/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python"
WANDB="/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/wandb"
mkdir -p "$PROJECT_DIR/logs"

"$PYTHON" -u "$PROJECT_DIR/python/sweep.py"

echo "Finished: $(date)"
