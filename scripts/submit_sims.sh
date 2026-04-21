#!/bin/bash
# submit_sims.sh -- Submit a SLURM array job for SPICE simulations followed
#                   by a finalize job that merges results and builds the dataset.
#
# Usage:
#   bash submit_sims.sh [design] [dataset]
#   bash submit_sims.sh 2inv dataset2

DESIGN=${1:-2inv}
DATASET=${2:-dataset2}

PYTHON=/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python
PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction

cd $PROJECT_DIR
mkdir -p logs

# Count how many array tasks are needed (one per model×pvt×skew combination)
NUM_TASKS=$($PYTHON python/run_sims.py --design $DESIGN --dataset $DATASET --count-tasks)
MAX_IDX=$((NUM_TASKS - 1))

echo "Submitting $NUM_TASKS simulation tasks for design '$DESIGN', dataset '$DATASET'..."

# Submit the array job
ARRAY_JOB=$(sbatch --parsable \
    --array=0-${MAX_IDX} \
    --export=DESIGN=$DESIGN,DATASET=$DATASET \
    sims.sbatch)
echo "Array job ID: $ARRAY_JOB  (tasks 0-${MAX_IDX})"

# Submit the finalize job, runs only after every array task succeeds
FINAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$ARRAY_JOB \
    --export=DESIGN=$DESIGN,DATASET=$DATASET \
    finalize.sbatch)
echo "Finalize job ID: $FINAL_JOB  (depends on $ARRAY_JOB)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Array logs:    logs/sim-${ARRAY_JOB}_*.out"
echo "Finalize log:  logs/finalize-${FINAL_JOB}.out"
