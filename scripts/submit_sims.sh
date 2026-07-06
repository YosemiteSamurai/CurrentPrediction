#!/bin/bash
# submit_sims.sh -- Submit a SLURM array job for SPICE simulations followed
#                   by a finalize job that merges results and builds the dataset.
#
# Usage:
#   bash submit_sims.sh [design] [dataset] [num_samples] [model]
#   bash submit_sims.sh 2inv dataset2 4000 22nm_LP

DESIGN=${1:-2inv}
DATASET=${2:-dataset2}
NUM_SAMPLES=${3:-7}
MODEL=${4:-}
# Export so run_sims.py --count-sims below reads the SAME NUM_SAMPLES the array
# tasks use. Without this it would fall back to the default (7), undercounting
# total sims -> too few array tasks -> each task's chunk covers only the first
# corner combo (the cause of the 1200-row single-combo dataset bug).
export NUM_SAMPLES

# Max sims per array task. Total work (combos × NUM_SAMPLES) is split into
# chunks of this size so every task fits in the 30-min sims.sbatch limit;
# more samples just means more tasks, not longer ones. Exported so the
# --count-tasks call below and each array task compute the same chunking.
SIMS_PER_TASK=${SIMS_PER_TASK:-1200}
export SIMS_PER_TASK

PYTHON=/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python
PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction

cd $PROJECT_DIR
mkdir -p logs

# Clear stale metadata so finalize builds the dataset from ONLY this run's
# sims. Otherwise the merged metadata_${DESIGN}.json accumulates entries from
# previous runs (mixing old corner combos / sample counts into the dataset).
rm -f "results/metadata_${DESIGN}.json" results/metadata_${DESIGN}_task_*.json

# Total simulations for the selected single process model.
TOTAL_SIMS=$($PYTHON python/run_sims.py --design $DESIGN --dataset $DATASET --model $MODEL --count-sims)

# SLURM caps the number of array tasks (MaxArraySize; index range 0..N-1).
# Query it so we can guarantee the array fits; fall back to 1001 if unknown.
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/{gsub(/ /,"",$2); print $2}')
[ -z "$MAX_ARRAY" ] && MAX_ARRAY=1001

# ceil(TOTAL_SIMS / SIMS_PER_TASK)
NUM_TASKS=$(( (TOTAL_SIMS + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))

# If the chunking would need more array tasks than SLURM allows, grow the
# chunk size just enough to fit (ceil(TOTAL_SIMS / MAX_ARRAY)). Tasks may then
# run longer than 30 min, so warn the operator.
if [ "$NUM_TASKS" -gt "$MAX_ARRAY" ]; then
    SIMS_PER_TASK=$(( (TOTAL_SIMS + MAX_ARRAY - 1) / MAX_ARRAY ))
    NUM_TASKS=$(( (TOTAL_SIMS + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
    echo "WARNING: $TOTAL_SIMS sims would exceed MaxArraySize=$MAX_ARRAY at the" >&2
    echo "         requested chunk size; bumped SIMS_PER_TASK to $SIMS_PER_TASK" >&2
    echo "         ($NUM_TASKS tasks). Individual tasks may run longer than 30 min;" >&2
    echo "         raise sims.sbatch --time if they time out." >&2
fi
export SIMS_PER_TASK

MAX_IDX=$((NUM_TASKS - 1))
NUM_SIMS=$TOTAL_SIMS

echo "Submitting $NUM_TASKS simulation tasks for design '$DESIGN', dataset '$DATASET'..."

# Submit the array job
ARRAY_JOB=$(sbatch --parsable \
    --array=0-${MAX_IDX} \
    --export=DESIGN=$DESIGN,DATASET=$DATASET,NUM_SAMPLES=$NUM_SAMPLES,MODEL=$MODEL,SIMS_PER_TASK=$SIMS_PER_TASK \
    scripts/sims.sbatch)
echo "Array job ID: $ARRAY_JOB  (tasks 0-${MAX_IDX})"

# Submit the finalize job, runs only after every array task succeeds
FINAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$ARRAY_JOB \
    --export=DESIGN=$DESIGN,DATASET=$DATASET,MODEL=$MODEL \
    scripts/finalize.sbatch)
echo "Finalize job ID: $FINAL_JOB  (depends on $ARRAY_JOB)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Array logs:    logs/sim-${ARRAY_JOB}_*.out"
echo "Finalize log:  logs/finalize-${FINAL_JOB}.out"

# Machine-readable summary consumed by integrated_pipeline.py's
# _parse_submit_output(). Must be plain KEY=VALUE lines.
echo "ARRAY_JOB=$ARRAY_JOB"
echo "FINAL_JOB=$FINAL_JOB"
echo "NUM_TASKS=$NUM_TASKS"
echo "JOBS_PER_TASK=$SIMS_PER_TASK"
echo "NUM_SIMS=$NUM_SIMS"
