#!/bin/bash
# =============================================================================
# resubmit_failed_sims.sh -- ONE-SHOT recovery for the 22nm_LP run
#
# Background
# ----------
# Job 20214046 submitted a 991-task array (JOBS_PER_TASK=101) for dataset
# dataset_22nm_LP. Tasks 166..327 (162 tasks, contiguous) failed at startup
# because run_sims.py was being edited on NFS while those ranks were
# importing it, producing a `NameError: name 'argparse' is not defined`
# before argparse was even reached. This left 16,362 missing run IDs
# (tasks * JOBS_PER_TASK = 162 * 101 = 16,362 -- matches the observed gap
# in sims/metadata_2inv.json and dataset/dataset_22nm_LP.json).
#
# What this script does
# ---------------------
# Resubmits only tasks 166..327 with the EXACT same JOBS_PER_TASK and
# NUM_SIMS as the original run, then chains a finalize job afterok. Each
# resubmitted task regenerates its 101 metadata_2inv_<run>.json files;
# finalize reglobs the full set and rewrites metadata_2inv.json and
# dataset/dataset_22nm_LP.json.
#
# Run from the project root on the HPC node:
#   bash scripts/resubmit_failed_sims.sh
# =============================================================================
set -e

DESIGN=2inv
DATASET=dataset_22nm_LP
MODEL=22nm_LP.pm
NUM_SAMPLES=7

# Pin these to the original run. Recomputing them would produce a different
# slicing and corrupt the run-id mapping.
JOBS_PER_TASK=101
NUM_SIMS=100000

FAILED_ARRAY="166-327"

PROJECT_DIR=/nfs/stak/users/jonesm25/CurrentPrediction
cd "$PROJECT_DIR"
mkdir -p logs sims

echo "Resubmitting failed tasks: --array=$FAILED_ARRAY"
echo "JOBS_PER_TASK=$JOBS_PER_TASK  NUM_SIMS=$NUM_SIMS"

ARRAY_JOB=$(sbatch --parsable \
    --array=$FAILED_ARRAY \
    --export=DESIGN=$DESIGN,DATASET=$DATASET,NUM_SAMPLES=$NUM_SAMPLES,MODEL=$MODEL,JOBS_PER_TASK=$JOBS_PER_TASK,NUM_SIMS=$NUM_SIMS \
    scripts/sims.sbatch)
echo "Array job: $ARRAY_JOB"

FINAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$ARRAY_JOB \
    --export=DESIGN=$DESIGN,DATASET=$DATASET,MODEL=$MODEL \
    scripts/finalize.sbatch)
echo "Finalize job: $FINAL_JOB  (dependency afterok:$ARRAY_JOB)"

echo ""
echo "DO NOT edit python/run_sims.py or any file under python/ until the"
echo "array job is fully PENDING/RUNNING (squeue -u \$USER)."
