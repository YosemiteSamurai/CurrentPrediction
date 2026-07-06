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
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# classgpu: class-access GPU partition (1-day limit)
#   nodes: dgx2 (V100 32GB), cn-r (A40 48GB), cn-s (A40 48GB), cn-gpu3/4 (RTX8000 44GB)
# Alternatives if classgpu is unavailable:
#   preempt   -- wider GPU pool but jobs can be preempted
#   ampere    -- A40-only nodes, 2-day limit (requires group membership)
#SBATCH --account=eecs

# #SBATCH --partition=eecs   # Default: eecs (A40, RTX8000, V100, etc.)
#SBATCH --partition=ampere # Uncomment for A40-only nodes (2-day limit, group membership required)
# #SBATCH --partition=gpu    # Uncomment for general GPU pool (may include older GPUs)
# #SBATCH --partition=dgx2   # Uncomment for V100 32GB nodes only
#SBATCH --exclude=cn-r-1,cn-r-2,cn-gpu3


###############################################################################
# Environment setup
###############################################################################

# Initialize modules in non-interactive shells (required on some clusters)
if [ -f /etc/profile.d/modules.sh ]; then
	source /etc/profile.d/modules.sh
fi

set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO" >&2' ERR

ts() {
	date "+%Y-%m-%d %H:%M:%S"
}

log() {
	echo "[$(ts)] $*"
}


log "Job ID: $SLURM_JOB_ID"
log "Node:   $SLURM_NODELIST"
log "Start:  $(date)"
log "Current working directory: $(pwd)"
log "User: $(whoami)"
log "Hostname: $(hostname)"
log "PATH: $PATH"
log "Conda base: $(which conda || echo 'conda not found')"


CONDA_ENV_PY="/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python"
PYTHON_BIN=""

# Load conda module (needed for conda to be available, but we call the
# environment's Python binary directly to avoid conda activate/run issues
# in non-interactive SLURM shells).
if module load conda; then
	echo "Conda module loaded: $(date)"
	which conda || echo '[WARN] conda still not found after module load' >&2
	echo "PATH after module load: $PATH"
	conda info --envs || echo "[WARN] conda info --envs failed"
	conda env list || echo "[WARN] conda env list failed"
	echo "Attempting to activate environment: currentprediction"
	if source activate currentprediction; then
		echo "Conda env activated: $(date)"
		PYTHON_BIN="$(which python)"
	else
		echo "[WARN] Failed to activate conda environment 'currentprediction'; falling back to direct env python" >&2
	fi
else
	echo "[WARN] Could not load conda module; falling back to direct env python" >&2
fi

if [ -z "$PYTHON_BIN" ]; then
	if [ -x "$CONDA_ENV_PY" ]; then
		PYTHON_BIN="$CONDA_ENV_PY"
		log "Using fallback python: $PYTHON_BIN"
	else
		echo "[ERROR] No usable Python found. Tried conda activation and $CONDA_ENV_PY" >&2
		exit 2
	fi
fi

# Avoid occasional torch/MKL import hangs on shared HPC nodes.
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
log "Thread env: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

PROJECT_DIR="/nfs/stak/users/jonesm25/CurrentPrediction"
mkdir -p "$PROJECT_DIR/logs"


# Archive the finalize (simulation) job's logs under results/, keyed by
# process model AND run tag (e.g. results/22nm_LP_baseline_simulation.{out,err}).
# This training job runs afterok:finalize, so the finalize logs are complete
# and closed by now. Empty .err files are not archived (and any stale copy is
# removed). Guarded so archiving never aborts training.
archive_finalize_logs() {
	local dataset="${DATASET:-}"
	local model="${dataset#dataset_}"
	local tag="${RUN_TAG:-baseline}"  # untagged runs archive as <model>_baseline_* (matches the Python convention)
	local suffix="${model}_${tag}"

	local src_out=""
	local src_err=""
	if [ -n "${FINAL_JOB:-}" ] && [ -f "$PROJECT_DIR/logs/finalize-${FINAL_JOB}.out" ]; then
		src_out="$PROJECT_DIR/logs/finalize-${FINAL_JOB}.out"
		src_err="$PROJECT_DIR/logs/finalize-${FINAL_JOB}.err"
	else
		# Fall back to the most recent finalize-*.out if FINAL_JOB is unset.
		src_out="$(ls -t "$PROJECT_DIR"/logs/finalize-*.out 2>/dev/null | head -1)"
		[ -n "$src_out" ] && src_err="${src_out%.out}.err"
	fi

	if [ -n "$src_out" ] && [ -f "$src_out" ]; then
		cp -f "$src_out" "$PROJECT_DIR/results/${suffix}_simulation.out" \
			&& log "Archived finalize stdout -> results/${suffix}_simulation.out"
	fi

	local dst_err="$PROJECT_DIR/results/${suffix}_simulation.err"
	if [ -n "$src_err" ] && [ -f "$src_err" ]; then
		if [ -s "$src_err" ]; then
			cp -f "$src_err" "$dst_err" \
				&& log "Archived finalize stderr -> results/${suffix}_simulation.err"
		else
			rm -f "$dst_err"
		fi
	fi
}
archive_finalize_logs || log "[WARN] finalize log archiving failed (non-fatal)"

# Archive this training job's own SLURM log under results/, keyed by process
# model AND run tag (e.g. results/22nm_LP_baseline_training.{out,err}). Run
# from an EXIT trap so the copy happens whether training succeeds or fails.
# The source log is still open, so it captures everything up to the trap.
archive_training_logs() {
	local dataset="${DATASET:-}"
	local model="${dataset#dataset_}"
	local tag="${RUN_TAG:-baseline}"  # untagged runs archive as <model>_baseline_* (matches the Python convention)
	local suffix="${model}_${tag}"

	local src_out="$PROJECT_DIR/logs/slurm-${SLURM_JOB_ID}.out"
	local src_err="$PROJECT_DIR/logs/slurm-${SLURM_JOB_ID}.err"

	if [ -f "$src_out" ]; then
		cp -f "$src_out" "$PROJECT_DIR/results/${suffix}_training.out" \
			&& log "Archived training stdout -> results/${suffix}_training.out"
	fi

	local dst_err="$PROJECT_DIR/results/${suffix}_training.err"
	if [ -f "$src_err" ] && [ -s "$src_err" ]; then
		cp -f "$src_err" "$dst_err" \
			&& log "Archived training stderr -> results/${suffix}_training.err"
	else
		rm -f "$dst_err"
	fi
}
trap 'archive_training_logs || true' EXIT


	PRIVACY_MODE="${PRIVACY_MODE:-neither}"
	# Single unified entry point handles all PRIVACY_MODE values
	# (neither|dp|sl|both) in-process; there is no separate split backend.
	SWEEP_SCRIPT="$PROJECT_DIR/python/sweep.py"
	if [ ! -f "$SWEEP_SCRIPT" ]; then
		echo "[ERROR] Missing sweep.py at $SWEEP_SCRIPT" >&2
		exit 4
	fi
	if [ ! -d "$PROJECT_DIR/datasets" ]; then
		echo "[ERROR] Missing dataset directory at $PROJECT_DIR/datasets" >&2
		exit 4
	fi

	log "Run options (from integrated_pipeline.py):"
	log "  DATASET        = ${DATASET:-<unset>}"
	log "  DESIGN         = ${DESIGN:-<unset>}"
	log "  RUN_TAG        = ${RUN_TAG:-<unset>}"
	log "  PRIVACY_MODE   = ${PRIVACY_MODE}"
	log "  INFERENCE_ONLY = ${INFERENCE_ONLY:-0}"
	log "  RUN_ATTACKS    = ${RUN_ATTACKS:-0}"
	if [ "$PRIVACY_MODE" = "dp" ] || [ "$PRIVACY_MODE" = "both" ]; then
		log "  DP_NOISE_MULT  = ${DP_NOISE_MULTIPLIER:-0.6}"
		log "  DP_MAX_GRAD    = ${DP_MAX_GRAD_NORM:-1.0}"
	fi

	log "About to run sweep.py: $PYTHON_BIN"
	LAUNCH_CMD="import runpy; runpy.run_path('$SWEEP_SCRIPT', run_name='__main__')"
	PY_IMPORT_FLAGS=()
	if [ "${IMPORT_TIME_TRACE:-0}" = "1" ]; then
		PY_IMPORT_FLAGS=(-X importtime)
		log "[launcher] import tracing enabled (-X importtime)"
	fi
	SWEEP_TIMEOUT_SECONDS="${SWEEP_TIMEOUT_SECONDS:-21600}"
	if command -v timeout >/dev/null 2>&1; then
		set +e
		timeout "$SWEEP_TIMEOUT_SECONDS" "$PYTHON_BIN" "${PY_IMPORT_FLAGS[@]}" -u -c "$LAUNCH_CMD" 2>&1
		rc=$?
		set -e
		if [ "$rc" -ne 0 ]; then
			if [ "$rc" -eq 124 ]; then
				echo "[ERROR] sweep.py startup/run timed out after ${SWEEP_TIMEOUT_SECONDS}s. Aborting early for diagnosis." >&2
			else
				echo "[ERROR] sweep.py failed with exit code $rc. See logs above." >&2
			fi
			exit 1
		fi
	else
		echo "[WARN] 'timeout' command not found; running sweep.py without startup watchdog" >&2
		if ! "$PYTHON_BIN" "${PY_IMPORT_FLAGS[@]}" -u -c "$LAUNCH_CMD" 2>&1; then
			echo "[ERROR] sweep.py failed. See above for details." >&2
			exit 1
		fi
	fi
	log "sweep.py finished: $?"

log "Finished: $(date)"
