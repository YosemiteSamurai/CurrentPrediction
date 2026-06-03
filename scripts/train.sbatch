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


CONDA_ENV_PY="/nfs/stak/users/renya/jonesm25/.conda/envs/currentprediction/bin/python"
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


	log "About to run hello world test: $PYTHON_BIN"
	if ! "$PYTHON_BIN" -c "print('Hello world from SLURM batch!')"; then
		echo "[ERROR] Python hello world test failed" >&2
		exit 3
	fi
	log "Hello world test finished: $?"

	log "Running lightweight preflight (paths and inputs)"
	if [ ! -f "$PROJECT_DIR/python/sweep.py" ]; then
		echo "[ERROR] Missing sweep.py at $PROJECT_DIR/python/sweep.py" >&2
		exit 4
	fi
	if [ ! -d "$PROJECT_DIR/dataset" ]; then
		echo "[ERROR] Missing dataset directory at $PROJECT_DIR/dataset" >&2
		exit 4
	fi
	log "[preflight] sweep.py found"
	log "[preflight] dataset dir found"
	log "[preflight] DATASET='${DATASET:-<unset>}' DESIGN='${DESIGN:-<unset>}'"

	log "About to run sweep.py: $PYTHON_BIN"
	LAUNCH_CMD="import os, runpy, datetime; p='$PROJECT_DIR/python/sweep.py'; now=lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); print(f'[{now()}] [launcher] python alive', flush=True); print(f'[{now()}] [launcher] sweep exists: {os.path.exists(p)}', flush=True); runpy.run_path(p, run_name='__main__')"
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
