# =============================================================================
# integrated_pipeline.py -- Interactive end-to-end driver
#
# Top-level orchestration script that walks the user through a full run:
#   1. Pick a process model from models/ (e.g. 22nm_LP.pm).
#   2. If a matching dataset already exists, show its row count and offer
#      to reuse it -- on reuse, skip straight to training.
#   3. Otherwise prompt for the target dataset size and derive num_samples
#      per PVT x skew combination.
#   4. Optionally clean logs/ (preserving slurm* files by moving them into
#      results/) and always clean sims/ (per-run SPICE artifacts), then
#      submit scripts/submit_sims.sh to run the ngspice array job +
#      finalize job on SLURM.
#   5. Wait for the finalized dataset JSON, sanity-check its row count,
#      and if short, auto-resubmit only the failed array tasks (up to
#      MAX_RETRIES times) before aborting.
#   6. Remove per-run artifacts, then submit scripts/train.sbatch to
#      train the GAN.
#
# Designed to run from a submit node; all SLURM interaction is via sbatch.
# =============================================================================


import os
import glob
import math
import subprocess
import re
import sys
import time
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="End-to-end pipeline for CurrentPrediction")
parser.add_argument('--no-slurm', action='store_true', help='Run training locally without SLURM')
parser.add_argument('--run-tag', default='', help='Run name tag appended to saved artifacts (default: baseline)')  # original: parser.add_argument('--run-tag', default='', help='Run name tag appended to saved artifacts (use baseline for default filenames)')
parser.add_argument('--resume-inference-only', action='store_true', help='Skip training and rerun only inference/artifact export from existing checkpoint')
parser.add_argument('--privacy-mode', default='', help='Privacy mode for training: neither, dp, sl, or both (default: prompt -> neither)')
parser.add_argument('--model', default='', help='Process model name (e.g., 22nm_LP or 22nm_LP.pm) to skip selection prompt')
parser.add_argument('--re-simulate', default='', help='Dataset action when dataset exists: yes/no (yes=re-simulate, no=reuse existing dataset)')
parser.add_argument('--min-dataset-size', type=int, default=0, help='Minimum dataset size for simulation (positive integer)')
parser.add_argument('--clean-logs', default='', help='Clean logs directory before submit: yes/no')
parser.add_argument('--monitor', action='store_true', help='After submitting training, run monitor_squeue.py and wait until that specific JOBID finishes')
args = parser.parse_args()

# Clear the screen
os.system('cls' if os.name == 'nt' else 'clear')

# Step 1: List available process models
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model_files = glob.glob(os.path.join(MODELS_DIR, '*.pm'))

def train_model(dataset_path, design='2inv', inference_only=False):  # original: def train_model(dataset_path, design='2inv'):
    """Train the GAN on the given dataset, either via SLURM or locally."""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    process_name = dataset_name[len('dataset_'):] if dataset_name.startswith('dataset_') else dataset_name
    run_suffix = f"_{EFFECTIVE_RUN_TAG}" if EFFECTIVE_RUN_TAG else ""
    mode_label = "inference-only" if inference_only else "training"
    project_root_local = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_sbatch = os.path.join(project_root_local, 'scripts', 'train.sbatch')
    sweep_py = os.path.join(project_root_local, 'python', 'sweep.py')
    split_sweep_py = os.path.join(project_root_local, 'privacy', 'split', 'sweep.py')
    local_sweep_py = split_sweep_py if EFFECTIVE_PRIVACY_MODE in ('sl', 'both') else sweep_py

    if args.no_slurm:
        print(f"[NO-SLURM MODE] Running {mode_label} locally: {local_sweep_py}")  # original: print(f"[NO-SLURM MODE] Running training locally: {sweep_py}")
        print(f"[NO-SLURM MODE] Privacy mode: {EFFECTIVE_PRIVACY_MODE}")
        # Import sweep.py as a module and call main(config)
        import importlib.util
        import types
        import sys as _sys
        sweep_spec = importlib.util.spec_from_file_location("sweep", local_sweep_py)  # original: sweep_spec = importlib.util.spec_from_file_location("sweep", sweep_py)
        if sweep_spec is None or sweep_spec.loader is None:
            print(f"[ERROR] Could not load training module spec from {local_sweep_py}", file=sys.stderr)
            sys.exit(1)
        sweep = importlib.util.module_from_spec(sweep_spec)
        _sys.modules["sweep"] = sweep
        sweep_spec.loader.exec_module(sweep)
        # Build config object as in sweep.py
        config = sweep.SimpleNamespace(
            batch_size=32,
            model='block',
            lr=0.0001,
            layers=3,
            hidden_dim=64,
            heads=4,
            test_size=0.3,
            epochs=100,
            edges_per_graph=7,
            target_edge_idx=3,
        )
        # Set DATASET and DESIGN environment variables for sweep.py
        os.environ['DATASET'] = dataset_name
        os.environ['DESIGN'] = design
        os.environ['RUN_TAG'] = EFFECTIVE_RUN_TAG  # original: os.environ['RUN_TAG'] = args.run_tag  # original: os.environ['DESIGN'] = design
        os.environ['INFERENCE_ONLY'] = '1' if inference_only else ''
        os.environ['PRIVACY_MODE'] = EFFECTIVE_PRIVACY_MODE
        try:
            sweep.main(config)
        except Exception as e:
            print(f"[ERROR] Local {mode_label} failed: {e}", file=sys.stderr)  # original: print(f"[ERROR] Local training failed: {e}", file=sys.stderr)
            sys.exit(1)
        if inference_only:
            print(f"[NO-SLURM MODE] Inference-only complete. Check privacy/predictions_{process_name}{run_suffix}.csv and privacy/inference_outputs_{process_name}{run_suffix}.npz.")  # original: print(f"[NO-SLURM MODE] Inference-only complete. Check results/predictions_{process_name}{run_suffix}.csv and results/inference_outputs_{process_name}{run_suffix}.npz.")
        else:
            print(f"[NO-SLURM MODE] Training complete. Check results/model_{process_name}{run_suffix}.pt for output.")
        if args.monitor:
            print("[NO-SLURM MODE] --monitor ignored (no SLURM JOBID to monitor).")
        return None
    else:
        # SLURM rejects scripts with DOS line endings. If the file was checked out
        # (or edited) on Windows, normalize it to LF in-place before submitting.
        try:
            with open(train_sbatch, 'rb') as f:
                data = f.read()
            if b'\r\n' in data:
                with open(train_sbatch, 'wb') as f:
                    f.write(data.replace(b'\r\n', b'\n'))
        except Exception as e:
            print(f"Warning: Could not normalize line endings on {train_sbatch}: {e}")

        cmd = [
            'sbatch', '--parsable',
            f'--export=DATASET={dataset_name},DESIGN={design},RUN_TAG={EFFECTIVE_RUN_TAG},INFERENCE_ONLY={1 if inference_only else 0},PRIVACY_MODE={EFFECTIVE_PRIVACY_MODE}',  # original: f'--export=DATASET={dataset_name},DESIGN={design},RUN_TAG={EFFECTIVE_RUN_TAG},INFERENCE_ONLY={1 if inference_only else 0}',
            train_sbatch,
        ]
        print(f"Submitting {mode_label} job for dataset '{dataset_name}'...")  # original: print(f"Submitting training job for dataset '{dataset_name}'...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            print(f"sbatch failed (exit {result.returncode}).", file=sys.stderr)
            if result.stdout.strip():
                print(f"stdout: {result.stdout.strip()}", file=sys.stderr)
            if result.stderr.strip():
                print(f"stderr: {result.stderr.strip()}", file=sys.stderr)
            print(f"command: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(result.returncode)
        job_id = result.stdout.strip()
        print(f"\nTraining job submitted: job ID {job_id}")
        if inference_only:
            print("  Mode:          inference-only (reuses existing checkpoint)")
        print(f"  Privacy mode:  {EFFECTIVE_PRIVACY_MODE}")
        print(f"  Monitor with:  squeue -j {job_id}")
        print(f"  Output log:    logs/slurm-{job_id}.out")
        print(f"  Error log:     logs/slurm-{job_id}.err")
        print(f"  Checkpoint:    results/model_{process_name}{run_suffix}.pt (when complete)\n")  # original: print(f"  Checkpoint:    results/model_{process_name}.pt (when complete)\n")
        if inference_only:
            print(f"  Predictions:   privacy/predictions_{process_name}{run_suffix}.csv")  # original: print(f"  Predictions:   results/predictions_{process_name}{run_suffix}.csv")
            print(f"  Inference NPZ: privacy/inference_outputs_{process_name}{run_suffix}.npz\n")  # original: print(f"  Inference NPZ: results/inference_outputs_{process_name}{run_suffix}.npz\n")

        if args.monitor:
            monitor_py = os.path.join(os.path.dirname(__file__), 'monitor_squeue.py')
            monitor_cmd = [
                sys.executable,
                monitor_py,
                '--monitor',
                '--job-id',
                job_id,
                '--model',
                process_name,
                '--run-tag',
                EFFECTIVE_RUN_TAG,
            ]
            print(f"Starting job-specific monitor for JOBID {job_id}...")
            print(f"Command: {' '.join(monitor_cmd)}")
            subprocess.run(monitor_cmd, check=True)
        return job_id

def extract_nm(filename):
    match = re.match(r'(\d+)nm', filename)
    return int(match.group(1)) if match else float('inf')


def _normalize_resimulate_choice(raw_value):
    value = (raw_value or '').strip().lower()
    aliases = {
        'y': True,
        'yes': True,
        'true': True,
        '1': True,
        'r': True,
        'resimulate': True,
        're-simulate': True,
        'n': False,
        'no': False,
        'false': False,
        '0': False,
        'u': False,
        'use': False,
        'reuse': False,
    }
    if value not in aliases:
        raise ValueError("--re-simulate must be yes/no (or y/n, true/false, use/reuse)")
    return aliases[value]


def _normalize_yes_no(raw_value, arg_name):
    value = (raw_value or '').strip().lower()
    aliases = {
        'y': True,
        'yes': True,
        'true': True,
        '1': True,
        'n': False,
        'no': False,
        'false': False,
        '0': False,
    }
    if value not in aliases:
        raise ValueError(f"{arg_name} must be yes/no (or y/n, true/false)")
    return aliases[value]


def _resolve_model_path(model_arg, available_model_files):
    raw = (model_arg or '').strip()
    if not raw:
        return None

    target = raw if raw.endswith('.pm') else f"{raw}.pm"
    target_lower = target.lower()

    for model_path in available_model_files:
        if os.path.basename(model_path).lower() == target_lower:
            return model_path

    return None

# Sort by feature size (nm), largest last
model_files.sort(key=lambda path: extract_nm(os.path.basename(path)))

print("")
print("Available manufacturing process models:\n")
for idx, model_path in enumerate(model_files):
    print(f"  [{idx+1}] {os.path.basename(model_path)}")
print("")

selected_model = _resolve_model_path(args.model, model_files)
if selected_model is not None:
    print(f"Using model from --model: {os.path.basename(selected_model)}")
else:
    if args.model.strip():
        print(f"ERROR: --model '{args.model}' did not match any file in {MODELS_DIR}", file=sys.stderr)
        sys.exit(2)
    while True:
        try:
            model_idx = int(input("Select a process model by number: ")) - 1
            if 0 <= model_idx < len(model_files):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")
    selected_model = model_files[model_idx]

if selected_model is None:
    print("ERROR: model selection failed", file=sys.stderr)
    sys.exit(2)

selected_model_name = os.path.basename(selected_model)
print(f"Selected process model: {selected_model_name}")

# Build dataset name from process model and set up paths
process_name = os.path.splitext(selected_model_name)[0]
dataset_name = f"dataset_{process_name}"
design_name = '2inv'  # You can prompt for this or generalize later

def _normalize_run_name(raw_name):
    """Return (requested_name, effective_run_tag). baseline is an explicit run tag."""  # original: """Return (requested_name, effective_run_tag). baseline maps to default filenames."""
    requested = (raw_name or '').strip()
    if requested == '':
        return 'baseline', 'baseline'  # original: return 'baseline', ''
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', requested)
    safe = re.sub(r'_+', '_', safe).strip('_')
    if safe == '':
        return 'baseline', 'baseline'  # original: return 'baseline', ''
    if safe.lower() == 'baseline':
        return 'baseline', 'baseline'
    return safe, safe

def _normalize_privacy_mode(raw_mode):
    """Return canonical privacy mode: neither, dp, sl, or both."""
    mode = (raw_mode or '').strip().lower()
    aliases = {
        '': 'neither',
        'n': 'neither',
        'none': 'neither',
        'neither': 'neither',
        'dp': 'dp',
        'd': 'dp',
        'sl': 'sl',
        'split': 'sl',
        's': 'sl',
        'both': 'both',
        'b': 'both',
    }
    if mode not in aliases:
        raise ValueError("Privacy mode must be one of: neither, dp, sl, both")
    return aliases[mode]

if args.run_tag.strip():
    RUN_NAME, EFFECTIVE_RUN_TAG = _normalize_run_name(args.run_tag)
    print(f"Using run name from --run-tag: {RUN_NAME}")
else:
    _user_run_name = input("Enter run name [baseline]: ").strip()
    RUN_NAME, EFFECTIVE_RUN_TAG = _normalize_run_name(_user_run_name)
    print(f"Using run name: {RUN_NAME}")

if args.privacy_mode.strip():
    EFFECTIVE_PRIVACY_MODE = _normalize_privacy_mode(args.privacy_mode)
else:
    _privacy_prompt = input("Select privacy mode [neither/dp/sl/both] (default: neither): ").strip()
    EFFECTIVE_PRIVACY_MODE = _normalize_privacy_mode(_privacy_prompt)
print(f"Using privacy mode: {EFFECTIVE_PRIVACY_MODE}")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(project_root, 'dataset', f'{dataset_name}.json')
logs_dir = os.path.join(project_root, 'logs')
sims_dir = os.path.join(project_root, 'sims')
results_dir = os.path.join(project_root, 'results')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(sims_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Step 2: Check for existing dataset BEFORE asking about dataset size, so
# we can skip straight to training if the user wants to reuse the existing
# file.
if os.path.exists(dataset_path):
    import json
    try:
        with open(dataset_path, 'r') as _f:
            existing_rows = len(json.load(_f))
        size_str = f"{existing_rows:,} rows"
    except Exception as e:
        size_str = f"could not read size: {e}"
    if args.re_simulate.strip():
        try:
            should_resimulate = _normalize_resimulate_choice(args.re_simulate)
            print(f"Using dataset action from --re-simulate: {'re-simulate' if should_resimulate else 'use existing'}")
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        resp = input(
            f"\nDataset file '{dataset_path}' already exists ({size_str}).\n"
            f"Use this file to train the model, or re-simulate? "
            f"[U=use / R=re-simulate]: "
        ).strip().lower()
        should_resimulate = not (resp == '' or resp.startswith('u'))
    if not should_resimulate:
        print(f"\nUsing existing dataset: {dataset_path}.\nProceeding to training phase...\n")
        checkpoint_path = os.path.join(results_dir, f"model_{process_name}_{EFFECTIVE_RUN_TAG}.pt")
        inference_only = False
        if args.resume_inference_only:
            inference_only = True
        elif args.re_simulate.strip():
            inference_only = False
        elif os.path.exists(checkpoint_path):
            resume_resp = input(
                f"Found existing checkpoint for this run name: {checkpoint_path}\n"
                f"Resume inference/artifact export only (skip retraining)? [y/N]: "
            ).strip().lower()
            if resume_resp.startswith('y'):
                inference_only = True
        if inference_only and not os.path.exists(checkpoint_path):
            print(f"ERROR: Inference-only requested but checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        train_model(dataset_path, design=design_name, inference_only=inference_only)
        sys.exit(0)
    else:
        try:
            os.remove(dataset_path)
            print(f"Deleted existing dataset: {dataset_path}")
        except Exception as e:
            print(f"Warning: Could not delete {dataset_path}: {e}")

# Step 3: Count PVT and skew corners
CORNERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'corners')
pvt_corners = [f for f in os.listdir(CORNERS_DIR) if f.endswith('.sp') and not f.startswith('skew_')]
skew_corners = [f for f in os.listdir(CORNERS_DIR) if f.startswith('skew_') and f.endswith('.sp')]

num_pvt = len(pvt_corners)
num_skew = len(skew_corners)
total_combinations = num_pvt * num_skew
print(f"\nFound {num_pvt} PVT corners and {num_skew} skew corners, totaling {total_combinations} combinations.")
print("")

# Step 3b: Prompt for minimum dataset size
if args.min_dataset_size > 0:
    min_dataset_size = args.min_dataset_size
    print(f"Using minimum dataset size from --min-dataset-size: {min_dataset_size}")
else:
    while True:
        try:
            min_dataset_size = int(input("Enter minimum dataset size: "))
            if min_dataset_size > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

# Step 3c: Calculate NUM_SAMPLES
num_samples = math.ceil(min_dataset_size / (num_pvt * num_skew))
print(f"A dataset of {min_dataset_size} samples means {num_samples} simulations per combination.")
print(f"Dataset will be named: {dataset_name}.json\n")

# Step 4: Clean stale artifacts and submit the simulation + finalize jobs.
submit_sims_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'submit_sims.sh'))

def _parse_submit_output(text):
    """Parse the trailing KEY=VALUE lines printed by submit_sims.sh."""
    info = {}
    for line in text.splitlines():
        m = re.match(r'^(ARRAY_JOB|FINAL_JOB|JOBS_PER_TASK|NUM_SIMS|NUM_TASKS)=(\S+)$', line.strip())
        if m:
            info[m.group(1)] = m.group(2)
    return info


def _find_failed_task_ids(array_job_id):
    """Return sorted list of array-task indices whose .err is non-empty.

    Matches logs/sim-<array_job_id>_<task>.err produced by sims.sbatch.
    """
    failed = []
    pat = re.compile(rf'^sim-{re.escape(str(array_job_id))}_(\d+)\.err$')
    try:
        with os.scandir(logs_dir) as it:
            for de in it:
                m = pat.match(de.name)
                if m and de.stat().st_size > 0:
                    failed.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return sorted(failed)


def _compress_task_ids(ids):
    """Compress a sorted list of ints into SLURM --array syntax (ranges+CSV)."""
    if not ids:
        return ""
    parts = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = x
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _resubmit_failed(failed_ids, jobs_per_task, num_sims):
    """Resubmit only the failed array tasks + a dependent finalize.

    Pins JOBS_PER_TASK and NUM_SIMS to the original run's values so the
    (task -> run_id) slicing in run_sims.py stays consistent.
    Returns (new_array_job_id, new_final_job_id).
    """
    array_spec = _compress_task_ids(failed_ids)
    print(f"  resubmitting tasks --array={array_spec}  "
          f"(JOBS_PER_TASK={jobs_per_task}, NUM_SIMS={num_sims})")

    export = (
        f"DESIGN={design_name},DATASET={dataset_name},"
        f"NUM_SAMPLES={num_samples},MODEL={selected_model_name},"
        f"JOBS_PER_TASK={jobs_per_task},NUM_SIMS={num_sims}"
    )

    array_res = subprocess.run(
        ['sbatch', '--parsable', f'--array={array_spec}',
         f'--export={export}', 'scripts/sims.sbatch'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, cwd=project_root, check=True,
    )
    new_array_job = array_res.stdout.strip()

    final_export = (
        f"DESIGN={design_name},DATASET={dataset_name},"
        f"MODEL={selected_model_name}"
    )
    final_res = subprocess.run(
        ['sbatch', '--parsable',
         f'--dependency=afterok:{new_array_job}',
         f'--export={final_export}', 'scripts/finalize.sbatch'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, cwd=project_root, check=True,
    )
    new_final_job = final_res.stdout.strip()
    return new_array_job, new_final_job


def _wait_for_dataset(path, poll=10):
    """Block until the dataset file exists."""
    while not os.path.exists(path):
        time.sleep(poll)


def _read_dataset_rows(path):
    with open(path, 'r') as _f:
        return len(json.load(_f))


# Pass the model filename as the 4th argument
cmd = [submit_sims_path, design_name, dataset_name, str(num_samples), selected_model_name]

# Clean up stale artifacts right before launching the simulation array.
# Deferred to this point so reusing an existing dataset (which exited
# earlier) doesn't wipe the logs directory. Files beginning with 'slurm'
# are preserved by moving them into results/ rather than deleting.
if args.clean_logs.strip():
    try:
        should_clean_logs = _normalize_yes_no(args.clean_logs, '--clean-logs')
        print(f"Using logs cleanup setting from --clean-logs: {'yes' if should_clean_logs else 'no'}")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
else:
    resp = input(f"Clean out the logs directory before submitting? [Y/n]: ").strip().lower()
    should_clean_logs = (resp == '' or resp.startswith('y'))

if should_clean_logs:
    import shutil
    for f in os.listdir(logs_dir):
        fp = os.path.join(logs_dir, f)
        if not os.path.isfile(fp):
            continue
        try:
            if f.startswith('slurm'):
                dest = os.path.join(results_dir, f)
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(fp, dest)
            else:
                os.remove(fp)
        except Exception as e:
            print(f"Warning: Could not clean {fp}: {e}")

# Always clean out sims/ (per-run SPICE artifacts) before a fresh submission.
for f in os.listdir(sims_dir):
    fp = os.path.join(sims_dir, f)
    try:
        if os.path.isfile(fp):
            os.remove(fp)
    except Exception as e:
        print(f"Warning: Could not delete {fp}: {e}")


# If --no-slurm is set, print a warning and skip simulation phase (unless you have a local fallback)
if args.no_slurm:
    print("[NO-SLURM MODE] Simulation phase is not supported without SLURM. Please generate the dataset manually if needed.")
else:
    submit_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, check=True)
    sys.stdout.write(submit_res.stdout)
    run_info = _parse_submit_output(submit_res.stdout)
    if not {'ARRAY_JOB', 'JOBS_PER_TASK', 'NUM_SIMS'}.issubset(run_info):
        print("ERROR: Could not parse submit_sims.sh output for ARRAY_JOB / "
              "JOBS_PER_TASK / NUM_SIMS. Retry-on-shortfall will be unavailable.",
              file=sys.stderr)
        sys.exit(1)
    array_job_id = run_info['ARRAY_JOB']
    final_job_id = run_info.get('FINAL_JOB')
    jobs_per_task = int(run_info['JOBS_PER_TASK'])
    num_sims = int(run_info['NUM_SIMS'])
    print(f"\nSLURM job submitted for simulation and dataset generation: {dataset_name}.json\n")

# Step 5: Wait for dataset file to exist, then sanity-check its row
# count. If the run came up short (e.g. some array tasks crashed),
# identify the failed tasks from logs/sim-<ARRAY_JOB>_*.err and resubmit
# only those -- up to MAX_RETRIES times. Pin JOBS_PER_TASK and NUM_SIMS
# so the (task->run_id) slicing stays consistent across passes.
import json

expected_rows = num_pvt * num_skew * num_samples
MAX_RETRIES = 2

print(f"Waiting for dataset file to be created: {dataset_path}")
_wait_for_dataset(dataset_path)

for attempt in range(MAX_RETRIES + 1):
    try:
        actual_rows = _read_dataset_rows(dataset_path)
    except Exception as e:
        print(f"ERROR: Could not read dataset file {dataset_path}: {e}",
              file=sys.stderr)
        sys.exit(1)

    if actual_rows >= expected_rows:
        print(f"Dataset complete: {actual_rows} rows "
              f"(expected {expected_rows}).")
        break

    shortfall = expected_rows - actual_rows
    print(f"\nDataset is short by {shortfall} rows "
          f"(have {actual_rows}, expected {expected_rows}).")

    if attempt >= MAX_RETRIES:
        print(f"ERROR: Exhausted {MAX_RETRIES} retries; aborting before "
              f"training.", file=sys.stderr)
        sys.exit(1)

    failed_ids = _find_failed_task_ids(array_job_id)
    if not failed_ids:
        print("ERROR: Dataset is short but no failed SLURM tasks were "
              f"found under logs/sim-{array_job_id}_*.err. Cannot "
              "auto-recover; aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Retry {attempt + 1}/{MAX_RETRIES}: "
          f"{len(failed_ids)} failed tasks detected for job {array_job_id}.")
    try:
        array_job_id, final_job_id = _resubmit_failed(failed_ids, jobs_per_task, num_sims)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: sbatch failed during retry: {e.stderr or e.stdout}",
              file=sys.stderr)
        sys.exit(1)

    # Delete the stale dataset so the poll loop waits for finalize to
    # rewrite it with the combined results.
    try:
        os.remove(dataset_path)
    except OSError:
        pass
    print(f"Waiting for dataset to be rebuilt: {dataset_path}")
    _wait_for_dataset(dataset_path)

# Step 6: Dataset is complete; remove per-task sim logs and per-run
# SPICE artifacts so the workspace is tidy, then submit training.
print("Cleaning up per-run artifacts...")
for f in os.listdir(logs_dir):
    if f.startswith('sim'):
        fp = os.path.join(logs_dir, f)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception as e:
            print(f"Warning: Could not delete {fp}: {e}")

for f in os.listdir(sims_dir):
    fp = os.path.join(sims_dir, f)
    try:
        if os.path.isfile(fp):
            os.remove(fp)
    except Exception as e:
        print(f"Warning: Could not delete {fp}: {e}")

# Archive the finalize job's logs under results/ keyed by process model
# (e.g. results/finalize-22nm_LP.out). Empty .err files are deleted
# rather than archived. Falls back to scanning logs/ if FINAL_JOB was
# not captured (older submit_sims.sh output).
import shutil as _shutil
_final_src_out = None
_final_src_err = None
if final_job_id:
    _final_src_out = os.path.join(logs_dir, f"finalize-{final_job_id}.out")
    _final_src_err = os.path.join(logs_dir, f"finalize-{final_job_id}.err")
else:
    _finalize_outs = sorted(glob.glob(os.path.join(logs_dir, "finalize-*.out")),
                            key=os.path.getmtime)
    if _finalize_outs:
        _final_src_out = _finalize_outs[-1]
        _final_src_err = _final_src_out[:-4] + ".err"

if _final_src_out and os.path.isfile(_final_src_out):
    dst_out = os.path.join(results_dir, f"finalize-{process_name}.out")
    try:
        _shutil.copy2(_final_src_out, dst_out)
        print(f"Archived finalize stdout -> {dst_out}")
    except Exception as e:
        print(f"Warning: Could not copy {_final_src_out} -> {dst_out}: {e}")

dst_err = os.path.join(results_dir, f"finalize-{process_name}.err")
if _final_src_err and os.path.isfile(_final_src_err):
    if os.path.getsize(_final_src_err) == 0:
        # Empty stderr: don't archive; also remove any stale file from a
        # previous run of this same process model.
        if os.path.exists(dst_err):
            try:
                os.remove(dst_err)
            except Exception as e:
                print(f"Warning: Could not remove stale {dst_err}: {e}")
    else:
        try:
            _shutil.copy2(_final_src_err, dst_err)
            print(f"Archived finalize stderr -> {dst_err}")
        except Exception as e:
            print(f"Warning: Could not copy {_final_src_err} -> {dst_err}: {e}")

print("Ready! Proceeding to training phase...")
train_model(dataset_path, design=design_name)
