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

# Clear the screen
os.system('cls' if os.name == 'nt' else 'clear')

# Step 1: List available process models
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model_files = glob.glob(os.path.join(MODELS_DIR, '*.pm'))

def train_model(dataset_path, design='2inv'):
    """Submit scripts/train.sbatch as a SLURM job to train the GAN on the
    given dataset. Returns the SLURM job ID. Does not wait for completion."""
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    project_root_local = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_sbatch = os.path.join(project_root_local, 'scripts', 'train.sbatch')

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
        f'--export=DATASET={dataset_name},DESIGN={design}',
        train_sbatch,
    ]
    print(f"Submitting training job for dataset '{dataset_name}'...")
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
    print(f"  Monitor with:  squeue -j {job_id}")
    print(f"  Output log:    logs/slurm-{job_id}.out")
    print(f"  Error log:     logs/slurm-{job_id}.err")
    process_name = dataset_name[len('dataset_'):] if dataset_name.startswith('dataset_') else dataset_name
    print(f"  Checkpoint:    results/model_{process_name}.pt (when complete)\n")
    return job_id

def extract_nm(filename):
    match = re.match(r'(\d+)nm', filename)
    return int(match.group(1)) if match else float('inf')

# Sort by feature size (nm), largest last
model_files.sort(key=lambda path: extract_nm(os.path.basename(path)))

print("")
print("Available manufacturing process models:\n")
for idx, model_path in enumerate(model_files):
    print(f"  [{idx+1}] {os.path.basename(model_path)}")
print("")

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
print(f"Selected process model: {os.path.basename(selected_model)}")

# Build dataset name from process model and set up paths
process_name = os.path.splitext(os.path.basename(selected_model))[0]
dataset_name = f"dataset_{process_name}"
design_name = '2inv'  # You can prompt for this or generalize later

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
    resp = input(
        f"\nDataset file '{dataset_path}' already exists ({size_str}).\n"
        f"Use this file to train the model, or re-simulate? "
        f"[U=use / R=re-simulate]: "
    ).strip().lower()
    if resp == '' or resp.startswith('u'):
        print(f"\nUsing existing dataset: {dataset_path}.\nProceeding to training phase...\n")
        train_model(dataset_path, design=design_name)
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
        f"NUM_SAMPLES={num_samples},MODEL={os.path.basename(selected_model)},"
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
        f"MODEL={os.path.basename(selected_model)}"
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
cmd = [submit_sims_path, design_name, dataset_name, str(num_samples), os.path.basename(selected_model)]

# Clean up stale artifacts right before launching the simulation array.
# Deferred to this point so reusing an existing dataset (which exited
# earlier) doesn't wipe the logs directory. Files beginning with 'slurm'
# are preserved by moving them into results/ rather than deleting.
resp = input(f"Clean out the logs directory before submitting? [Y/n]: ").strip().lower()
if resp == '' or resp.startswith('y'):
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
