# =============================================================================
# run_sims.py -- SPICE Simulation Automation
#
# Automates SPICE circuit simulations across process, voltage, and temperature (PVT)
# corners and device parameter sweeps. Generates simulation results, parses outputs,
# and creates datasets and matrices for machine learning (GNN) training. Handles
# cleaning, netlist selection, and robust path management for flexible execution.
#
# Usage:
#   python run_sims.py --design <design_name> [--dataset <name>] [--clean [all]]
#
# Example:
#   python run_sims.py --design 2inv --dataset dataset2 --clean
#   python run_sims.py --design 2inv --clean all
# =============================================================================

import subprocess
import os
import sys
import random
import json
import time
import numpy as np
import argparse
import datetime
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if os.getcwd() != project_root:
    os.chdir(project_root)

seed = (int(time.time() * 1e6) ^ os.getpid() ^ int(datetime.datetime.now().microsecond)) % (2**32 - 1)
np.random.seed(seed)
random.seed(seed)

try:
    from python.parse_results import create_dataset, save_dataset, parse_spice_log

except ModuleNotFoundError:
    from parse_results import create_dataset, save_dataset, parse_spice_log

parser = argparse.ArgumentParser(description="Run SPICE simulations and generate dataset.")
parser.add_argument('--design', type=str, required=True, help='Design netlist name (without .sp)')
parser.add_argument('--dataset', type=str, default='dataset2', help='Dataset base name (writes <name>.json)')  # original: help='Dataset base name (writes <name>.json and <name>.csv)'

parser.add_argument(

    '--clean',
    nargs='?',
    const='design',
    choices=['design', 'all'],
    help='Clean results before running: use --clean for current design only, or --clean all for all results'

)

parser.add_argument('--debug', action='store_true', help='Do not trim log files; keep full NGSPICE output')
parser.add_argument('--task-id', type=int, default=None,
    help='SLURM array task ID for parallel execution (0-based index into model×pvt×skew combos)')
parser.add_argument('--count-tasks', action='store_true',
    help='Print total number of array tasks needed and exit (used by submit_sims.sh)')
parser.add_argument('--count-sims', action='store_true',
    help='Print total number of simulations (combos x NUM_SAMPLES) and exit (used by submit_sims.sh)')
parser.add_argument('--finalize', action='store_true',
    help='Merge per-task metadata and build dataset (run after array job completes)')
parser.add_argument('--model', default=None,
    help='Restrict simulation to a single process model (e.g. 22nm_LP or 22nm_LP.pm); each dataset covers one model')
args = parser.parse_args()
dataset_name = os.path.splitext(args.dataset)[0]
# Only clear the screen for interactive runs. In SLURM batch jobs (--finalize,
# array tasks) and when stdout is captured (--count-tasks) there is no TTY/TERM,
# so `clear` would emit escape codes or print "TERM environment variable not set."
if sys.stdout.isatty():  # original: (unconditional)
    os.system('cls' if os.name == 'nt' else 'clear')

NGSPICE = "ngspice"
BASE_SPICE = os.path.join("designs", "template.sp")
MODELS_DIR = "models"
CORNERS_DIR = "corners"
OUT_DIR = "results"
SIMS_DIR = "sims"  # per-run SPICE logs (run_<design>_<id>.log) live here
MATRICES_DIR = "matrices"
DATASET_DIR = "datasets"  # original: DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 7))  # original: NUM_SAMPLES = 7
# Max simulations per SLURM array task. The total work (combos x NUM_SAMPLES)
# is split into contiguous chunks of this size so every array task finishes
# within the sims.sbatch wall-clock limit (~1 sim/sec -> 1200 sims ~= 20 min,
# comfortably inside 30 min). More samples just means more tasks, not longer
# ones. Override via the SIMS_PER_TASK env var if node speed differs.
SIMS_PER_TASK = int(os.environ.get("SIMS_PER_TASK", 1200))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SIMS_DIR, exist_ok=True)  # per-run SPICE logs
os.makedirs(MATRICES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if args.task_id is None:

    if args.clean == 'all':
        print("Cleaning all results for fresh run...")

        for file in os.listdir(OUT_DIR):
            file_path = os.path.join(OUT_DIR, file)

            if os.path.isfile(file_path):
                os.remove(file_path)

        # Per-run SPICE logs now live in sims/, so clear them there too.
        if os.path.isdir(SIMS_DIR):
            for file in os.listdir(SIMS_DIR):
                file_path = os.path.join(SIMS_DIR, file)

                if os.path.isfile(file_path):
                    os.remove(file_path)

    elif args.clean == 'design':

        print(f"Cleaning results for design '{args.design}'...")
        design_log_pattern = re.compile(rf"run_{re.escape(args.design)}_(\d+)\.log$")
        design_metadata = f"metadata_{args.design}.json"

        # Remove this design's metadata from results/.
        for file in os.listdir(OUT_DIR):
            file_path = os.path.join(OUT_DIR, file)

            if not os.path.isfile(file_path):
                continue

            if file == design_metadata:
                os.remove(file_path)

        # Remove this design's per-run SPICE logs from sims/.
        if os.path.isdir(SIMS_DIR):
            for file in os.listdir(SIMS_DIR):
                file_path = os.path.join(SIMS_DIR, file)

                if not os.path.isfile(file_path):
                    continue

                if design_log_pattern.fullmatch(file):
                    os.remove(file_path)

DESIGN_NETLIST = os.path.join("designs", args.design + '.sp')

with open(BASE_SPICE) as f:

    for line in f:

        if line.strip().startswith('.include') and os.path.basename(DESIGN_NETLIST) in line:

            netlist_name = line.strip().split()[-1].replace('"', '')
            break

    else:
        netlist_name = os.path.basename(DESIGN_NETLIST)

# Suppress this banner in --count-tasks/--count-sims mode so the captured
# stdout is just the integer that submit_sims.sh reads.
if not (args.count_tasks or args.count_sims):  # original: if not args.count_tasks:
    print(f"Simulating {netlist_name} to create a dataset...\n")

PROCESS_PARAM_RANGES = {

    "22nm_HP.pm": {
        "WN1": (0.2e-6, 8e-6), "WP1": (0.2e-6, 12e-6),
        "WN2": (0.2e-6, 8e-6), "WP2": (0.2e-6, 12e-6),
        "L1": (20e-9, 200e-9), "L2": (20e-9, 200e-9),
        "VDD": (0.7, 1.3), "TEMP": (-20, 125)
    },

    "22nm_LP.pm": {
        "WN1": (0.2e-6, 8e-6), "WP1": (0.2e-6, 12e-6),
        "WN2": (0.2e-6, 8e-6), "WP2": (0.2e-6, 12e-6),
        "L1": (20e-9, 200e-9), "L2": (20e-9, 200e-9),
        "VDD": (0.7, 1.3), "TEMP": (-20, 125)
    },

    "32nm_HP.pm": {
        "WN1": (0.25e-6, 10e-6), "WP1": (0.25e-6, 15e-6),
        "WN2": (0.25e-6, 10e-6), "WP2": (0.25e-6, 15e-6),
        "L1": (30e-9, 250e-9), "L2": (30e-9, 250e-9),
        "VDD": (0.8, 1.4), "TEMP": (-20, 125)
    },

    "32nm_LP.pm": {
        "WN1": (0.25e-6, 10e-6), "WP1": (0.25e-6, 15e-6),
        "WN2": (0.25e-6, 10e-6), "WP2": (0.25e-6, 15e-6),
        "L1": (30e-9, 250e-9), "L2": (30e-9, 250e-9),
        "VDD": (0.8, 1.4), "TEMP": (-20, 125)
    },

    "45nm_HP.pm": {
        "WN1": (0.3e-6, 12e-6), "WP1": (0.3e-6, 18e-6),
        "WN2": (0.3e-6, 12e-6), "WP2": (0.3e-6, 18e-6),
        "L1": (40e-9, 300e-9), "L2": (40e-9, 300e-9),
        "VDD": (0.9, 1.5), "TEMP": (-20, 125)
    },

    "45nm_LP.pm": {
        "WN1": (0.3e-6, 12e-6), "WP1": (0.3e-6, 18e-6),
        "WN2": (0.3e-6, 12e-6), "WP2": (0.3e-6, 18e-6),
        "L1": (40e-9, 300e-9), "L2": (40e-9, 300e-9),
        "VDD": (0.9, 1.5), "TEMP": (-20, 125)
    },

    "65nm_bulk.pm": {
        "WN1": (0.4e-6, 15e-6), "WP1": (0.4e-6, 20e-6),
        "WN2": (0.4e-6, 15e-6), "WP2": (0.4e-6, 20e-6),
        "L1": (60e-9, 400e-9), "L2": (60e-9, 400e-9),
        "VDD": (1.0, 1.6), "TEMP": (-20, 125)
    },

    "90nm_bulk.pm": {
        "WN1": (0.5e-6, 20e-6), "WP1": (0.5e-6, 25e-6),
        "WN2": (0.5e-6, 20e-6), "WP2": (0.5e-6, 25e-6),
        "L1": (80e-9, 500e-9), "L2": (80e-9, 500e-9),
        "VDD": (1.1, 1.8), "TEMP": (-20, 125)
    },

    "130nm_bulk.pm": {
        "WN1": (0.6e-6, 25e-6), "WP1": (0.6e-6, 30e-6),
        "WN2": (0.6e-6, 25e-6), "WP2": (0.6e-6, 30e-6),
        "L1": (120e-9, 600e-9), "L2": (120e-9, 600e-9),
        "VDD": (1.2, 2.0), "TEMP": (-20, 125)
    }
}

def rand_param(lo, hi):
    return random.uniform(lo, hi)

def gen_params(model):

    model_name = os.path.basename(model)

    if model_name not in PROCESS_PARAM_RANGES:

        print(f"Warning: No parameter ranges defined for {model_name}, using 45nm defaults")
        model_name = "45nm_HP.pm"

    ranges = PROCESS_PARAM_RANGES[model_name]

    return {

        k: rand_param(v[0], v[1])
        for k, v in ranges.items()

    }

def write_netlist(params, model, pvt_corner, skew_corner, run_id, design_netlist):

    with open(BASE_SPICE) as f:
        txt = f.read()

    design_filename = os.path.basename(design_netlist)

    def replace_include(match):

        before = match.group(0)
        prefix = match.group(1)
        replaced = f'{prefix}{design_filename}"'
        return replaced
    
    txt = re.sub(r'(^\s*\.include\s+")([^"/\\]+\.sp)"', replace_include, txt, flags=re.MULTILINE)

    filled = txt.format(
        model=model,
        pvt_corner=pvt_corner,
        skew_corner=skew_corner,
        **params
    )

    filled = re.sub(r'^\s*\.print\s+tran\s+I\(VDD\)\s+I\(Vin\)\s*$', '', filled, flags=re.MULTILINE | re.IGNORECASE)
    filled = re.sub(r'^\s*\.print\s+tran\s+I\(vmeas_vdd\)[^\n]*$', '', filled, flags=re.MULTILINE | re.IGNORECASE)
    filled = re.sub(r'\n{2,}', '\n', filled)

    print_lines = [
        '.print tran V(vdd) V(in) V(target) V(out)',
        '.print tran I(vmeas_vdd) I(vmeas_in) I(vmeas_out) I(vmeas_target)',
        '.meas tran I_VDD_MAX MAX vdd#branch',
        '.meas tran I_Vin_MAX MAX vin#branch'
    ]

    if '.end' in filled:
        filled = filled.replace('.end', '\n' + '\n'.join(print_lines) + '\n.end')

    else:
        filled += '\n' + '\n'.join(print_lines) + '\n.end'

    fname = os.path.join("designs", f"run_{args.design}_{run_id}.sp")

    with open(fname, "w") as f:
        f.write(filled)

    return fname

def run_ngspice(netlist, out_file):
    """Run one ngspice simulation.

    Returns True on success, False on a transient failure (non-zero exit or
    unparseable output) so the caller can retry with fresh params. A missing
    ngspice binary is fatal and is allowed to propagate.
    """

    cmd = [
        NGSPICE,
        "-b",
        "-o", out_file,
        netlist
    ]

    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:  # original: result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Warning: ngspice exited {e.returncode} for {netlist}; retrying with new params")
        return False

    if not args.debug:

        try:
            results = parse_spice_log(out_file)
        except Exception as e:
            print(f"Warning: could not parse ngspice output {out_file}: {e}; retrying")
            return False

        with open(out_file, 'w') as f:
            json.dump(results, f)

    return True

def regenerate_matrices():

    models = [
        os.path.join(MODELS_DIR, f)
        for f in sorted(os.listdir(MODELS_DIR))
        if f.endswith('.pm') and os.path.isfile(os.path.join(MODELS_DIR, f))
    ]

    # Each dataset covers a single process model. Restrict to the selected one
    # so NUM_TASKS = pvt × skew matches the pipeline's expected_rows.
    if args.model:
        target = args.model if args.model.endswith('.pm') else args.model + '.pm'
        models = [m for m in models if os.path.basename(m) == target]
        if not models:
            print(f"Error: model '{args.model}' not found in {MODELS_DIR}/")
            exit(1)

    pvt_corners = [
        os.path.join(CORNERS_DIR, f)
        for f in os.listdir(CORNERS_DIR)
        if f.endswith(".sp") and not f.startswith("skew_")
    ]

    skew_corners = [
        os.path.join(CORNERS_DIR, f)
        for f in os.listdir(CORNERS_DIR)
        if f.startswith("skew_") and f.endswith(".sp")
    ]

    all_combos = [(m, p, s) for m in sorted(models) for p in sorted(pvt_corners) for s in sorted(skew_corners)]
    total_simulations_all = len(all_combos) * NUM_SAMPLES

    if args.count_sims:

        # Total sims across all combos; submit_sims.sh uses this to size the
        # array (and to auto-bump SIMS_PER_TASK if it would exceed MaxArraySize).
        print(total_simulations_all)
        exit(0)

    if args.count_tasks:

        # Number of array tasks needed to cover every simulation in bounded
        # SIMS_PER_TASK-sized chunks (ceil division).
        num_tasks = (total_simulations_all + SIMS_PER_TASK - 1) // SIMS_PER_TASK
        print(max(num_tasks, 1))
        exit(0)

    # Determine which slice of the flattened (combo x NUM_SAMPLES) work list
    # this invocation runs. In array mode each --task-id owns a contiguous
    # SIMS_PER_TASK-sized slice so it finishes in bounded wall-clock time; a
    # full local run (no --task-id) runs everything.
    if args.task_id is not None:  # original: single combo per task (all_combos[task_id])
        chunk_start = args.task_id * SIMS_PER_TASK
        chunk_end = min(chunk_start + SIMS_PER_TASK, total_simulations_all)
        if chunk_start >= total_simulations_all:
            print(f"Task {args.task_id}: nothing to do "
                  f"(start {chunk_start} >= total {total_simulations_all}); exiting.")
            exit(0)
    else:
        chunk_start = 0
        chunk_end = total_simulations_all

    # Map the [chunk_start, chunk_end) global range onto (combo, lo, hi)
    # segments, where global run ids lo..hi-1 belong to that combo. run_id is
    # the global index so output filenames never collide across tasks.
    task_work = []
    for _ci, _combo in enumerate(all_combos):
        _c_lo = _ci * NUM_SAMPLES
        _c_hi = _c_lo + NUM_SAMPLES
        _lo = max(chunk_start, _c_lo)
        _hi = min(chunk_end, _c_hi)
        if _lo < _hi:
            task_work.append((_combo, _lo, _hi))

    if args.finalize:

        import glob as _glob
        task_files = sorted(_glob.glob(os.path.join(OUT_DIR, f'metadata_{args.design}_task_*.json')))
        print(f"Found {len(task_files)} per-task metadata files to merge...")
        all_entries = []

        for tf in task_files:

            with open(tf) as fh:

                try:

                    d = json.load(fh)
                    all_entries.extend(d if isinstance(d, list) else [d])

                except Exception:
                    pass

        if all_entries:

            metadata_path = os.path.join(OUT_DIR, f'metadata_{args.design}.json')
            existing = []

            if os.path.exists(metadata_path):

                with open(metadata_path) as fh:

                    try:
                        existing = json.load(fh)

                    except Exception:
                        pass

            existing_tuples = set(tuple(sorted(e['params'].items())) for e in existing)
            new_entries = [e for e in all_entries if tuple(sorted(e['params'].items())) not in existing_tuples]
            combined = existing + new_entries

            with open(metadata_path, 'w') as fh:
                json.dump(combined, fh, indent=2)

            print(f"Merged {len(new_entries)} new entries from {len(task_files)} task files -> {metadata_path}")

            for tf in task_files:
                os.remove(tf)

        print("Parsing results and creating dataset...")
        dataset, total_sims, _process_tag = create_dataset()  # original: dataset, total_sims = create_dataset()
        save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, f'{dataset_name}.json'))
        print()
        exit(0)

    # This invocation runs exactly the sims in task_work: a bounded chunk in
    # array mode, or everything in a full local run. run_id is the global sim
    # index, so per-run output filenames never collide across array tasks.
    total_simulations = chunk_end - chunk_start  # original: len(models)*len(pvt_corners)*len(skew_corners)*NUM_SAMPLES
    print(f"Starting {total_simulations} simulations...")

    if total_simulations > 500:
        print(f"0/{total_simulations} simulations complete", end='', flush=True)

    meta_data = []

    start_time = time.time()

    session_run_idx = 0

    for (model, pvt_corner, skew_corner), _g_lo, _g_hi in task_work:  # original: for model in models: for pvt_corner...: for skew_corner...:

                n_sims = _g_hi - _g_lo

                model_name = os.path.basename(model)

                if model_name not in PROCESS_PARAM_RANGES:

                    print(f"Warning: No parameter ranges defined for {model_name}, using 45nm defaults")
                    model_name = "45nm_HP.pm"

                ranges = PROCESS_PARAM_RANGES[model_name]
                new_unique_params = set()
                max_attempts = n_sims * 100  # Avoid infinite loop  # original: NUM_SAMPLES * 100
                attempts = 0
                produced = 0  # successful, recorded sims for this combo segment

                while produced < n_sims and attempts < max_attempts:  # original: while len(new_unique_params) < NUM_SAMPLES and attempts < max_attempts:

                    params = {k: np.random.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
                    param_tuple = tuple(sorted(params.items()))
                    attempts += 1

                    if param_tuple in new_unique_params:  # original: if param_tuple in existing_params_set or param_tuple in new_unique_params:
                        continue

                    run_id = _g_lo + produced  # global sim index  # original: (run_id incremented per sim)

                    netlist = write_netlist(
                        params, model, pvt_corner, skew_corner, run_id, DESIGN_NETLIST
                    )

                    out_file = os.path.join(
                        SIMS_DIR,  # original: OUT_DIR
                        f"run_{args.design}_{run_id}.log"
                    )

                    # Run the simulation first; only count and record it once
                    # it has succeeded. A transient ngspice/parse failure is
                    # discarded (with its netlist/log) and retried with fresh
                    # params, so each combo segment still reaches n_sims.
                    sim_ok = run_ngspice(netlist, out_file)

                    try:
                        os.remove(netlist)
                    except Exception as e:
                        print(f"Warning: Could not delete netlist {netlist}: {e}")

                    if not sim_ok:
                        try:
                            if os.path.exists(out_file):
                                os.remove(out_file)
                        except Exception:
                            pass
                        continue

                    new_unique_params.add(param_tuple)

                    model_basename = os.path.basename(model).replace('.pm', '')
                    pvt_name = os.path.basename(pvt_corner).replace('.sp', '')
                    skew_name = os.path.basename(skew_corner).replace('skew_', '').replace('.sp', '')

                    if '_' in model_basename:
                        process_node, process_option = model_basename.rsplit('_', 1)
                    else:
                        process_node = model_basename
                        process_option = ""

                    session_run_idx += 1
                    if total_simulations > 500:
                        if (session_run_idx) % 100 == 0 or (session_run_idx) == total_simulations:
                            print(f"\r{session_run_idx}/{total_simulations} simulations complete", end='', flush=True)
                    else:
                        print(
                            f"Run {session_run_idx:3d}/{total_simulations}: {process_node:8s} "
                            f"{process_option:4s} PVT:{pvt_name:8s} Skew:{skew_name:2s} | "
                            f"VDD={params['VDD']:.2f}V T={params['TEMP']:.0f}°C "
                            f"W1P={params.get('WP1', float('nan')):.2e} "
                            f"W1N={params.get('WN1', float('nan')):.2e} "
                            f"W2P={params.get('WP2', float('nan')):.2e} "
                            f"W2N={params.get('WN2', float('nan')):.2e}"
                        )

                    meta_data.append({
                        "run": run_id,
                        "design": args.design,
                        "model": model,
                        "pvt_corner": pvt_corner,
                        "skew_corner": skew_corner,
                        "params": params,
                        "output": out_file
                    })

                    produced += 1

                if produced < n_sims:
                    msg = (
                        f"Only {produced}/{n_sims} successful unique "
                        f"simulations for {model_name}, {pvt_corner}, {skew_corner} "
                        f"after {attempts} attempts."
                    )
                    if args.task_id is not None:
                        # Fail the array task so the afterok dependency keeps
                        # finalize (and training) from running until this
                        # chunk is complete.
                        print(f"ERROR: {msg}")
                        exit(1)
                    print(f"Warning: {msg}")

    end_time = time.time()
    elapsed = end_time - start_time
    avg_time = elapsed / total_simulations if total_simulations > 0 else 0

    if total_simulations > 500:
        print()  # Print newline after progress overwrite

    print(f"\nTotal simulation time: {elapsed:.2f} seconds")
    print(f"Average time per simulation: {avg_time:.2f} seconds")

    if args.task_id is not None:
        task_meta_path = os.path.join(OUT_DIR, f"metadata_{args.design}_task_{args.task_id}.json")
        with open(task_meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
    else:
        metadata_path = os.path.join(OUT_DIR, f"metadata_{args.design}.json")
        if args.clean:
            with open(metadata_path, "w") as f:
                json.dump(meta_data, f, indent=2)
        else:
            prev_meta = []
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        prev_meta = json.load(f)
                except Exception:
                    prev_meta = []
            prev_param_tuples = set(tuple(sorted(entry["params"].items())) for entry in prev_meta)
            new_entries = [entry for entry in meta_data if tuple(sorted(entry["params"].items())) not in prev_param_tuples]
            combined_meta = prev_meta + new_entries
            with open(metadata_path, "w") as f:
                json.dump(combined_meta, f, indent=2)

    if total_simulations > 500:
        print()

    if args.task_id is None:
        num_existing = 0  # original: len(existing_params_set) if not args.clean else 0
        num_new = len(meta_data)
        import os as _os_env
        _os_env.environ['NEW_SIM_COUNT'] = str(num_new)
        _os_env.environ['TOTAL_SIM_COUNT'] = str(num_existing + num_new)
        print("Parsing results and creating dataset...")
        dataset, total_sims, _process_tag = create_dataset()  # original: dataset, total_sims = create_dataset()
        save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, f'{dataset_name}.json'))
        print()
        print()


regenerate_matrices()
