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
parser.add_argument('--dataset', type=str, default='dataset2', help='Dataset base name (writes <name>.json and <name>.csv)')

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
parser.add_argument('--finalize', action='store_true',
    help='Merge per-task metadata and build dataset (run after array job completes)')
args = parser.parse_args()
dataset_name = os.path.splitext(args.dataset)[0]
os.system('cls' if os.name == 'nt' else 'clear')

NGSPICE = "ngspice"
BASE_SPICE = os.path.join("designs", "template.sp")
MODELS_DIR = "models"
CORNERS_DIR = "corners"
OUT_DIR = "results"
MATRICES_DIR = "matrices"
DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
NUM_SAMPLES = 7

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MATRICES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if args.task_id is None:

    if args.clean == 'all':
        print("Cleaning all results for fresh run...")

        for file in os.listdir(OUT_DIR):
            file_path = os.path.join(OUT_DIR, file)

            if os.path.isfile(file_path):
                os.remove(file_path)

    elif args.clean == 'design':

        print(f"Cleaning results for design '{args.design}'...")
        design_log_pattern = re.compile(rf"run_{re.escape(args.design)}_(\d+)\.log$")
        design_metadata = f"metadata_{args.design}.json"

        for file in os.listdir(OUT_DIR):
            file_path = os.path.join(OUT_DIR, file)

            if not os.path.isfile(file_path):
                continue

            if design_log_pattern.fullmatch(file) or file == design_metadata:
                os.remove(file_path)

DESIGN_NETLIST = os.path.join("designs", args.design + '.sp')

with open(BASE_SPICE) as f:

    for line in f:

        if line.strip().startswith('.include') and os.path.basename(DESIGN_NETLIST) in line:

            netlist_name = line.strip().split()[-1].replace('"', '')
            break

    else:
        netlist_name = os.path.basename(DESIGN_NETLIST)

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

    cmd = [
        NGSPICE,
        "-b",
        "-o", out_file,
        netlist
    ]

    result = subprocess.run(cmd, check=True, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)

    if not args.debug:

        results = parse_spice_log(out_file)

        with open(out_file, 'w') as f:
            json.dump(results, f)

def regenerate_matrices():

    models = [
        os.path.join(MODELS_DIR, f)
        for f in sorted(os.listdir(MODELS_DIR))
        if f.endswith('.pm') and os.path.isfile(os.path.join(MODELS_DIR, f))
    ]

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

    if args.count_tasks:

        print(len(models) * len(pvt_corners) * len(skew_corners))
        exit(0)

    if args.task_id is not None:

        all_combos = [(m, p, s) for m in sorted(models) for p in sorted(pvt_corners) for s in sorted(skew_corners)]

        if args.task_id >= len(all_combos):

            print(f"Error: --task-id {args.task_id} out of range (0-{len(all_combos)-1})")

            exit(1)

        _m, _p, _s = all_combos[args.task_id]
        models       = [_m]
        pvt_corners  = [_p]
        skew_corners = [_s]

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
        dataset, total_sims = create_dataset()
        save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, f'{dataset_name}.json'))
        print()
        exit(0)

    if args.task_id is not None:
        run_id = args.task_id * NUM_SAMPLES

    elif args.clean:
        run_id = 0

    else:

        log_pattern = rf"run_{args.design}_(\\d+)\\.log"
        existing_logs = [f for f in os.listdir(OUT_DIR) if re.fullmatch(log_pattern, f)]

        if existing_logs:
            max_id = max(int(re.fullmatch(log_pattern, f).group(1)) for f in existing_logs)
            run_id = max_id + 1

        else:
            run_id = 0

    total_simulations = len(models) * len(pvt_corners) * len(skew_corners) * NUM_SAMPLES
    print(f"Starting {total_simulations} simulations...")

    if total_simulations > 500:
        print(f"0/{total_simulations} simulations complete", end='', flush=True)

    meta_data = []

    existing_params_set = set()

    if not args.clean:
        metadata_path = os.path.join(OUT_DIR, f"metadata_{args.design}.json")

        if os.path.exists(metadata_path):

            with open(metadata_path, "r") as f:

                try:

                    prev_meta = json.load(f)
                    for entry in prev_meta:

                        param_tuple = tuple(sorted(entry["params"].items()))
                        existing_params_set.add(param_tuple)

                except Exception:
                    pass

    start_time = time.time()

    session_run_idx = 0

    for model in models:

        for pvt_corner in pvt_corners:

            for skew_corner in skew_corners:

                model_name = os.path.basename(model)

                if model_name not in PROCESS_PARAM_RANGES:

                    print(f"Warning: No parameter ranges defined for {model_name}, using 45nm defaults")
                    model_name = "45nm_HP.pm"

                ranges = PROCESS_PARAM_RANGES[model_name]
                new_unique_params = set()
                max_attempts = NUM_SAMPLES * 100  # Avoid infinite loop
                attempts = 0

                while len(new_unique_params) < NUM_SAMPLES and attempts < max_attempts:

                    params = {k: np.random.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
                    param_tuple = tuple(sorted(params.items()))
                    attempts += 1

                    if param_tuple in existing_params_set or param_tuple in new_unique_params:
                        continue

                    new_unique_params.add(param_tuple)
                    existing_params_set.add(param_tuple)

                    netlist = write_netlist(
                        params, model, pvt_corner, skew_corner, run_id, DESIGN_NETLIST
                    )

                    out_file = os.path.join(
                        OUT_DIR,
                        f"run_{args.design}_{run_id}.log"
                    )

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

                    run_ngspice(netlist, out_file)

                    meta_data.append({
                        "run": run_id,
                        "design": args.design,
                        "model": model,
                        "pvt_corner": pvt_corner,
                        "skew_corner": skew_corner,
                        "params": params,
                        "output": out_file
                    })

                    try:
                        os.remove(netlist)
                    except Exception as e:
                        print(f"Warning: Could not delete netlist {netlist}: {e}")

                    run_id += 1

                if len(new_unique_params) < NUM_SAMPLES:
                    print(
                        f"Warning: Only {len(new_unique_params)} unique parameter sets "
                        f"generated for {model_name}, {pvt_corner}, {skew_corner} "
                        f"after {attempts} attempts."
                    )

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
        num_existing = len(existing_params_set) if not args.clean else 0
        num_new = len(meta_data)
        import os as _os_env
        _os_env.environ['NEW_SIM_COUNT'] = str(num_new)
        _os_env.environ['TOTAL_SIM_COUNT'] = str(num_existing + num_new)
        print("Parsing results and creating dataset...")
        dataset, total_sims = create_dataset()
        save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, f'{dataset_name}.json'))
        print()
        print()
