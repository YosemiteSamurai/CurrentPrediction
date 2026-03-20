# ------------------------------------------------------------------------------
# run_sims.py
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
# ------------------------------------------------------------------------------

# --- Ensure working directory is project root ---
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if os.getcwd() != project_root:
    os.chdir(project_root)


import subprocess
import os
import random
import json
import time
import numpy as np

# Ensure randomness is different every run
import datetime
seed = (int(time.time() * 1e6) ^ os.getpid() ^ int(datetime.datetime.now().microsecond)) % (2**32 - 1)
np.random.seed(seed)
random.seed(seed)

try:
    from scripts.parse_results import create_dataset, save_dataset, parse_spice_log
except ModuleNotFoundError:
    from parse_results import create_dataset, save_dataset, parse_spice_log

# Argument parsing (must be before any use of args)
import argparse
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
args = parser.parse_args()

# Accept a base dataset name; if an extension is supplied, strip it.
dataset_name = os.path.splitext(args.dataset)[0]

# Clear the screen before starting
os.system('cls' if os.name == 'nt' else 'clear')

# Config
# -------------------------------

NGSPICE = "ngspice"
BASE_SPICE = os.path.join("designs", "template.sp")
MODELS_DIR = "models"  # If models/ is to be moved, update here
CORNERS_DIR = "corners"  # If corners/ is to be moved, update here
OUT_DIR = "results"  # If results/ is to be moved, update here
MATRICES_DIR = "matrices"
DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
NUM_SAMPLES = 7 # per model/corner - more samples to hit extremes

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MATRICES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
# Clean results directory for fresh run
if args.clean == 'all':
    print("Cleaning all results for fresh run...")
    for file in os.listdir(OUT_DIR):
        file_path = os.path.join(OUT_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
elif args.clean == 'design':
    print(f"Cleaning results for design '{args.design}'...")
    import re
    design_log_pattern = re.compile(rf"run_{re.escape(args.design)}_(\d+)\.log$")
    design_metadata = f"metadata_{args.design}.json"
    for file in os.listdir(OUT_DIR):
        file_path = os.path.join(OUT_DIR, file)
        if not os.path.isfile(file_path):
            continue
        if design_log_pattern.fullmatch(file) or file == design_metadata:
            os.remove(file_path)

# Set the design netlist file
DESIGN_NETLIST = os.path.join("designs", args.design + '.sp')

# Print the included netlist being simulated
with open(BASE_SPICE) as f:
    for line in f:
        if line.strip().startswith('.include') and os.path.basename(DESIGN_NETLIST) in line:
            netlist_name = line.strip().split()[-1].replace('"', '')
            break
    else:
        netlist_name = os.path.basename(DESIGN_NETLIST)
print(f"Simulating {netlist_name} to create a dataset...\n")

# -------------------------------
# Process-Specific Parameter Ranges  
# -------------------------------

PROCESS_PARAM_RANGES = {
    # Advanced nodes - can handle smaller features
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
    # Older bulk processes - need larger minimums
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

# -------------------------------
# Helpers
# -------------------------------

def rand_param(lo, hi):
    return random.uniform(lo, hi)

def gen_params(model):
    """Generate random parameters for specific process model"""
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
    # Replace the .include line for the design
    import re
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
    # Always ensure .print tran I(vmeas_vdd) I(vmeas_in) I(vmeas_out) I(vmeas_target) is present
    import re
    # Remove any existing .print tran I(VDD) I(Vin) lines (case-insensitive, allow whitespace)
    filled = re.sub(r'^\s*\.print\s+tran\s+I\(VDD\)\s+I\(Vin\)\s*$', '', filled, flags=re.MULTILINE | re.IGNORECASE)
    # Remove any existing .print tran I(vmeas_vdd ...) lines to avoid duplicates
    filled = re.sub(r'^\s*\.print\s+tran\s+I\(vmeas_vdd\)[^\n]*$', '', filled, flags=re.MULTILINE | re.IGNORECASE)
    # Remove any trailing blank lines
    filled = re.sub(r'\n{2,}', '\n', filled)
    print_lines = [
        '.print tran V(vdd) V(in) V(target) V(out)',
        '.print tran I(vmeas_vdd) I(vmeas_in) I(vmeas_out) I(vmeas_target)',
        '.meas tran I_VDD_MAX MAX vdd#branch',
        '.meas tran I_Vin_MAX MAX vin#branch'
    ]
    # Insert before .end
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
    # Suppress NGSpice output
    result = subprocess.run(cmd, check=True, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
    # Parse the full log and replace it with a compact JSON containing only
    # what parse_results.py needs. The parser detects JSON and returns it directly.
    if not args.debug:
        results = parse_spice_log(out_file)
        with open(out_file, 'w') as f:
            json.dump(results, f)

# -------------------------------
# Main Loop
# -------------------------------

models = [
    os.path.join(MODELS_DIR, f)
    for f in sorted(os.listdir(MODELS_DIR))
    if f.endswith('.pm') and os.path.isfile(os.path.join(MODELS_DIR, f))
]

# PVT corners (voltage/temperature variations)
pvt_corners = [
    os.path.join(CORNERS_DIR, f)
    for f in os.listdir(CORNERS_DIR)
    if f.endswith(".sp") and not f.startswith("skew_")
]

# Process skew corners (NMOS/PMOS device variations)
skew_corners = [
    os.path.join(CORNERS_DIR, f)
    for f in os.listdir(CORNERS_DIR)
    if f.startswith("skew_") and f.endswith(".sp")
]



# Determine starting run_id based on existing log files if not cleaning
if args.clean:
    run_id = 0
else:
    import re
    # Match log files for the current design only, using fullmatch for exact match
    log_pattern = rf"run_{args.design}_(\d+)\.log"
    existing_logs = [f for f in os.listdir(OUT_DIR) if re.fullmatch(log_pattern, f)]
    if existing_logs:
        max_id = max(int(re.fullmatch(log_pattern, f).group(1)) for f in existing_logs)
        run_id = max_id + 1
    else:
        run_id = 0

# Calculate total number of simulations  
total_simulations = len(models) * len(pvt_corners) * len(skew_corners) * NUM_SAMPLES
print(f"Starting {total_simulations} simulations...")

if total_simulations > 500:
    print(f"0/{total_simulations} simulations complete", end='', flush=True)

meta_data = []

# If not cleaning, load previous metadata and parameter sets
existing_params_set = set()
if not args.clean:
    metadata_path = os.path.join(OUT_DIR, f"metadata_{args.design}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            try:
                prev_meta = json.load(f)
                for entry in prev_meta:
                    # Use a tuple of sorted (key, value) pairs for uniqueness
                    param_tuple = tuple(sorted(entry["params"].items()))
                    existing_params_set.add(param_tuple)
            except Exception:
                pass

# Start timer
start_time = time.time()

def stratified_params(ranges, n):
    """Generate n stratified samples for each parameter in ranges dict."""
    # This function is now unused, but kept for reference
    params = {}
    for k, (lo, hi) in ranges.items():
        bins = np.linspace(lo, hi, n+1)
        vals = [np.random.uniform(bins[i], bins[i+1]) for i in range(n)]
        np.random.shuffle(vals)
        params[k] = vals
    return [dict(zip(params.keys(), v)) for v in zip(*params.values())]




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
                # Only add if not in global or local set
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
                # Print status
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
                # Delete the custom .sp netlist after simulation
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

# End timer
end_time = time.time()
elapsed = end_time - start_time
avg_time = elapsed / total_simulations if total_simulations > 0 else 0

if total_simulations > 500:
    print()  # Print newline after progress overwrite

print(f"\nTotal simulation time: {elapsed:.2f} seconds")
print(f"Average time per simulation: {avg_time:.2f} seconds")



metadata_path = os.path.join(OUT_DIR, f"metadata_{args.design}.json")
if args.clean:
    # Overwrite with only new runs
    with open(metadata_path, "w") as f:
        json.dump(meta_data, f, indent=2)
else:
    # Append new runs to previous metadata
    prev_meta = []
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                prev_meta = json.load(f)
        except Exception:
            prev_meta = []
    # Only add new runs (avoid duplicates)
    prev_param_tuples = set(tuple(sorted(entry["params"].items())) for entry in prev_meta)
    new_entries = [entry for entry in meta_data if tuple(sorted(entry["params"].items())) not in prev_param_tuples]
    combined_meta = prev_meta + new_entries
    with open(metadata_path, "w") as f:
        json.dump(combined_meta, f, indent=2)
if total_simulations > 500:
    print()  # Print newline after progress overwrite

# Automatically parse results and create dataset
num_existing = len(existing_params_set) if not args.clean else 0
num_new = len(meta_data)
import os as _os_env
_os_env.environ['NEW_SIM_COUNT'] = str(num_new)
_os_env.environ['TOTAL_SIM_COUNT'] = str(num_existing + num_new)
print("Parsing results and creating dataset...")
dataset, total_sims = create_dataset()
save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, f'{dataset_name}.json'))

print()

# After dataset creation
def regenerate_matrices():
    subprocess.run(['python', os.path.join(SCRIPTS_DIR, 'generate_matrices.py')], check=True)
regenerate_matrices()

print()
