# =============================================================================
# parse_results.py -- Results Parser for SPICE Dataset
#
# Extracts PEAK SUPPLY CURRENT from simulation logs for GNN training.
# Target: Predict peak supply current from circuit parameters (no simulation needed).
#
# Main function:
#   - parse_spice_log(log_file): Parses a SPICE log file and extracts peak supply current and other metrics for GNN training.
#
# Usage:
#   Used during dataset generation to extract target values for supervised learning.
# =============================================================================

import os
import json
import re
import numpy as np
import pandas as pd
from extract_model_params import extract_model_params

def parse_spice_log(log_file):

    try:

        with open(log_file, 'r') as f:
            content = f.read()

        stripped = content.lstrip()

        if stripped.startswith('{'):
            return json.loads(stripped)

        results = {
            'Target_Current': None,  # THE target for GNN prediction (renamed)
            'simulation_success': False,
            'error_message': None,
            'Node_Max_Voltage': {},
            'Node_Peak_Current': {}
        }

        if 'run simulation(s) aborted' in content or 'Fatal error' in content:

            error_match = re.search(r'Fatal error: (.+)', content)
            results['error_message'] = error_match.group(1) if error_match else 'Simulation failed'

            return results

        lines = content.split('\n')
        node_voltages = {}
        node_currents = {}

        def trunc15(s):
            return s[:15]
        
        node_current_vectors = {

            'vdd': trunc15('vmeas_vdd#branch'),
            'in': trunc15('vmeas_in#branch'),
            'out': trunc15('vmeas_out#branch'),
            'target': 'vmeas_target#br',

        }

        for line in lines:

            if re.match(r'^[a-zA-Z0-9_.#()]+\s+[-+eE0-9.]+$', line.strip()):

                parts = line.split()

                if len(parts) == 2:
                    name, val = parts

                    try:

                        fval = float(val)

                        if name.startswith('vmeas_') and name.endswith('#branch'):
                            node_currents.setdefault(name, []).append(abs(fval))

                        elif name.startswith('v(') and name.endswith(')'):

                            node = name[2:-1]
                            node_voltages.setdefault(node, []).append(fval)

                        elif name in node_current_vectors.values():
                            node_currents.setdefault(name, []).append(abs(fval))

                        elif name in ['vdd', 'in', 'out', 'target']:
                            node_voltages.setdefault(name, []).append(fval)

                    except Exception:
                        pass

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if line.startswith('Index') and 'time' in line:

                header = line
                m = re.findall(r'(v\([^)]*\)|[a-zA-Z0-9_#]+)', header)

                if m:

                    vars = [v[:15] for v in m[2:]]  # Skip Index, time
                    i += 2

                    while i < len(lines):

                        data_line = lines[i].strip()

                        if (not data_line or data_line.startswith('-') or 'Index' in data_line or 'Total analysis' in data_line):
                            break

                        try:

                            if '\t' in data_line:
                                parts = data_line.split('\t')

                            else:
                                parts = re.split(r'\s+', data_line)

                            for idx, var in enumerate(vars):

                                if len(parts) > idx+2:
                                    val = float(parts[idx+2])

                                    if var.startswith('v('):
                                        node = var[2:-1]
                                        node_voltages.setdefault(node, []).append(val)

                                    elif var.endswith('#branc') or var.endswith('#branch'):
                                        node_currents.setdefault(var, []).append(abs(val))

                        except Exception:
                            pass

                        i += 1
            i += 1


        for node in ['vdd', 'in', 'out', 'target']:
            vlist = node_voltages.get(node, [])

            if vlist:
                results['Node_Max_Voltage'][node] = float(np.max(vlist))

        node_peak_currents = {}

        for node, vec in node_current_vectors.items():

            ilist = node_currents.get(vec, [])

            if not ilist:
                ilist = node_currents.get(vec + 'h', [])  # e.g., vmeas_vdd#branch

            if ilist:
                node_peak_currents[node] = float(np.max(ilist))

        if 'target' not in node_peak_currents:

            for k in node_currents:

                if k.startswith('vmeas_target#br') and node_currents[k]:

                    node_peak_currents['target'] = float(np.max(node_currents[k]))

                    break

        if node_peak_currents:
            node_peak_currents['gnd'] = -sum(node_peak_currents.get(n, 0.0) for n in ['vdd', 'in', 'out', 'target'])

        results['Node_Peak_NodeCurrent'] = node_peak_currents

        meas_pattern = re.compile(r'(I_VDD_MAX|I_Vin_MAX)\s*=\s*([-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+)', re.IGNORECASE)

        for line in lines:

            m = meas_pattern.search(line)

            if m:

                key = m.group(1).upper()
                val = float(m.group(2))

                if key == 'I_VDD_MAX':

                    results['Target_Current'] = val
                    results['Node_Peak_Current']['VDD'] = val
                    results['simulation_success'] = True

                elif key == 'I_VIN_MAX':
                    results['Node_Peak_Current']['VIN'] = val

        if not results['simulation_success']:

            found_any = False

            for node, ilist in node_currents.items():

                if ilist:

                    results['Node_Peak_Current'][node] = float(np.max(ilist))
                    found_any = True

            if found_any:
                results['simulation_success'] = True

            if 'VDD' in node_currents and node_currents['VDD']:
                results['Target_Current'] = float(np.max(node_currents['VDD']))

        return results
    
    except Exception as e:

        return {

            'target_current_peak': None,
            'target_current_avg': None,
            'target_current_rms': None,
            'simulation_success': False,
            'error_message': str(e)

        }

def create_dataset():

    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    metadata = []
    meta_files = glob.glob(os.path.join(results_dir, 'metadata_*.json'))

    if not meta_files:
        raise FileNotFoundError(f"No metadata_*.json files found in {results_dir}. Run the simulations first!")
    
    for meta_path in meta_files:

        with open(meta_path, 'r') as f:

            try:

                meta = json.load(f)

                if isinstance(meta, list):
                    metadata.extend(meta)

                else:
                    metadata.append(meta)

            except Exception as e:
                print(f"Warning: Could not load {meta_path}: {e}")
    
    dataset = []
    process_tag = ''
    
    print(f"Processing {len(metadata)} simulation results...")
    
    for i, entry in enumerate(metadata):

        model_params = extract_model_params(entry['model'])
        model_name = os.path.basename(entry['model']).replace('.pm', '')
        pvt_name = os.path.basename(entry['pvt_corner']).replace('.sp', '')
        skew_name = os.path.basename(entry['skew_corner']).replace('skew_', '').replace('.sp', '')

        if '_' in model_name:

            feature_size_str, option = model_name.split('_', 1)
            feature_size = feature_size_str.replace('nm', '')

        else:

            feature_size = model_name.replace('nm', '')
            option = ''

        if not process_tag:

            process_tag = f"{feature_size}{option}"

        design_name = entry.get('design')

        if not design_name:
            design_name = os.environ.get('SPICE_DESIGN_NAME')

        if not design_name:
            design_name = os.environ.get('DESIGN')

        if not design_name:
            design_name = '2inv'

        features = {

            'ID': entry['run'],
            'Design': design_name,
            'PVT': pvt_name,
            'Skew': skew_name,
            'WN1': entry['params']['WN1'],
            'WP1': entry['params']['WP1'],
            'WN2': entry['params']['WN2'],
            'WP2': entry['params']['WP2'],
            'L1': entry['params']['L1'],
            'L2': entry['params']['L2'],
            'VDD': entry['params']['VDD'],
            'Temp': entry['params']['TEMP'],

        }

        features.update(model_params)
        log_path = entry['output']

        if os.path.exists(log_path):

            print(f"Processing run {entry['run']}... ({i+1}/{len(metadata)})", end='\r', flush=True)
            results = parse_spice_log(log_path)

            if results['simulation_success']:

                node_voltage = results.get('Node_Max_Voltage', {})
                node_nodecurrent = results.get('Node_Peak_NodeCurrent', {})
                flat_entry = {**features}
                voltage_nodes = ['vdd', 'gnd', 'in', 'out', 'target']
                current_nodes = ['vdd', 'gnd', 'in', 'out', 'target']

                for node in voltage_nodes:

                    if node == 'gnd':
                        flat_entry['V_gnd'] = 0.0

                    else:
                        flat_entry[f'V_{node}'] = node_voltage.get(node, float('nan'))

                for node in current_nodes:

                    if node == 'gnd':
                        flat_entry['I_gnd'] = node_nodecurrent.get('gnd', float('nan'))

                    else:
                        flat_entry[f'I_{node}'] = node_nodecurrent.get(node, float('nan'))

                dataset.append(flat_entry)

        else:
            print(f"Processing run {entry['run']}... X (Log file not found: {log_path})")
            
    print(f"Processing complete! Processed {len(metadata)} runs.                    ")
            
    if dataset:

        df = pd.DataFrame(dataset)

        for col in df.columns:

            if df[col].apply(lambda x: isinstance(x, dict)).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

        nunique = df.nunique(dropna=False)
        always_keep = {'Design'}

        if dataset:

            sample = dataset[0]

            for k in sample.keys():

                if k.startswith('V_') or k.startswith('I_'):
                    always_keep.add(k)

        constant_cols = [col for col in nunique[nunique <= 1].index.tolist() if col not in always_keep]

        if constant_cols:

            print(f"\nRemoving constant columns from dataset: {constant_cols}")
            df = df.drop(columns=constant_cols)

        for col in always_keep:

            if col not in df.columns:

                vals = [d.get(col, None) for d in dataset]
                df.insert(1, col, vals)

        dataset = df.to_dict(orient='records')

    return dataset, len(metadata), process_tag

def save_dataset(dataset, metadata_count, output_file='dataset.json'):

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    if not dataset:
        print("\nNo successful simulation results found. Dataset is empty. Skipping CSV and summary statistics.")

        return

    df = pd.DataFrame(dataset)

    if 'Design' not in df.columns:

        design_val = None

        if len(dataset) > 0 and 'Design' in dataset[0]:
            design_val = dataset[0]['Design']

        if not design_val:
            design_val = os.environ.get('SPICE_DESIGN_NAME', '2inv')

        df.insert(1, 'Design', design_val)

    else:

        cols = list(df.columns)
        cols.remove('Design')
        cols.insert(1, 'Design')
        df = df[cols]

    csv_file = output_file.replace('.json', '.csv')
    df.to_csv(csv_file, index=False)
    successful_runs = len(dataset)
    print(f"\n--- Dataset Summary ---\n")
    new_sim_count = os.environ.get('NEW_SIM_COUNT')

    if new_sim_count is not None:
        print(f"New simulations run: {new_sim_count}")

    print(f"Total simulations: {metadata_count}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {metadata_count - successful_runs}")

    if successful_runs > 0:

        df = pd.DataFrame(dataset)

        if 'Target_Current' in df.columns:

            print(f"\nGNN Target: Target_Current (successful runs)")
            target_currents = df['Target_Current']
            print(f"Range: {target_currents.min():.2e} to {target_currents.max():.2e} A")
            print(f"Mean: {target_currents.mean():.2e} A")
            print(f"Std: {target_currents.std():.2e} A")

    print(f"\nDataset saved to: {output_file} and {csv_file}")

if __name__ == "__main__":
    
    RESULTS_DIR = "results"
    DATASET_DIR = "dataset"
    meta_path = os.path.join(RESULTS_DIR, 'metadata.json')

    if not os.path.exists(meta_path):
        print("Error: metadata.json not found. Run the simulations first!")

        exit(1)

    dataset, total_sims, process_tag = create_dataset()
    filename = f"{process_tag}_dataset.json" if process_tag else "dataset.json"
    save_dataset(dataset, total_sims, os.path.join(DATASET_DIR, filename))
