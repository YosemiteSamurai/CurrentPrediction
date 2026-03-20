# generate_matrices.py
# This script generates adjacency and feature matrices for GNN training from a SPICE netlist and simulation dataset.
# ---
# Adjacency Matrix Node Mapping
# ---------------------------------------------
# This script automatically extracts all unique node names from the SPICE netlist.
# To control how nodes appear in the adjacency matrix, update the NODE_NAME_MAP dictionary below.
# Example: {'vdd': 'Vdd', '0': 'Gnd', 'in': 'Vi', 'out': 'Vo', 'target': 'It'}
# Any node not in NODE_NAME_MAP will use its original name.
# ---------------------------------------------

import numpy as np
import pandas as pd
import re


# File paths for input data
import os
DATASET_DIR = "dataset"
DESIGNS_DIR = "designs"
MATRICES_DIR = "matrices"
csv_path = os.path.join(DATASET_DIR, 'dataset.csv')         # Simulation results CSV
# Dynamically determine the netlist file to use
import glob
import sys
spice_path = None
# 1. Try environment variable
design_env = os.environ.get('DESIGN_NETLIST')
if design_env and os.path.isfile(design_env):
    spice_path = design_env
else:
    # 2. Try to infer from dataset.csv (if present)
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'design' in df.columns:
            design_name = df['design'].iloc[0]
            candidate = os.path.join(DESIGNS_DIR, f"{design_name}.sp")
            if os.path.isfile(candidate):
                spice_path = candidate
    except Exception:
        pass
    # 3. Fallback: use first .sp file in designs/
    if not spice_path:
        sp_files = glob.glob(os.path.join(DESIGNS_DIR, '*.sp'))
        if sp_files:
            spice_path = sp_files[0]
if not spice_path or not os.path.isfile(spice_path):
    sys.exit(f"ERROR: Could not find a valid SPICE netlist in {DESIGNS_DIR}. Please provide a netlist file.")


# Node name mapping: update this dictionary to control the names used in the adjacency matrix
# Example: {'vdd': 'Vdd', '0': 'Gnd', 'in': 'Vi', 'out': 'Vo', 'target': 'It'}
NODE_NAME_MAP = {
    'vdd': 'Vdd',
    '0': 'Gnd',
    'in': 'Vi',
    'out': 'Vo',
    'target': 'It',
    # Add or modify mappings as needed
}

def parse_spice_nodes_edges(spice_path):
    edges = []
    node_set = set()
    device_nets = {}
    with open(spice_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('*'):
                continue
            parts = line_stripped.split()
            dev_name = parts[0].upper() if parts else ''
            if dev_name in ['MP1', 'MN1', 'MP2', 'MN2'] and len(parts) >= 5:
                dev, d, g, s, b = parts[:5]
                device_nets[dev_name] = {'d': d, 'g': g, 's': s, 'b': b}
                # Instead of net-to-net edges, connect device to its nets
                # We'll add device as a vertex
            elif dev_name == 'VDD' and len(parts) >= 3:
                node_set.update([parts[1], parts[2]])
                edges.append((parts[1], parts[2]))
            elif dev_name == 'VIN' and len(parts) >= 3:
                node_set.update([parts[1], parts[2]])
                edges.append((parts[1], parts[2]))
            elif dev_name == 'CLOAD' and len(parts) >= 3:
                node_set.update([parts[1], parts[2]])
                edges.append((parts[1], parts[2]))
    # Build the set of all nodes (nets) and devices
    all_nets = set()
    for dev, nets in device_nets.items():
        all_nets.update(nets.values())
    all_nets.update(node_set)
    all_devices = list(device_nets.keys())
    all_vertices = sorted({NODE_NAME_MAP.get(n, n) for n in all_nets}) + all_devices
    # Edges: connect each device to its nets
    adj = np.zeros((len(all_vertices), len(all_vertices)), dtype=int)
    vertex_idx = {n: i for i, n in enumerate(all_vertices)}
    for dev, nets in device_nets.items():
        for net in nets.values():
            mapped_net = NODE_NAME_MAP.get(net, net)
            if dev in vertex_idx and mapped_net in vertex_idx:
                i, j = vertex_idx[dev], vertex_idx[mapped_net]
                adj[i, j] = 1
                adj[j, i] = 1
    # Add net-to-net edges for supplies, input, load cap
    for n1, n2 in edges:
        m1 = NODE_NAME_MAP.get(n1, n1)
        m2 = NODE_NAME_MAP.get(n2, n2)
        if m1 in vertex_idx and m2 in vertex_idx:
            i, j = vertex_idx[m1], vertex_idx[m2]
            adj[i, j] = 1
            adj[j, i] = 1
    return all_vertices, adj

# Parse the simulation CSV to build the feature matrix for each run
def parse_feature_matrix(csv_path):
    df = pd.read_csv(csv_path)
    # Remove run_id column if present
    if 'run_id' in df.columns:
        df = df.drop('run_id', axis=1)
    # Remove target output column (Target_Current) to keep only input features
    features = df.drop('Target_Current', axis=1, errors='ignore').values
    return features, df.columns.drop(['run_id', 'Target_Current'], errors='ignore')

if __name__ == '__main__':


    # Print which netlist file is being parsed

    # If the file is a wrapper (only .param/.include), follow the .include to the real netlist
    def find_real_netlist(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        # Look for .include lines that reference a .sp file in designs/
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith('.include'):
                parts = line_stripped.split()
                if len(parts) > 1:
                    inc_file = parts[1].strip('"')
                    # If it's a relative path and exists in designs/, use it
                    candidate = os.path.join(DESIGNS_DIR, os.path.basename(inc_file))
                    if os.path.isfile(candidate):
                        return candidate
        return path

    real_spice_path = find_real_netlist(spice_path)
    node_list, adj = parse_spice_nodes_edges(real_spice_path)
    adj_npy_path = os.path.join(MATRICES_DIR, 'adjacency_matrix.npy')
    adj_csv_path = os.path.join(MATRICES_DIR, 'adjacency_matrix.csv')
    np.save(adj_npy_path, adj)
    pd.DataFrame(adj, index=node_list, columns=node_list).to_csv(adj_csv_path)
    print(f'Adjacency matrix saved as {adj_npy_path} and {adj_csv_path}')

    # Generate feature matrix from simulation dataset
    features, feature_names = parse_feature_matrix(csv_path)
    feat_npy_path = os.path.join(MATRICES_DIR, 'feature_matrix.npy')
    feat_csv_path = os.path.join(MATRICES_DIR, 'feature_matrix.csv')
    np.save(feat_npy_path, features)
    pd.DataFrame(features, columns=feature_names).to_csv(feat_csv_path, index=False)
    print(f'Feature matrix saved as {feat_npy_path} and {feat_csv_path}')

    # Generate relevancy matrix (features x nodes) with robust node mapping
    nodes = node_list
    df = pd.read_csv(csv_path, nrows=1)
    feature_cols = [col for col in df.columns if col not in ['run_id', 'Target_Current']]

    # Use both device names and net names as columns for relevancy matrix
    device_names = ['MN1', 'MP1', 'MN2', 'MP2']
    # Get net names from device_nets (from adjacency matrix parse)
    # Re-parse the real netlist to get device_nets and all_nets
    device_nets = {}
    all_nets = set()
    with open(real_spice_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('*'):
                continue
            parts = line_stripped.split()
            dev_name = parts[0].upper() if parts else ''
            if dev_name in device_names and len(parts) >= 5:
                dev, d, g, s, b = parts[:5]
                device_nets[dev_name] = {'d': d, 'g': g, 's': s, 'b': b}
                all_nets.update([d, g, s, b])
    net_names = sorted({NODE_NAME_MAP.get(n, n) for n in all_nets})
    all_cols = device_names + net_names
    relevancy = np.zeros((len(feature_cols), len(all_cols)), dtype=int)
    def safe_index_any(name, names):
        if name in names:
            return names.index(name)
        else:
            print(f"Warning: '{name}' not found in list {names}. Skipping.")
            return None

    for i, feat in enumerate(feature_cols):
        assigned = False
        # WN1: MN1 only
        if feat == 'WN1':
            idx_dev = safe_index_any('MN1', all_cols)
            if idx_dev is not None:
                relevancy[i, idx_dev] = 1
            assigned = True
        # WN2: MN2 only
        elif feat == 'WN2':
            idx_dev = safe_index_any('MN2', all_cols)
            if idx_dev is not None:
                relevancy[i, idx_dev] = 1
            assigned = True
        # WP1: MP1 only
        elif feat == 'WP1':
            idx_dev = safe_index_any('MP1', all_cols)
            if idx_dev is not None:
                relevancy[i, idx_dev] = 1
            assigned = True
        # WP2: MP2 only
        elif feat == 'WP2':
            idx_dev = safe_index_any('MP2', all_cols)
            if idx_dev is not None:
                relevancy[i, idx_dev] = 1
            assigned = True
        # L1: MN1 and MP1 only
        elif feat == 'L1':
            for dev in ['MN1', 'MP1']:
                idx_dev = safe_index_any(dev, all_cols)
                if idx_dev is not None:
                    relevancy[i, idx_dev] = 1
            assigned = True
        # L2: MN2 and MP2 only
        elif feat == 'L2':
            for dev in ['MN2', 'MP2']:
                idx_dev = safe_index_any(dev, all_cols)
                if idx_dev is not None:
                    relevancy[i, idx_dev] = 1
            assigned = True
        # _N: MN1 and MN2 only
        elif feat.endswith('_N'):
            for dev in ['MN1', 'MN2']:
                idx_dev = safe_index_any(dev, all_cols)
                if idx_dev is not None:
                    relevancy[i, idx_dev] = 1
            assigned = True
        # _P: MP1 and MP2 only
        elif feat.endswith('_P'):
            for dev in ['MP1', 'MP2']:
                idx_dev = safe_index_any(dev, all_cols)
                if idx_dev is not None:
                    relevancy[i, idx_dev] = 1
            assigned = True
        # Temp: relevant to all devices and all nets
        elif feat.lower().startswith('temp'):
            for name in all_cols:
                idx = safe_index_any(name, all_cols)
                if idx is not None:
                    relevancy[i, idx] = 1
            assigned = True
        # VDD: relevant to all devices and all nets
        elif feat == 'VDD':
            for name in all_cols:
                idx = safe_index_any(name, all_cols)
                if idx is not None:
                    relevancy[i, idx] = 1
            assigned = True
        # Process, other global features: relevant to all devices and all nets
        if not assigned:
            for name in all_cols:
                idx = safe_index_any(name, all_cols)
                if idx is not None:
                    relevancy[i, idx] = 1
    # Save relevancy matrix with device and net names as columns
    pd.DataFrame(relevancy, index=feature_cols, columns=all_cols).to_csv(os.path.join(MATRICES_DIR, 'relevancy_matrix.csv'))
    print('Relevancy matrix saved as', os.path.join(MATRICES_DIR, 'relevancy_matrix.csv'))


