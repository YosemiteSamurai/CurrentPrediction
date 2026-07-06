"""
extract_ground_truths.py
Extracts ground-truth arrays for embedding inversion, edge reconstruction, and membership inference
from the original dataset file. Outputs .npz files in attacks/inputs/ for use in privacy attack evaluation.
"""
import numpy as np
import json
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset JSON or CSV file')
    parser.add_argument('--privacy_dir', default='.', help='Output directory for .npz files')
    parser.add_argument('--train_ids', default=None, help='Optional: Path to .npy or .npz file with training IDs for membership inference')
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith('.json'):
        with open(args.dataset, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif args.dataset.endswith('.csv'):
        df = pd.read_csv(args.dataset)
    else:
        raise ValueError('Unsupported dataset format')

    # --- Embedding Inversion ground-truth ---
    # Use all numeric columns except ID, Design, Skew, PVT, Option
    drop = {'ID', 'Design', 'Skew', 'PVT', 'Option'}
    feature_cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    embeddings = df[feature_cols].to_numpy()
    np.savez(os.path.join(args.privacy_dir, 'original_embeddings.npz'), embeddings=embeddings)
    print(f"[extract_ground_truths] Saved original_embeddings.npz with shape {embeddings.shape}")

    # --- Edge Reconstruction ground-truth ---
    # Use edge label columns (e.g., I_vdd, I_in, I_out, I_gnd, I_target)
    edge_cols = [c for c in df.columns if c.startswith('I_')]
    edges = df[edge_cols].to_numpy()
    np.savez(os.path.join(args.privacy_dir, 'ground_truth_edges.npz'), edges=edges)
    print(f"[extract_ground_truths] Saved ground_truth_edges.npz with shape {edges.shape}")

    # --- Membership Inference ground-truth ---
    # If train_ids is provided, label 1 for member, 0 for non-member
    if args.train_ids:
        if args.train_ids.endswith('.npz'):
            train_ids = np.load(args.train_ids)['ids']
        else:
            train_ids = np.load(args.train_ids)
        all_ids = df['ID'].to_numpy()
        labels = np.isin(all_ids, train_ids).astype(int)
        np.savez(os.path.join(args.privacy_dir, 'membership_labels.npz'), labels=labels)
        print(f"[extract_ground_truths] Saved membership_labels.npz with shape {labels.shape}")
    else:
        print("[extract_ground_truths] Skipped membership_labels.npz (no train_ids provided)")

if __name__ == '__main__':
    main()
