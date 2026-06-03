"""
Edge Reconstruction Attack Script
Loads embeddings, logits, and attention weights from results/inference_outputs_<process>.npz
and attempts to reconstruct edge information from model outputs.
"""
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP_baseline)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    args = parser.parse_args()

    npz_path = os.path.join(args.privacy_dir, f'inference_outputs_{args.process}.npz')
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    logits = data['logits']
    attention_weights = data['attention_weights']

    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded logits shape: {logits.shape}")
    print(f"Loaded attention weights: {len(attention_weights)} layers")

    # Simple baseline: reconstruct edge presence from logits thresholding
    reconstructed = (logits > 0).astype(int)
    np.savez(os.path.join(args.privacy_dir, f'edge_reconstruction_{args.process}.npz'), reconstructed=reconstructed)
    print("[edge_reconstruction] Baseline output saved.")

if __name__ == '__main__':
    main()
