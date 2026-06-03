"""
Membership Inference Attack Script
Loads embeddings, logits, and attention weights from results/inference_outputs_<process>.npz
and attempts to determine if a sample was in the training set.
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

    # Simple baseline: random scores for membership
    np.random.seed(0)
    scores = np.random.rand(len(embeddings))
    np.savez(os.path.join(args.privacy_dir, f'membership_inference_{args.process}.npz'), scores=scores)
    print("[membership_inference] Baseline output saved.")

if __name__ == '__main__':
    main()
