"""
compare_attacks.py
Compare Embedding Inversion, Edge Reconstruction, and Membership Inference results
across multiple runs (e.g., baseline vs. defense) for a given process.
"""
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

def load_npz(prefix, process, attack):
    path = f"{prefix}/{attack}_{process}.npz"
    return np.load(path, allow_pickle=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP)')
    parser.add_argument('--suffixes', nargs='+', required=True, help='Suffixes to compare (e.g., baseline defense1 defense2)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    args = parser.parse_args()

    # Infer ground-truth/originals file names from process and privacy_dir
    gt_emb_path = os.path.join(args.privacy_dir, f'original_embeddings_{args.process}.npz')
    gt_edges_path = os.path.join(args.privacy_dir, f'ground_truth_edges_{args.process}.npz')
    gt_labels_path = os.path.join(args.privacy_dir, f'membership_labels_{args.process}.npz')

    # Load ground-truth/originals
    orig_emb = np.load(gt_emb_path)['embeddings'] if os.path.exists(gt_emb_path) else None
    true_edges = np.load(gt_edges_path)['edges'] if os.path.exists(gt_edges_path) else None
    labels = np.load(gt_labels_path)['labels'] if os.path.exists(gt_labels_path) else None

    ei_mse = []
    er_acc = []
    mi_auc = []

    for suffix in args.suffixes:
        process = f"{args.process}_{suffix}"
        # Embedding Inversion
        ei = load_npz(args.privacy_dir, process, 'embedding_inversion')['reconstructed']
        if orig_emb is not None:
            mse = mean_squared_error(orig_emb.flatten(), ei.flatten())
        else:
            mse = np.nan
        ei_mse.append(mse)
        # Edge Reconstruction
        er = load_npz(args.privacy_dir, process, 'edge_reconstruction')['reconstructed']
        if true_edges is not None:
            acc = accuracy_score(true_edges.flatten(), er.flatten())
        else:
            acc = np.nan
        er_acc.append(acc)
        # Membership Inference
        mi = load_npz(args.privacy_dir, process, 'membership_inference')['scores']
        if labels is not None:
            auc = roc_auc_score(labels, mi)
        else:
            auc = np.nan
        mi_auc.append(auc)

    # Print metrics
    print("\n=== Embedding Inversion (MSE) ===")
    for s, v in zip(args.suffixes, ei_mse):
        print(f"{s}: {v:.4f}")
    print("\n=== Edge Reconstruction (Accuracy) ===")
    for s, v in zip(args.suffixes, er_acc):
        print(f"{s}: {v:.4f}")
    print("\n=== Membership Inference (AUC) ===")
    for s, v in zip(args.suffixes, mi_auc):
        print(f"{s}: {v:.4f}")

    # Visualization
    x = np.arange(len(args.suffixes))
    width = 0.25
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(x, ei_mse, width, tick_label=args.suffixes)
    ax[0].set_title('Embedding Inversion (MSE)')
    ax[0].set_ylabel('MSE')
    ax[1].bar(x, er_acc, width, tick_label=args.suffixes)
    ax[1].set_title('Edge Reconstruction (Accuracy)')
    ax[1].set_ylabel('Accuracy')
    ax[2].bar(x, mi_auc, width, tick_label=args.suffixes)
    ax[2].set_title('Membership Inference (AUC)')
    ax[2].set_ylabel('AUC')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
