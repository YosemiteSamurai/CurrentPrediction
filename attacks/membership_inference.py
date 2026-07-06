"""
Membership Inference Attack (loss-based threshold attack)
Training samples tend to have lower prediction loss than held-out samples.
Score = -MSE(logits, I_* ground truth); higher score = more likely member.
AUC > 0.5 indicates a privacy leak. DP/SL should push AUC toward 0.5.

Only loads logits (small) from inference_outputs.
"""
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _to_2d(array_like):
    """Convert regular/object arrays to a dense 2-D float array."""
    arr = np.asarray(array_like)
    if arr.dtype == object:
        rows = [np.asarray(x, dtype=np.float64).reshape(-1) for x in arr]
        return np.stack(rows, axis=0)
    arr = arr.astype(np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP_baseline)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with attack .npz artifacts')  # original: parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    parser.add_argument('--results_dir', default=None, help='Directory with general inference outputs (defaults to --privacy_dir)')
    args = parser.parse_args()

    d = args.privacy_dir
    r = args.results_dir or d
    proc = args.process

    inf = np.load(os.path.join(r, f'{proc}_inference.npz'), allow_pickle=True)  # original: inf = np.load(os.path.join(r, f'inference_outputs_{proc}.npz'), allow_pickle=True)
    logits = _to_2d(inf['logits'])  # (N, n_out)
    print(f'[membership_inference] logits shape: {logits.shape}')

    edges_data = np.load(os.path.join(d, f'ground_truth_edges_{proc}.npz'), allow_pickle=True)
    edges = _to_2d(edges_data['edges'])  # (N, n_I)
    print(f'[membership_inference] edges shape: {edges.shape}')

    ml = np.load(os.path.join(d, f'membership_labels_{proc}.npz'), allow_pickle=True)
    labels = ml['labels'].astype(int)

    n_cols = min(logits.shape[1], edges.shape[1])

    # Normalize each column by global std to make features scale-independent.
    std_logits = logits[:, :n_cols].std(axis=0)
    std_edges = edges[:, :n_cols].std(axis=0)
    # Avoid division by zero
    std_logits = np.where(std_logits > 0, std_logits, 1.0)
    std_edges = np.where(std_edges > 0, std_edges, 1.0)

    logits_n = logits[:, :n_cols] / std_logits
    edges_n = edges[:, :n_cols] / std_edges

    # Build richer per-sample attack features from prediction residuals.
    diff = logits_n - edges_n
    feat_mse = np.mean(diff ** 2, axis=1)
    feat_mae = np.mean(np.abs(diff), axis=1)
    feat_max = np.max(np.abs(diff), axis=1)
    feat_mean_pred = np.mean(logits_n, axis=1)
    feat_std_pred = np.std(logits_n, axis=1)
    attack_X = np.column_stack([feat_mse, feat_mae, feat_max, feat_mean_pred, feat_std_pred])

    # Train attack model on a held-out split and report AUC on its test fold.
    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )
    x_scaler = StandardScaler().fit(attack_X[tr_idx])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_scaler.transform(attack_X[tr_idx]), labels[tr_idx])

    # Scores for all points, so compare_attacks.py can compute one global AUC.
    scores = clf.predict_proba(x_scaler.transform(attack_X))[:, 1]
    auc_holdout = float(roc_auc_score(labels[te_idx], scores[te_idx]))
    auc_global = float(roc_auc_score(labels, scores))

    print(f'[membership_inference] holdout AUC: {auc_holdout:.6f}')
    print(f'[membership_inference] global  AUC: {auc_global:.6f}')

    np.savez(
        os.path.join(d, f'membership_inference_{proc}.npz'),
        scores=scores,
        auc_holdout=np.float64(auc_holdout),
        auc_global=np.float64(auc_global),
    )
    print('[membership_inference] done.')


if __name__ == '__main__':
    main()
