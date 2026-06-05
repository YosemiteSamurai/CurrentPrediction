"""
Edge Reconstruction Attack
Learns a multi-output linear decoder from logits -> I_* currents, then evaluates
binary edge-state reconstruction on held-out samples.

To avoid trivial near-100% scores from sign-thresholding at zero, each I_* column
is binarized using a threshold computed from the *training* distribution
(quantile, default median). This produces a more informative privacy signal.
"""
import numpy as np
import argparse
import os
from sklearn.linear_model import Ridge


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


def _bootstrap_accuracy(y_true, y_pred, trials=200, seed=42):
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if n == 0:
        return np.nan, np.nan
    vals = []
    for _ in range(trials):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(y_true[idx] == y_pred[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP_baseline)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    parser.add_argument('--quantile', type=float, default=0.5,
                        help='Train-set quantile used to binarize each I_* column (default: 0.5)')
    args = parser.parse_args()

    d = args.privacy_dir
    proc = args.process

    inf = np.load(os.path.join(d, f'inference_outputs_{proc}.npz'), allow_pickle=True)
    logits = _to_2d(inf['logits'])  # (N, n_logit)
    print(f'[edge_reconstruction] logits shape: {logits.shape}')

    edges_data = np.load(os.path.join(d, f'ground_truth_edges_{proc}.npz'), allow_pickle=True)
    edges = _to_2d(edges_data['edges'])  # (N, n_I)
    print(f'[edge_reconstruction] edges shape: {edges.shape}')

    ml = np.load(os.path.join(d, f'membership_labels_{proc}.npz'), allow_pickle=True)
    is_train = ml['labels'].astype(bool)

    # Train a multi-output linear decoder: logits -> I_* currents.
    # This is stronger and more realistic than fixed thresholding.
    reg = Ridge(alpha=1.0)
    reg.fit(logits[is_train], edges[is_train])
    pred_te = reg.predict(logits[~is_train])
    true_te = edges[~is_train]

    # Compute per-column thresholds from train targets, then evaluate column-wise
    # accuracy and macro-average so one dominant column cannot hide poor columns.
    q = float(args.quantile)
    q = min(max(q, 0.0), 1.0)
    thresholds = np.quantile(edges[is_train], q, axis=0)

    pred_bin_2d = (pred_te > thresholds.reshape(1, -1)).astype(int)
    true_bin_2d = (true_te > thresholds.reshape(1, -1)).astype(int)

    per_col_acc = np.mean(pred_bin_2d == true_bin_2d, axis=0)
    acc = float(np.mean(per_col_acc)) if per_col_acc.size else 0.0

    # CI over flattened binary decisions for a stable interval estimate.
    pred_bin = pred_bin_2d.reshape(-1)
    true_bin = true_bin_2d.reshape(-1)
    ci_lo, ci_hi = _bootstrap_accuracy(true_bin, pred_bin)
    print(f'[edge_reconstruction] quantile threshold q={q:.2f}')
    print(f'[edge_reconstruction] per-column accuracy: {np.array2string(per_col_acc, precision=4)}')
    print(f'[edge_reconstruction] macro test accuracy: {acc:.6f}')
    print(f'[edge_reconstruction] 95% bootstrap CI: [{ci_lo:.6f}, {ci_hi:.6f}]')

    np.savez(
        os.path.join(d, f'edge_reconstruction_{proc}.npz'),
        accuracy=np.float64(acc),
        accuracy_ci_low=np.float64(ci_lo),
        accuracy_ci_high=np.float64(ci_hi),
        per_col_accuracy=per_col_acc.astype(np.float64),
        thresholds=thresholds.astype(np.float64),
        quantile=np.float64(q),
    )
    print('[edge_reconstruction] done.')


if __name__ == '__main__':
    main()
