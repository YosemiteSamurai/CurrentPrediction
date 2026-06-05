"""
Embedding Inversion Attack
Trains a linear decoder (Ridge regression) from model logits to input features.
Measures how well an attacker can recover process parameters from model outputs.
Higher MSE = harder to invert = better privacy.

Only loads logits (small) from inference_outputs — skips the large object-array embeddings.
"""
import numpy as np
import argparse
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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


def _bootstrap_mse(y_true, y_pred, trials=200, seed=42):
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if n == 0:
        return np.nan, np.nan
    vals = []
    for _ in range(trials):
        idx = rng.integers(0, n, size=n)
        vals.append(mean_squared_error(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP_baseline)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    args = parser.parse_args()

    d = args.privacy_dir
    proc = args.process

    # Load logits only (embeddings are large object arrays — skip them)
    inf = np.load(os.path.join(d, f'inference_outputs_{proc}.npz'), allow_pickle=True)
    logits = _to_2d(inf['logits'])  # (N, n_out)
    print(f'[embedding_inversion] logits shape: {logits.shape}')

    # Input features (numeric dataset columns — ground truth to recover)
    feat = np.load(os.path.join(d, f'original_embeddings_{proc}.npz'), allow_pickle=True)
    features = _to_2d(feat['embeddings'])  # (N, n_feat)
    print(f'[embedding_inversion] features shape: {features.shape}')

    # Membership labels (True = training sample)
    ml = np.load(os.path.join(d, f'membership_labels_{proc}.npz'), allow_pickle=True)
    is_train = ml['labels'].astype(bool)

    X_tr, X_te = logits[is_train], logits[~is_train]
    y_tr, y_te = features[is_train], features[~is_train]
    print(f'[embedding_inversion] train={X_tr.shape[0]}  test={X_te.shape[0]}')

    # Normalize independently (fit on train only to avoid data leakage)
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr)

    # Train linear decoder: logits -> input features
    reg = Ridge(alpha=1.0).fit(sx.transform(X_tr), sy.transform(y_tr))
    y_te_n = sy.transform(y_te)
    y_hat_n = reg.predict(sx.transform(X_te))
    mse = float(mean_squared_error(y_te_n, y_hat_n))
    ci_lo, ci_hi = _bootstrap_mse(y_te_n, y_hat_n)
    print(f'[embedding_inversion] test MSE (normalized): {mse:.6f}')
    print(f'[embedding_inversion] 95% bootstrap CI: [{ci_lo:.6f}, {ci_hi:.6f}]')

    np.savez(
        os.path.join(d, f'embedding_inversion_{proc}.npz'),
        mse=np.float64(mse),
        mse_ci_low=np.float64(ci_lo),
        mse_ci_high=np.float64(ci_hi),
    )
    print('[embedding_inversion] done.')


if __name__ == '__main__':
    main()
