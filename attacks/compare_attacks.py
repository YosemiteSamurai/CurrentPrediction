"""
compare_attacks.py
Compare Embedding Inversion, Edge Reconstruction, and Membership Inference results
across multiple runs (e.g., baseline vs. defense) for a given process.
"""
import argparse
import os
import sys


def _ensure_python_with_numpy():
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    fallback_python = "/nfs/stak/users/jonesm25/.conda/envs/currentprediction/bin/python"
    current_python = os.path.realpath(sys.executable)
    fallback_real = os.path.realpath(fallback_python)

    if os.path.exists(fallback_python) and current_python != fallback_real:
        print(f"[compare_attacks] numpy not found in {sys.executable}; relaunching with {fallback_python}")
        os.execv(fallback_python, [fallback_python, __file__, *sys.argv[1:]])

    raise


_ensure_python_with_numpy()

import numpy as np
from sklearn.metrics import roc_auc_score

def load_npz(prefix, process, attack):
    path = f"{prefix}/{attack}_{process}.npz"
    return np.load(path, allow_pickle=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP)')
    parser.add_argument('--tags', nargs='+', required=True, help='Tags to compare (e.g., baseline dp sl both)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs')
    parser.add_argument('--plot', action='store_true', help='Show comparison plots')
    args = parser.parse_args()

    ei_mse = []
    ei_ci = []
    er_acc = []
    er_ci = []
    mi_auc = []
    mi_holdout_auc = []

    for suffix in args.tags:
        process = f"{args.process}_{suffix}"

        # Prefer tag-scoped label file; fall back to legacy untagged.
        gt_labels_path = os.path.join(args.privacy_dir, f'membership_labels_{process}.npz')
        if not os.path.exists(gt_labels_path):
            gt_labels_path = os.path.join(args.privacy_dir, f'membership_labels_{args.process}.npz')

        labels = np.load(gt_labels_path)['labels'] if os.path.exists(gt_labels_path) else None

        # Embedding Inversion — read MSE saved by embedding_inversion.py
        ei_data = load_npz(args.privacy_dir, process, 'embedding_inversion')
        if 'mse' in ei_data.files:
            mse = float(ei_data['mse'])
            if 'mse_ci_low' in ei_data.files and 'mse_ci_high' in ei_data.files:
                ei_ci.append((float(ei_data['mse_ci_low']), float(ei_data['mse_ci_high'])))
            else:
                ei_ci.append((np.nan, np.nan))
        else:
            print(f'  [WARN] embedding_inversion_{process}.npz has no "mse" key; re-run attacks.')
            mse = np.nan
            ei_ci.append((np.nan, np.nan))
        ei_mse.append(mse)
        # Edge Reconstruction — read accuracy saved by edge_reconstruction.py
        er_data = load_npz(args.privacy_dir, process, 'edge_reconstruction')
        if 'accuracy' in er_data.files:
            acc = float(er_data['accuracy'])
            if 'accuracy_ci_low' in er_data.files and 'accuracy_ci_high' in er_data.files:
                er_ci.append((float(er_data['accuracy_ci_low']), float(er_data['accuracy_ci_high'])))
            else:
                er_ci.append((np.nan, np.nan))
        else:
            print(f'  [WARN] edge_reconstruction_{process}.npz has no "accuracy" key; re-run attacks.')
            acc = np.nan
            er_ci.append((np.nan, np.nan))
        er_acc.append(acc)
        # Membership Inference
        mi = load_npz(args.privacy_dir, process, 'membership_inference')['scores']
        if labels is not None:
            if mi.shape[0] == labels.shape[0]:
                try:
                    auc = roc_auc_score(labels, mi)
                except Exception as exc:
                    print(f"  [WARN] membership inference AUC skipped for {suffix}: {exc}")
                    auc = np.nan
            else:
                print(f"  [WARN] membership inference shape mismatch for {suffix}: "
                      f"labels {labels.shape} vs scores {mi.shape}. Reporting NaN.")
                auc = np.nan
        else:
            auc = np.nan
        mi_data = load_npz(args.privacy_dir, process, 'membership_inference')
        if 'auc_holdout' in mi_data.files:
            mi_holdout_auc.append(float(mi_data['auc_holdout']))
        else:
            mi_holdout_auc.append(np.nan)
        mi_auc.append(auc)

    # Print metrics
    print("\n=== Embedding Inversion (MSE) ===")
    for s, v, ci in zip(args.tags, ei_mse, ei_ci):
        if np.isnan(ci[0]) or np.isnan(ci[1]):
            print(f"{s}: {v:.4f}")
        else:
            print(f"{s}: {v:.4f}  [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")
    print("\n=== Edge Reconstruction (Accuracy) ===")
    for s, v, ci in zip(args.tags, er_acc, er_ci):
        if np.isnan(ci[0]) or np.isnan(ci[1]):
            print(f"{s}: {v:.4f}")
        else:
            print(f"{s}: {v:.4f}  [95% CI: {ci[0]:.4f}, {ci[1]:.4f}]")
    print("\n=== Membership Inference (AUC) ===")
    for s, v, vh in zip(args.tags, mi_auc, mi_holdout_auc):
        if np.isnan(vh):
            print(f"{s}: {v:.4f}")
        else:
            print(f"{s}: {v:.4f}  (holdout: {vh:.4f})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for --plot. Run without --plot, "
                "or install matplotlib in the current environment."
            ) from exc

        x = np.arange(len(args.tags))
        width = 0.25
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].bar(x, ei_mse, width, tick_label=args.tags)
        ax[0].set_title('Embedding Inversion (MSE)')
        ax[0].set_ylabel('MSE')
        ax[1].bar(x, er_acc, width, tick_label=args.tags)
        ax[1].set_title('Edge Reconstruction (Accuracy)')
        ax[1].set_ylabel('Accuracy')
        ax[2].bar(x, mi_auc, width, tick_label=args.tags)
        ax[2].set_title('Membership Inference (AUC)')
        ax[2].set_ylabel('AUC')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
