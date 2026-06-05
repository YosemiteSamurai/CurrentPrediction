"""
Validate and prepare privacy attack artifacts for a given dataset/process.

This script checks for the files needed by the privacy attacks and can
rebuild the ID-based ground-truth files from the source dataset when they are
missing.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def resolve_dataset_path(dataset_value, repo_root):
    dataset_dir = repo_root / "dataset"
    candidate = Path(dataset_value)

    if candidate.exists():
        return candidate

    if candidate.suffix:
        for base in (dataset_dir, repo_root):
            resolved = base / candidate.name
            if resolved.exists():
                return resolved
        return candidate

    for extension in (".json", ".csv"):
        resolved = dataset_dir / f"{dataset_value}{extension}"
        if resolved.exists():
            return resolved

    if dataset_value.startswith("dataset_"):
        for extension in (".json", ".csv"):
            resolved = dataset_dir / f"{dataset_value}{extension}"
            if resolved.exists():
                return resolved

    return dataset_dir / dataset_value


def load_dataframe(dataset_path):
    if dataset_path.suffix.lower() == ".json":
        with open(dataset_path, "r") as file_handle:
            data = json.load(file_handle)
        return pd.DataFrame(data)
    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def split_train_test(frame, test_size=0.3, seed=42):
    total_rows = len(frame)
    if total_rows == 0:
        return frame, frame

    indices = list(range(total_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_rows = int(round(total_rows * test_size))
    test_rows = max(1, min(total_rows - 1, test_rows)) if total_rows > 1 else 0

    test_idx = indices[:test_rows]
    train_idx = indices[test_rows:]

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    return train_frame, test_frame


def dataset_process_name(dataset_path):
    stem = dataset_path.stem
    if stem.startswith("dataset_"):
        return stem[len("dataset_") :]
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset JSON/CSV file or dataset name")
    parser.add_argument("--privacy-dir", default=None, help="Directory containing privacy artifacts")
    parser.add_argument("--tag", default="", help="Optional run tag suffix used in artifact names")
    parser.add_argument("--test-size", type=float, default=0.3, help="Train/test split fraction used by sweep.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by sweep.py")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    privacy_dir = Path(args.privacy_dir) if args.privacy_dir else script_dir
    privacy_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset, repo_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    process = dataset_process_name(dataset_path)
    tag_suffix = f"_{args.tag.strip()}" if args.tag.strip() else ""
    process_with_tag = f"{process}{tag_suffix}"
    print(f"[validate_privacy_artifacts] dataset: {dataset_path}")
    print(f"[validate_privacy_artifacts] process: {process}")
    print(f"[validate_privacy_artifacts] tag: {args.tag.strip() or '<none>'}")
    print(f"[validate_privacy_artifacts] privacy_dir: {privacy_dir}")

    frame = load_dataframe(dataset_path)
    if "ID" not in frame.columns:
        raise KeyError("Dataset does not contain an 'ID' column")

    train_ids_path = privacy_dir / f"train_ids_{process}{tag_suffix}.npy"  # original: train_ids_path = privacy_dir / "train_ids.npy"
    legacy_train_ids_path = privacy_dir / "train_ids.npy"
    if train_ids_path.exists():
        train_ids = np.load(train_ids_path, allow_pickle=True)
        print(f"[validate_privacy_artifacts] found {train_ids_path.name} ({train_ids.shape[0]} IDs)")  # original: print(f"[validate_privacy_artifacts] found train_ids.npy ({train_ids.shape[0]} IDs)")
    elif legacy_train_ids_path.exists():
        train_ids = np.load(legacy_train_ids_path, allow_pickle=True)
        print(f"[validate_privacy_artifacts] found legacy {legacy_train_ids_path.name} ({train_ids.shape[0]} IDs)")
    else:
        train_frame, _ = split_train_test(frame, test_size=args.test_size, seed=args.seed)
        train_ids = train_frame["ID"].to_numpy()
        np.save(train_ids_path, train_ids)
        print(f"[validate_privacy_artifacts] wrote {train_ids_path.name} ({train_ids.shape[0]} IDs)")  # original: print(f"[validate_privacy_artifacts] wrote train_ids.npy ({train_ids.shape[0]} IDs)")

    membership_labels_path = privacy_dir / f"membership_labels_{process_with_tag}.npz"
    legacy_membership_labels_path = privacy_dir / f"membership_labels_{process}.npz"
    if not membership_labels_path.exists():
        labels = np.isin(frame["ID"].to_numpy(), train_ids).astype(int)
        np.savez(membership_labels_path, labels=labels)
        print(f"[validate_privacy_artifacts] wrote {membership_labels_path.name}")
        if legacy_membership_labels_path.exists():
            print(f"[validate_privacy_artifacts] NOTE: legacy {legacy_membership_labels_path.name} also exists")
    else:
        print(f"[validate_privacy_artifacts] found {membership_labels_path.name}")

    embeddings_path = privacy_dir / f"original_embeddings_{process_with_tag}.npz"
    if not embeddings_path.exists():
        drop = {"ID", "Design", "Skew", "PVT", "Option"}
        feature_cols = [
            c for c in frame.columns
            if c not in drop and is_numeric_dtype(frame[c])
        ]
        embeddings = frame[feature_cols].to_numpy()
        np.savez(embeddings_path, embeddings=embeddings)
        print(f"[validate_privacy_artifacts] wrote {embeddings_path.name} ({embeddings.shape})")
    else:
        print(f"[validate_privacy_artifacts] found {embeddings_path.name}")

    edges_path = privacy_dir / f"ground_truth_edges_{process_with_tag}.npz"
    if not edges_path.exists():
        edge_cols = [c for c in frame.columns if c.startswith("I_")]
        edges = frame[edge_cols].to_numpy()
        np.savez(edges_path, edges=edges)
        print(f"[validate_privacy_artifacts] wrote {edges_path.name} ({edges.shape})")
    else:
        print(f"[validate_privacy_artifacts] found {edges_path.name}")

    required_files = {
        f"original_embeddings_{process_with_tag}.npz": "embedding inversion ground truth",
        f"ground_truth_edges_{process_with_tag}.npz": "edge reconstruction ground truth",
        f"membership_labels_{process_with_tag}.npz": "membership inference labels",
        f"inference_outputs_{process_with_tag}.npz": "model inference outputs",
    }
    optional_files = {
        f"predictions_{process_with_tag}.csv": "privacy prediction CSV",
        f"model_{process_with_tag}.pt": "trained checkpoint",
    }

    missing_required = []
    for file_name, description in required_files.items():
        file_path = privacy_dir / file_name
        if file_path.exists():
            print(f"[validate_privacy_artifacts] OK: {file_name} ({description})")
        else:
            print(f"[validate_privacy_artifacts] MISSING: {file_name} ({description})")
            missing_required.append(file_name)

    for file_name, description in optional_files.items():
        file_path = privacy_dir / file_name
        if file_path.exists():
            print(f"[validate_privacy_artifacts] OK: {file_name} ({description})")
        else:
            print(f"[validate_privacy_artifacts] NOTE: {file_name} not found ({description})")

    if missing_required:
        raise SystemExit(1)

    print("[validate_privacy_artifacts] all required privacy artifacts are present")


if __name__ == "__main__":
    main()