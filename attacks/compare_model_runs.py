"""
Compare multiple run-tagged models trained on the same dataset.

This script summarizes dataset load time, training time, and the presence of
attack outputs for each run tag.
"""

import argparse
import json
import os
import math
from pathlib import Path


def load_metadata(results_dir, process, tag):  # original: def load_metadata(privacy_dir, process, tag):
    tag_part = tag or 'baseline'
    metadata_path = results_dir / f"{process}_{tag_part}_runmetadata.json"  # original: metadata_path = results_dir / f"run_metadata_{process}{suffix}.json"
    if not metadata_path.exists():
        return None, metadata_path
    with open(metadata_path, "r") as handle:
        return json.load(handle), metadata_path


def file_status(results_dir, inputs_dir, process, tag):  # original: def file_status(privacy_dir, process, tag):
    suffix = f"_{tag}" if tag else ""
    tag_part = tag or 'baseline'
    names = {
        "model": results_dir / f"{process}_{tag_part}_model.pt",  # original: f"model_{process}{suffix}.pt"
        "predictions": results_dir / f"{process}_{tag_part}_predictions.csv",  # original: f"predictions_{process}{suffix}.csv"
        "inference_outputs": results_dir / f"{process}_{tag_part}_inference.npz",  # original: f"inference_outputs_{process}{suffix}.npz"
        "embedding_inversion": inputs_dir / f"embedding_inversion_{process}{suffix}.npz",  # original: "embedding_inversion": privacy_dir / f"embedding_inversion_{process}{suffix}.npz",
        "edge_reconstruction": inputs_dir / f"edge_reconstruction_{process}{suffix}.npz",  # original: "edge_reconstruction": privacy_dir / f"edge_reconstruction_{process}{suffix}.npz",
        "membership_inference": inputs_dir / f"membership_inference_{process}{suffix}.npz",  # original: "membership_inference": privacy_dir / f"membership_inference_{process}{suffix}.npz",
    }
    return {name: path.exists() for name, path in names.items()}


def format_seconds(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
        return f"{value:.2f}s"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", required=True, help="Process name, e.g. 22nm_LP")
    parser.add_argument("--tags", nargs=4, required=True, help="Exactly four run tags to compare")
    parser.add_argument("--privacy-dir", default=".", help="Directory containing attack artifacts (attacks/inputs)")  # original: parser.add_argument("--privacy-dir", default=".", help="Directory containing run artifacts")
    parser.add_argument("--results-dir", default=None, help="Directory containing general outputs (defaults to --privacy-dir)")
    args = parser.parse_args()

    inputs_dir = Path(args.privacy_dir)  # original: privacy_dir = Path(args.privacy_dir)
    results_dir = Path(args.results_dir) if args.results_dir else inputs_dir
    print(f"[compare_model_runs] process: {args.process}")
    print(f"[compare_model_runs] results_dir: {results_dir}")  # original: print(f"[compare_model_runs] privacy_dir: {privacy_dir}")
    print(f"[compare_model_runs] inputs_dir: {inputs_dir}")
    print("")

    for tag in args.tags:
        metadata, metadata_path = load_metadata(results_dir, args.process, tag)  # original: metadata, metadata_path = load_metadata(privacy_dir, args.process, tag)
        status = file_status(results_dir, inputs_dir, args.process, tag)  # original: status = file_status(privacy_dir, args.process, tag)

        print(f"=== {tag} ===")
        if metadata is None:
            print(f"metadata: missing ({metadata_path})")
        else:
            load_seconds = metadata.get("data_prep_seconds")
            if load_seconds is None or (isinstance(load_seconds, float) and math.isnan(load_seconds)):
                load_seconds = metadata.get("dataset_load_seconds")
            train_seconds = metadata.get("training_seconds")
            epochs = metadata.get("epochs")
            epoch_seconds = metadata.get("epoch_seconds", [])
            avg_epoch = sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None
            print(f"data prep/load: {format_seconds(load_seconds)}")
            print(f"training: {format_seconds(train_seconds)}")
            print(f"epochs: {epochs}")
            if avg_epoch is not None:
                print(f"avg epoch: {avg_epoch:.2f}s")

        for name, exists in status.items():
            print(f"{name}: {'OK' if exists else 'missing'}")
        print("")


if __name__ == "__main__":
    main()