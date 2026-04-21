# =============================================================================
# predict.py -- Run inference on new data using a saved model checkpoint
#
# Loads a trained GCN from results/model.pt and predicts branch currents
# for each row in a JSON dataset file. The checkpoint bundles the
# StandardScaler (fit during training) so features are scaled identically to
# what the model saw during training.
#
# Usage:
#   python predict.py --input ../dataset/dataset2.json
#   python predict.py --input ../dataset/my_new_data.json --output predictions.csv
#   python predict.py --input ../dataset/dataset2.json --checkpoint ../results/model.pt
# =============================================================================

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
import models
from gan import GAN
from graph import Graph
from dataset import SKEW_CODES, OPTION_CODES

CURRENT_LABELS = ["I_vdd", "I_gnd", "I_in", "I_out", "I_target"]
I_TARGET_EDGE_INDEX = 3
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "model.pt")

def load_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = SimpleNamespace(**checkpoint["config"])

    gan = GAN(


        checkpoint["embedding_dim"],
        config.hidden_dim,
        checkpoint["embedding_dim"],
        config.layers,
        heads=config.heads,

    )

    gan.load_state_dict(checkpoint["model_state_dict"])
    gan.to(device)
    gan.eval()

    return gan, config, checkpoint["label_log_mean"], checkpoint["label_log_std"], checkpoint["scaler"]

def encode_categoricals(row: dict) -> dict:

    row = dict(row)
    skew = row.pop("Skew", None)

    if skew is not None:
        row["SkewL"] = SKEW_CODES[skew[0]]
        row["SkewR"] = SKEW_CODES[skew[1]]

    if "Option" in row and isinstance(row["Option"], str):
        row["Option"] = OPTION_CODES[row["Option"]]

    return row

def scale_row(row: dict, scaler, feature_columns: list) -> dict:

    values = np.array([[row.get(c, 0.0) for c in feature_columns]], dtype=np.float32)
    scaled = scaler.transform(values)[0]

    return {c: float(scaled[i]) for i, c in enumerate(feature_columns)}

def predict_row(gan, config, label_log_mean, label_log_std, scaler, feature_columns, raw_row, device):

    row = encode_categoricals(raw_row)
    design = row.get("Design", "2inv")
    encoder = getattr(models, config.model + "_" + design)

    for field in CURRENT_LABELS:
        row[field] = 0.0

    if scaler is not None:
        scaled = scale_row(row, scaler, feature_columns)
        row.update(scaled)

    edges, X = encoder(row, design)
    graph = Graph(edges, X, config)
    A = graph.A.to(device)
    X_t = graph.X.to(device)

    with torch.no_grad():
        z = gan.encode(X_t, A)
        out = gan.decode(z, A).view(-1)

    i_target_pred = 10 ** (float(out[I_TARGET_EDGE_INDEX]) * label_log_std + label_log_mean)
    return i_target_pred

def get_feature_columns(scaler, data_sample: dict) -> list:

    drop = {"Design", "ID", "Skew"}  # dropped before scaler.fit in dataset.py
    cols = [k for k in data_sample.keys() if k not in drop]
    return cols

def main():

    parser = argparse.ArgumentParser(description="Run GCN current prediction inference")
    parser.add_argument("--input", required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to model checkpoint (.pt)")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from {args.checkpoint} ...")
    gcn, config, label_log_mean, label_log_std, scaler = load_checkpoint(args.checkpoint, device)
    print(f"  model={config.model}, layers={config.layers}, hidden={config.hidden_dim}, heads={config.heads}")
    print(f"  label_log_mean={label_log_mean:.4f}, label_log_std={label_log_std:.4f}")
    print(f"Loading data from {args.input} ...")

    with open(args.input) as f:
        data = json.load(f)

    print(f"  {len(data)} rows loaded")
    first_encoded = encode_categoricals(data[0])
    feature_columns = get_feature_columns(scaler, first_encoded)
    print("Running inference ...")
    rows_out = []

    for raw_row in data:

        i_target = predict_row(gcn, config, label_log_mean, label_log_std,
                               scaler, feature_columns, raw_row, device)
        
        rows_out.append({

            "ID": raw_row.get("ID"),
            "Design": raw_row.get("Design"),
            "I_target_pred": i_target,

        })

    df = pd.DataFrame(rows_out)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
