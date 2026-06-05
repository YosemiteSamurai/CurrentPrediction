# =============================================================================
# predict.py -- Run inference on new data using a saved model checkpoint
#
# Loads a trained GAN from results/model.pt (v2.0 monolithic) or
# results/model_split.pt (v3.0 split-learning) and predicts branch
# currents for each row in a JSON dataset file.
#
# Checkpoint formats supported:
#   - v2.0:  {model_state_dict, config, label_log_*, scaler, ...}
#            Uses models.block_2inv with raw BSIM4 columns from the row.
#   - v3.0:  v2.0 keys plus {public_scaler, foundry_state_dict, ...}.
#            Reconstructs the foundry encoder, runs models.block_2inv_public,
#            and stitches the foundry's embedding into the M1 / M2 nodes.
#
# Deployment emulation:
#   --frozen-embedding path/to/process_embeddings.npz
#       Skips the encoder entirely and uses a precomputed embedding file
#       (the realistic deployed case where the foundry has published
#       z_pmos, z_nmos vectors per process and the design house has
#       received them but not the encoder weights).
#
# Usage:
#   python predict.py --input ../dataset/dataset_22nm_HP.json
#   python predict.py --input ../dataset/my_data.json \
#                     --checkpoint ../results/model_split.pt
#   python predict.py --input ../dataset/my_data.json \
#                     --checkpoint ../results/model_split.pt \
#                     --frozen-embedding ../results/process_22nm_HP.npz
# =============================================================================

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_python_dir = os.path.join(_project_root, 'python')

# Force split-local modules (dataset.py, models.py, foundry.py) to resolve
# before similarly named modules under python/.
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
if _python_dir not in sys.path:
    sys.path.insert(1, _python_dir)

import models
from gan import GAN
from graph import Graph
from dataset import (
    SKEW_CODES, OPTION_CODES,
    PMOS_BSIM_FIELDS, NMOS_BSIM_FIELDS,
    PMOS_BSIM_COLS, NMOS_BSIM_COLS,
)

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

    return gan, config, checkpoint


def load_foundry(checkpoint, device, checkpoint_path=None):
    """Reconstruct the Foundry from a v3.0 checkpoint.

    Tries the bundled `foundry_state_dict` key first; if absent, looks
    for a sibling `model_split_foundry.pt` next to `checkpoint_path`
    (the deployment-style layout sweep.py writes when split_learning is
    on). Returns None when neither source is available.
    """

    fsd = checkpoint.get("foundry_state_dict")

    if fsd is None and checkpoint_path is not None:

        sibling = os.path.join(os.path.dirname(checkpoint_path),
                               "model_split_foundry.pt")
        if os.path.exists(sibling):
            fsd = torch.load(sibling, map_location=device)

    if fsd is None:
        return None

    from foundry import Foundry
    return Foundry.from_state_dict(fsd, device=device)


def load_frozen_embedding(path):
    """Load a precomputed (z_pmos, z_nmos) embedding pair from a .npz file.

    Expected keys: 'z_pmos' (D,) and 'z_nmos' (D,). Single-process
    deployment scenario: foundry publishes one .npz per .pm release.
    """

    with np.load(path) as fp:
        z_p = fp['z_pmos']
        z_n = fp['z_nmos']
    return (torch.as_tensor(z_p, dtype=torch.float32),
            torch.as_tensor(z_n, dtype=torch.float32))


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

    values = np.array([[row.get(c, 0.0) for c in feature_columns]],
                      dtype=np.float32)
    scaled = scaler.transform(values)[0]

    return {c: float(scaled[i]) for i, c in enumerate(feature_columns)}


def _process_name_for_row(row: dict, fallback: str) -> str:
    """Mirror dataset.py's per-row process-name derivation."""

    size = row.get('Size')
    option = row.get('Option')
    if size is not None and isinstance(option, str):
        try:
            return f"{int(round(float(size)))}nm_{option}"
        except (TypeError, ValueError):
            pass
    return fallback


def predict_row_monolithic(gan, config, label_log_mean, label_log_std,
                           scaler, feature_columns, raw_row, device):
    """v2.0 inference: feed raw BSIM4 columns straight into block_2inv."""

    row = encode_categoricals(raw_row)
    design = row.get("Design", "2inv")
    encoder = getattr(models, config.model + "_" + design)

    for field in CURRENT_LABELS:
        row[field] = 0.0

    if scaler is not None:
        scaled = scale_row(row, scaler, feature_columns)
        row.update(scaled)

    model_row = pd.Series(row)
    edges, X = encoder(model_row, design)
    graph = Graph(edges, X, config)
    A = graph.A.to(device)
    X_t = graph.X.to(device)

    with torch.no_grad():
        z, attn_weights = gan.encode(X_t, A, return_attention_weights=True)
        out = gan.decode(z, A).view(-1)

    i_target_pred = 10 ** (float(out[I_TARGET_EDGE_INDEX]) * label_log_std
                           + label_log_mean)
    return i_target_pred, z.cpu().numpy(), out.cpu().numpy(), attn_weights


def predict_row_split(gan, config, label_log_mean, label_log_std,
                       public_scaler, public_feature_cols, raw_row, device,
                       foundry=None, frozen_embedding=None,
                       fallback_process="default"):
    """v3.0 inference: build X_public, then assemble the foundry embedding.

    Either `foundry` or `frozen_embedding` must be provided. Frozen
    embeddings emulate the deployed case where the design house only has
    the published (z_pmos, z_nmos) vectors.
    """

    row = encode_categoricals(raw_row)
    design = row.get("Design", "2inv")
    encoder_fn = getattr(models, config.model + "_" + design + "_public")

    for field in CURRENT_LABELS:
        row[field] = 0.0

    if public_scaler is not None and public_feature_cols:
        scaled = scale_row(row, public_scaler, public_feature_cols)
        row.update(scaled)

    model_row = pd.Series(row)
    edges, X_public = encoder_fn(
        model_row, design,
        embed_dim=getattr(config, 'embed_dim', 16))
    graph = Graph(edges, X_public, config)
    A = graph.A.to(device)
    X_t = graph.X.to(device)

    if frozen_embedding is not None:

        z_p_single, z_n_single = frozen_embedding
        z_p_send = z_p_single.to(device).unsqueeze(0)
        z_n_send = z_n_single.to(device).unsqueeze(0)

    else:

        if foundry is None:
            raise RuntimeError(
                "Split-learning checkpoint requires a Foundry or a "
                "--frozen-embedding file")
        proc = _process_name_for_row(raw_row, fallback_process)
        if proc not in foundry._process_table:
            raise KeyError(
                f"Foundry has no entry for process '{proc}'. Available: "
                f"{list(foundry._process_table.keys())}")
        z_p, z_n = foundry.encode_for_inference(proc)
        z_p_send = z_p.unsqueeze(0)
        z_n_send = z_n.unsqueeze(0)

    X_assembled = models.assemble_block_2inv(
        X_t, z_p_send, z_n_send,
        num_nodes_per_graph=getattr(config, 'nodes_per_graph', 6),
        pmos_offset=getattr(config, 'pmos_offset', 4),
    )

    with torch.no_grad():
        z, attn_weights = gan.encode(
            X_assembled, A, return_attention_weights=True)
        out = gan.decode(z, A).view(-1)

    i_target_pred = 10 ** (float(out[I_TARGET_EDGE_INDEX]) * label_log_std
                           + label_log_mean)
    return i_target_pred, z.cpu().numpy(), out.cpu().numpy(), attn_weights


def get_feature_columns(scaler, data_sample: dict) -> list:

    drop = {"Design", "ID", "Skew"}
    cols = [k for k in data_sample.keys() if k not in drop]
    return cols


def main():

    parser = argparse.ArgumentParser(
        description="Run GAN current prediction inference")
    parser.add_argument("--input", required=True,
                        help="Path to input JSON dataset")
    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV file path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--frozen-embedding", default=None,
                        help="(v3.0 only) Path to a precomputed .npz file "
                             "with 'z_pmos' and 'z_nmos' arrays. Skips the "
                             "encoder and emulates the deployed case where "
                             "the design house only has published "
                             "embeddings, not the encoder weights.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from {args.checkpoint} ...")

    gan, config, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"  model={config.model}, layers={config.layers}, "
          f"hidden={config.hidden_dim}, heads={config.heads}")
    print(f"  label_log_mean={checkpoint['label_log_mean']:.4f}, "
          f"label_log_std={checkpoint['label_log_std']:.4f}")

    is_split = bool(checkpoint.get("split_learning", False)) or \
               ("foundry_state_dict" in checkpoint)

    foundry = None
    frozen_embedding = None

    if is_split:

        if args.frozen_embedding is not None:

            frozen_embedding = load_frozen_embedding(args.frozen_embedding)
            print(f"  using frozen embedding from {args.frozen_embedding}")

        else:

            foundry = load_foundry(checkpoint, device,
                                   checkpoint_path=args.checkpoint)
            print(f"  loaded foundry: "
                  f"{len(foundry._process_table) if foundry else 0} process(es)")

    print(f"Loading data from {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  {len(data)} rows loaded")

    first_encoded = encode_categoricals(data[0])

    if is_split:

        public_scaler = checkpoint.get("public_scaler")
        public_feature_cols = checkpoint.get("public_feature_cols", [])
        if not public_feature_cols and public_scaler is not None:
            # Older checkpoints may not have stored the column list. Fall
            # back to the keys present on the encoded row, minus drops.
            public_feature_cols = get_feature_columns(public_scaler,
                                                      first_encoded)

    else:

        # v2.0 path uses the original global scaler. Older v3.0 bundled
        # checkpoints may also have it for back-compat; v3.0 strict
        # design-house checkpoints intentionally omit it.
        scaler = checkpoint.get("scaler")
        feature_columns = (get_feature_columns(scaler, first_encoded)
                           if scaler is not None else [])

    fallback_process = os.path.basename(args.input).removesuffix('.json')
    if fallback_process.endswith('_dataset'):
        fallback_process = fallback_process[:-len('_dataset')]
    elif fallback_process.startswith('dataset_'):
        fallback_process = fallback_process[len('dataset_'):]

    print("Running inference ...")
    rows_out = []
    embeddings_list = []
    logits_list = []
    attn_weights_list = []

    for raw_row in data:

        if is_split:

            i_target, embedding, logits, attn_weights = predict_row_split(
                gan, config,
                checkpoint['label_log_mean'], checkpoint['label_log_std'],
                public_scaler, public_feature_cols, raw_row, device,
                foundry=foundry, frozen_embedding=frozen_embedding,
                fallback_process=fallback_process,
            )

        else:

            i_target, embedding, logits, attn_weights = predict_row_monolithic(
                gan, config,
                checkpoint['label_log_mean'], checkpoint['label_log_std'],
                scaler, feature_columns, raw_row, device,
            )

        rows_out.append({

            "ID": raw_row.get("ID"),
            "Design": raw_row.get("Design"),
            "I_target_pred": i_target,

        })
        embeddings_list.append(embedding)
        logits_list.append(logits)
        attn_weights_list.append(attn_weights)

    df = pd.DataFrame(rows_out)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}  ({len(df)} rows)")

    process_name = os.path.splitext(os.path.basename(args.output))[0].replace(
        'predictions_', '')
    npz_path = os.path.join(
        os.path.dirname(args.output),
        f"inference_outputs_{process_name}.npz")
    np.savez(
        npz_path,
        embeddings=np.array(embeddings_list, dtype=object),
        logits=np.array(logits_list, dtype=object),
        attention_weights=np.array(attn_weights_list, dtype=object),
    )
    print(f"Embeddings, logits, and attention weights saved to {npz_path}")


if __name__ == "__main__":
    main()
