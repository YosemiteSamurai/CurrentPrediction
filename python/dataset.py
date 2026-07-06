# =============================================================================
# dataset.py -- Dataset, Loss Functions, and Training Loop
#
# Contains everything needed to load data and train the GAN (GATv2), separated from
# the sweep/entry-point configuration in sweep.py.
#
# Data loading & preprocessing:
#   - Loads a JSON dataset from the local dataset/ directory
#   - Encodes categoricals: Skew (S/T/F -> 0/1/2), Option (LP/bulk/HP -> 0/1/2)
#   - StandardScaler-normalizes input features; labels are log10 + standardized
#
# v3.0 split-learning artifacts (computed unconditionally; consumers opt in
# via config.split_learning):
#   - PMOS_BSIM_COLS / NMOS_BSIM_COLS / LABEL_COLS / PUBLIC_FEATURE_COLS /
#     PRIVATE_FEATURE_COLS  -- column-set constants for the cut layer.
#   - process_table         -- {process_name: (scaled_pmos, scaled_nmos)}.
#                              Source of truth: the foundry's .pm files
#                              under ../models/, falling back to per-row
#                              BSIM4 columns when present.
#   - private_pmos_scaler /
#     private_nmos_scaler   -- handed to Foundry; never saved in the
#                              design-house checkpoint.
#   - public_scaler         -- StandardScaler fit on the public columns
#                              only; saved in the design-house checkpoint.
#   - data_frame_public     -- design-house dataframe with BSIM4 columns
#                              dropped. Carries a `_process` column so the
#                              dataset class can hand a process_name to the
#                              foundry at forward time.
#
# circuit_dataset: PyTorch Dataset that converts each data row into a
#   (edges, X) graph tensor pair by dispatching to the appropriate model
#   encoder in models.py. When config.split_learning is True the dispatch
#   target switches to <model>_public and the return value gains a
#   process_name string.
#
# Custom loss functions:
#   - NMAELoss: Normalized MAE weighted by inverse of each label's mean
#   - LogL1Loss: L1 loss computed in log-space
#   - MAPELoss / LogMAPELoss: Mean absolute percentage error (active default)
#
# Training loop:
#   - train: Single epoch; encodes -> decodes -> backpropagates per batch.
#            Optional `foundry` argument enables the split-learning cut.
#   - test:  Evaluates MAPE, mean relative error, and min/max relative
#            error. Same optional foundry argument.
# =============================================================================

import json
import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from graph import Graph, batch_graph

# v3.0 split-learning imports (additive; monolithic path does not use these).
import re
from sklearn.preprocessing import StandardScaler
try:
    from extract_model_params import extract_model_params
except ImportError:
    extract_model_params = None


class NumpyStandardScaler:
    """Small sklearn-like scaler used for training/inference compatibility."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Avoid divide-by-zero for constant columns.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# Allow override of dataset path via environment variable (set by sweep.py or pipeline)
DATASET_PATH = os.environ.get('DATASET_PATH', '').strip()  # original: DATASET_PATH = os.environ.get('DATASET_PATH', '').strip()
DATASET_NAME = os.environ.get('DATASET', '').strip()
_dataset_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")  # original: _dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
_models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

_dataset_load_started = time.perf_counter()

if DATASET_PATH and os.path.exists(DATASET_PATH):
    DATA_FILE = DATASET_PATH
elif DATASET_NAME:
    candidate = DATASET_NAME
    if not os.path.isabs(candidate):
        candidate = os.path.join(_dataset_dir, candidate)
    if not candidate.endswith('.json'):
        candidate = candidate + '.json'
    if os.path.exists(candidate):
        DATA_FILE = candidate
    else:
        raise FileNotFoundError(f"[dataset] DATASET points to missing file: {candidate}")
else:
    _process_datasets = sorted(
        os.path.join(_dataset_dir, f) for f in os.listdir(_dataset_dir)
        if f.startswith("dataset_") and f.endswith(".json")
    )
    if _process_datasets:
        DATA_FILE = _process_datasets[0]
    else:
        DATA_FILE = os.path.join(_dataset_dir, "dataset.json")


with open(DATA_FILE, "r") as f:
    data = json.load(f)

data_frame = pd.DataFrame(data)
row_ids = data_frame['ID'].reset_index(drop=True) if 'ID' in data_frame.columns else None
_num_ids = int(row_ids.nunique()) if row_ids is not None else 0
print(f"[dataset] read {len(data_frame)} rows ({_num_ids} unique IDs) from {os.path.basename(DATA_FILE)}", flush=True)
data_frame = data_frame.drop(columns=["ID", "PVT"])  # original: data_frame = data_frame.drop(columns=["ID", "PVT"])
design_col = data_frame['Design']
data_frame = data_frame.drop(columns=["Design"])

SKEW_CODES = {'S': 0,
              'T': 1,
              'F': 2}

data_frame['SkewL'] = data_frame['Skew'].apply(lambda x: SKEW_CODES[x[0]])
data_frame['SkewR'] = data_frame['Skew'].apply(lambda x: SKEW_CODES[x[1]])
data_frame = data_frame.drop(columns=["Skew"])

OPTION_CODES = {'LP': 0,
                'bulk': 1,
                'HP': 2}

# `Size` and `Option` are dropped by parse_results.py when constant across a
# single-process dataset, so guard the encode step rather than assuming
# they exist. We capture the original strings before encoding so the
# split-learning code can derive a stable per-row process_name.
_raw_option_strings = None

if 'Option' in data_frame.columns:

    _raw_option_strings = data_frame['Option'].astype(str).values
    data_frame['Option'] = data_frame['Option'].apply(
        lambda x: OPTION_CODES.get(x, 0))

if 'Size' in data_frame.columns:

    _raw_size_values = data_frame['Size'].astype(float).values

else:

    _raw_size_values = None

label_df = data_frame.loc[:, "I_vdd":"I_target":1]

label_values = np.maximum(np.abs(label_df.values), 1e-30)
all_log_labels = np.log10(label_values.flatten())
label_log_mean = float(all_log_labels.mean())
label_log_std = float(all_log_labels.std())

log_label_df = pd.DataFrame(
    (np.log10(label_values) - label_log_mean) / label_log_std,
    columns=label_df.columns,
    index=label_df.index
)

# =============================================================================
# v3.0 split-learning artifacts -- computed before the global scaler runs so
# raw BSIM4 values stay accessible when both Size/Option and the BSIM4
# columns themselves were already dropped from the dataset on disk.
# =============================================================================

LABEL_COLS = ["I_vdd", "I_gnd", "I_in", "I_out", "I_target"]

# These match the field lists in models.block_2inv (pFields/nFields). Kept
# as constants here so foundry.py and predict.py can rebuild the encoder's
# input vector in the right order.
PMOS_BSIM_FIELDS = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
                    'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
                    'CGDO']
NMOS_BSIM_FIELDS = PMOS_BSIM_FIELDS + ['PCLM', 'K2', 'DVT2']
PMOS_BSIM_COLS = [f + '_P' for f in PMOS_BSIM_FIELDS]
NMOS_BSIM_COLS = [f + '_N' for f in NMOS_BSIM_FIELDS]

PRIVATE_FEATURE_COLS = [c for c in data_frame.columns
                        if (c.endswith('_P') or c.endswith('_N'))
                        and c not in LABEL_COLS]
PUBLIC_FEATURE_COLS = [c for c in data_frame.columns
                       if c not in PRIVATE_FEATURE_COLS
                       and c not in LABEL_COLS]


def _process_name_from_filename(path):
    """Derive a process tag from a dataset filename.

    Accepts e.g. '22HP_dataset.json' -> '22HP', or
    'dataset_22nm_HP.json' -> '22nm_HP'.
    """
    base = os.path.basename(path)
    if base.endswith('.json'):
        base = base[:-len('.json')]
    if base.endswith('_dataset'):
        return base[:-len('_dataset')]
    if base.startswith('dataset_'):
        return base[len('dataset_'):]
    return base or 'default'


# Per-row process_name. For a multi-process dataset use the (Size, Option)
# tuple; for the common single-process case fall back to the dataset
# filename.
if _raw_option_strings is not None and _raw_size_values is not None:

    PROCESS_NAMES = np.array(
        [f"{int(round(s))}nm_{o}"
         for s, o in zip(_raw_size_values, _raw_option_strings)],
        dtype=object,
    )

else:

    _filename_proc = _process_name_from_filename(DATA_FILE)
    PROCESS_NAMES = np.array([_filename_proc] * len(data_frame), dtype=object)


def _find_pm_file(process_name):
    """Look up the .pm file in ../models/ corresponding to a process_name."""

    candidates = [os.path.join(_models_dir, f"{process_name}.pm")]

    # Tolerate variations like '22HP' <-> '22nm_HP'.
    m = re.match(r'^(\d+)(?:nm)?_?(.*)$', process_name)
    if m:

        size, suffix = m.group(1), m.group(2)
        suffix = suffix.lstrip('_')
        if suffix:
            candidates.append(os.path.join(_models_dir, f"{size}nm_{suffix}.pm"))
            candidates.append(os.path.join(_models_dir, f"{size}_{suffix}.pm"))
            candidates.append(os.path.join(_models_dir, f"{size}{suffix}.pm"))

    for c in candidates:
        if os.path.exists(c):
            return c

    return None


def _extract_raw_bsim_for_process(process_name):
    """Return (pmos_vec, nmos_vec) of unscaled BSIM4 floats for a process.

    Foundry-authoritative source (the .pm file itself) is preferred; falls
    back to the dataframe row when those columns survived parse_results.py.
    """

    if extract_model_params is not None:

        pm_path = _find_pm_file(process_name)
        if pm_path is not None:

            params = extract_model_params(pm_path)
            pmos_vec = np.array(
                [_safe_float(params.get(c)) for c in PMOS_BSIM_COLS],
                dtype=np.float32,
            )
            nmos_vec = np.array(
                [_safe_float(params.get(c)) for c in NMOS_BSIM_COLS],
                dtype=np.float32,
            )
            return pmos_vec, nmos_vec

    # Fall back to per-row dataframe lookup -- only useful for multi-process
    # datasets where parse_results.py left the BSIM4 columns in.
    if all(c in data_frame.columns for c in PMOS_BSIM_COLS + NMOS_BSIM_COLS):

        first_idx = int(np.where(PROCESS_NAMES == process_name)[0][0])
        pmos_vec = data_frame[PMOS_BSIM_COLS].iloc[first_idx].astype(np.float32).values
        nmos_vec = data_frame[NMOS_BSIM_COLS].iloc[first_idx].astype(np.float32).values
        return pmos_vec, nmos_vec

    # Last resort: zero vector. Encoder gets a constant input per process,
    # which still trains via the GAT gradient signal.
    return (np.zeros(len(PMOS_BSIM_COLS), dtype=np.float32),
            np.zeros(len(NMOS_BSIM_COLS), dtype=np.float32))


def _safe_float(v):

    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# Build the foundry's process table from raw (unscaled) BSIM4 values.
_raw_process_lookup = {}

for _proc in np.unique(PROCESS_NAMES):
    _raw_process_lookup[_proc] = _extract_raw_bsim_for_process(_proc)

# Fit the private scalers on the per-row stack of raw vectors. This way the
# scaler reflects the empirical distribution actually trained on, even when
# multi-process data is unbalanced. Single-process datasets get
# zero-variance columns; we preserve those columns at their raw values
# rather than emitting NaNs.
_all_raw_pmos = np.stack([_raw_process_lookup[p][0] for p in PROCESS_NAMES],
                         axis=0)
_all_raw_nmos = np.stack([_raw_process_lookup[p][1] for p in PROCESS_NAMES],
                         axis=0)

private_pmos_scaler = StandardScaler()
private_nmos_scaler = StandardScaler()
private_pmos_scaler.fit(_all_raw_pmos)
private_nmos_scaler.fit(_all_raw_nmos)

# `process_table`: design-house side hands this to the Foundry constructor.
# Values are scaled with the private scalers; the foundry's ProcessEncoder
# expects standardised inputs.
process_table = {}

for _proc, (_raw_p, _raw_n) in _raw_process_lookup.items():

    _scaled_p = private_pmos_scaler.transform(_raw_p.reshape(1, -1))[0]
    _scaled_n = private_nmos_scaler.transform(_raw_n.reshape(1, -1))[0]
    _scaled_p = np.where(np.isnan(_scaled_p), _raw_p, _scaled_p)
    _scaled_n = np.where(np.isnan(_scaled_n), _raw_n, _scaled_n)
    process_table[_proc] = (_scaled_p.astype(np.float32),
                            _scaled_n.astype(np.float32))

# Public scaler fits only on the columns that survive the cut. Saved in
# the design-house checkpoint and used by predict.py at inference.
public_scaler = StandardScaler()

if PUBLIC_FEATURE_COLS:
    public_scaler.fit(data_frame[PUBLIC_FEATURE_COLS].astype(np.float32).values)

# =============================================================================
# v2.0 global scaler: the original single global feature scaler running on
# the full dataframe. v2.0 checkpoints continue to load against this.
# =============================================================================

scaler = NumpyStandardScaler()  # saved in checkpoint for inference feature scaling
scaled_data = scaler.fit_transform(data_frame)
data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
data_frame['Design'] = design_col
data_frame.loc[:, "I_vdd":"I_target":1] = log_label_df

# =============================================================================
# Build the v3.0 design-house dataframe by dropping all BSIM4 columns.
# `_process` is added as a (string) column so circuit_dataset can hand a
# process_name to the foundry at forward time.
# =============================================================================

data_frame_public = data_frame.drop(columns=PRIVATE_FEATURE_COLS).copy()
data_frame_public['_process'] = PROCESS_NAMES
data_frame['_process'] = PROCESS_NAMES

# Sanity check: the design-house dataframe must contain no BSIM4 columns.
_leaked = [c for c in data_frame_public.columns
           if c.endswith('_P') or c.endswith('_N')]
assert not _leaked, (
    f"[dataset] BSIM4 leak: columns {_leaked} survived in data_frame_public"
)

DATASET_LOAD_SECONDS = time.perf_counter() - _dataset_load_started
print(f"[dataset] data load + preprocessing time: {DATASET_LOAD_SECONDS:.2f}s", flush=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[dataset] device: {device}", flush=True)

class circuit_dataset(Dataset):

    def __init__(self, data_frame, config):

        self.df = data_frame.reset_index(drop=True)
        self.config = config
        self.split_learning = bool(getattr(config, 'split_learning', False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        design = row['Design']

        if self.split_learning:

            model_fn = getattr(models, self.config.model + '_' + design + '_public')
            edges, X_public = model_fn(row, design)
            process_name = row.get('_process', 'default')
            return edges, X_public, process_name

        model_fn = getattr(models, self.config.model + '_' + design)
        return model_fn(row, design)


def batch_graph_split(batch, config, foundry):
    """Split-learning batch builder.

    Returns (merged_graph, z_p_send, z_n_send, process_names) where
    merged_graph.X has zero-padded BSIM4 slots; the caller is expected to
    invoke models.assemble_block_2inv(...) before passing to the GAT.
    """

    # PyTorch's default_collate stacks the per-sample tensors into batched
    # tensors and turns the per-sample process_name strings into a list.
    edges_batch = batch[0]
    X_public_batch = batch[1]
    process_names = list(batch[2]) if len(batch) > 2 else []

    graphs = []
    n = len(edges_batch) if isinstance(edges_batch, list) else edges_batch.shape[0]

    for i in range(n):

        edges_i = edges_batch[i] if isinstance(edges_batch, list) else edges_batch[i]
        X_i = X_public_batch[i] if isinstance(X_public_batch, list) else X_public_batch[i]
        graphs.append(Graph(edges_i, X_i, config))

    merged = graphs[0]
    for g in graphs[1:]:
        merged.merge(g)

    z_p_send, z_n_send = foundry.encode_batch(process_names)
    return merged, z_p_send, z_n_send, process_names

class NMAELoss(nn.L1Loss):

    def __init__(self, reduction: str = "mean"):

        super().__init__()
        label_mean = label_df.mean().values
        self.norm = [1/label_mean[0]]*2 + [1/label_mean[1]]*2 + [1/label_mean[2]]*2 + [1/label_mean[3]]*2 + [1/label_mean[4]]*4

    def forward(self, y_pred, y_true):

        norm = torch.tensor(self.norm * int(y_pred.shape[0] / len(self.norm))).to(device)
        return F.l1_loss(torch.mul(y_pred, norm), torch.mul(y_true, norm))

class LogL1Loss(nn.L1Loss):

    def __init__(self, reduction: str = "mean"):
        super().__init__()

    def forward(self, y_pred, y_true):
        return F.l1_loss(torch.log10(y_pred), torch.log10(y_true))

def MAPELoss(y_pred, y_true):

    epsilon = 1e-8  # To avoid division by zero

    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))

def LogMAPELoss(y_pred, y_true):    
    return MAPELoss(torch.log(y_pred), torch.log(y_true))

criterion = nn.L1Loss()

def train(gan, optimizer, trainloader, config, foundry=None):

    total_loss = 0
    batches = 0

    for batch in trainloader:

        if foundry is not None:

            graph, z_p_send, z_n_send, _ = batch_graph_split(batch, config, foundry)

        else:

            graph = batch_graph(batch, config)
            z_p_send = z_n_send = None

        A = graph.A.to(device)
        y = graph.y.to(device)
        X = graph.X.to(device)

        if foundry is not None:

            X = models.assemble_block_2inv(
                X, z_p_send, z_n_send,
                num_nodes_per_graph=getattr(config, 'nodes_per_graph', 6),
                pmos_offset=getattr(config, 'pmos_offset', 4),
            )

        optimizer.zero_grad()
        z = gan.encode(X, A)
        out = gan.decode(z, A).view(-1)
        n_edges = y.shape[0]
        mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        mask[config.target_edge_idx::config.edges_per_graph] = True
        loss = criterion(out[mask], y[mask])
        loss.backward()

        if getattr(config, 'dp_enabled', False):
            dp_c = float(getattr(config, 'dp_max_grad_norm', 1.0))
            dp_sigma = float(getattr(config, 'dp_noise_multiplier', 0.0))
            torch.nn.utils.clip_grad_norm_(gan.parameters(), dp_c)
            if dp_sigma > 0:
                for p in gan.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * (dp_sigma * dp_c))

        optimizer.step()

        if foundry is not None:
            foundry.backward(z_p_send, z_n_send)

        total_loss += loss.item()
        batches += 1

    return gan, optimizer, total_loss / batches

def test(gan, testloader, config, foundry=None):

    total_loss = 0
    batches = 0
    predictions = 0
    error = 0
    max_error = 0
    min_error = float('inf')

    with torch.no_grad():

        for batch in testloader:

            if foundry is not None:

                graph, z_p_send, z_n_send, _ = batch_graph_split(batch, config, foundry)

            else:

                graph = batch_graph(batch, config)
                z_p_send = z_n_send = None

            A = graph.A.to(device)
            y = graph.y.to(device)
            X = graph.X.to(device)

            if foundry is not None:

                X = models.assemble_block_2inv(
                    X, z_p_send, z_n_send,
                    num_nodes_per_graph=getattr(config, 'nodes_per_graph', 6),
                    pmos_offset=getattr(config, 'pmos_offset', 4),
                )

            z = gan.encode(X, A)
            out = gan.decode(z, A).view(-1)
            n_edges = y.shape[0]
            mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
            mask[config.target_edge_idx::config.edges_per_graph] = True
            loss = criterion(out[mask], y[mask])
            total_loss += loss.item()
            batches += 1
            y_pred_phys = 10 ** (out[mask].cpu() * label_log_std + label_log_mean)
            y_true_phys = 10 ** (y[mask].cpu() * label_log_std + label_log_mean)
            error_tensor = torch.abs((y_pred_phys - y_true_phys) / (y_true_phys + 1e-30))

            if error_tensor.max() > max_error:
                max_error = error_tensor.max()

            if error_tensor.min() < min_error:
                min_error = error_tensor.min()

            error += error_tensor.sum()
            predictions += len(y_pred_phys)

    return total_loss / batches, error / predictions, max_error, min_error
