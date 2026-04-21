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
# circuit_dataset: PyTorch Dataset that converts each data row into a
#   (edges, X) graph tensor pair by dispatching to the appropriate model
#   encoder in models.py.
#
# Custom loss functions:
#   - NMAELoss: Normalized MAE weighted by inverse of each label's mean
#   - LogL1Loss: L1 loss computed in log-space
#   - MAPELoss / LogMAPELoss: Mean absolute percentage error (active default)
#
# Training loop:
#   - train: Single epoch; encodes -> decodes -> backpropagates per batch
#   - test: Evaluates MAPE, mean relative error, and min/max relative error
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from graph import Graph, batch_graph

print("[dataset] imports complete", flush=True)

_dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

_process_datasets = sorted(

    os.path.join(_dataset_dir, f) for f in os.listdir(_dataset_dir)
    if f.endswith("_dataset.json")

)

if _process_datasets:

    DATA_FILE = _process_datasets[0]
    print(f"[dataset] auto-selected dataset: {os.path.basename(DATA_FILE)}", flush=True)

else:

    DATA_FILE = os.path.join(_dataset_dir, "dataset.json")
    print(f"[dataset] falling back to dataset.json", flush=True)

print("[dataset] loading JSON...", flush=True)

with open(DATA_FILE, "r") as f:
    data = json.load(f)

print(f"[dataset] loaded {len(data)} rows", flush=True)

data_frame = pd.DataFrame(data)
data_frame = data_frame.drop(columns=["ID", "PVT"])
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

data_frame['Option'] = data_frame['Option'].apply(lambda x: OPTION_CODES[x])
label_df = data_frame.loc[:, "I_vdd":"I_target":1]
print("[dataset] normalizing labels...", flush=True)

label_values = np.maximum(np.abs(label_df.values), 1e-30)
all_log_labels = np.log10(label_values.flatten())
label_log_mean = float(all_log_labels.mean())
label_log_std = float(all_log_labels.std())

log_label_df = pd.DataFrame(
    (np.log10(label_values) - label_log_mean) / label_log_std,
    columns=label_df.columns,
    index=label_df.index
)

scaler = StandardScaler()  # saved in checkpoint for inference feature scaling
scaled_data = scaler.fit_transform(data_frame)
data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)
data_frame['Design'] = design_col
data_frame.loc[:, "I_vdd":"I_target":1] = log_label_df

print("===================Data loaded successfully===============")
print("head")
print(data_frame.head())
print("infomation")
print(data_frame.info())
print("size of DF")
print(data_frame.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[dataset] device: {device}", flush=True)

class circuit_dataset(Dataset):

    def __init__(self, data_frame, config):

        self.df = data_frame.reset_index(drop=True)
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        design = row['Design']
        model = getattr(models, self.config.model + '_' + design)
        return model(row, design)

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

def train(gcn, optimizer, trainloader, config):

    total_loss = 0
    batches = 0

    for batch in trainloader:

        graph = batch_graph(batch, config)

        A = graph.A.to(device)
        y = graph.y.to(device)
        X = graph.X.to(device)

        optimizer.zero_grad()
        z = gcn.encode(X, A)
        out = gcn.decode(z, A).view(-1)
        n_edges = y.shape[0]
        mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        mask[config.target_edge_idx::config.edges_per_graph] = True
        loss = criterion(out[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    return gcn, optimizer, total_loss / batches

def test(gcn, testloader, config):

    total_loss = 0
    batches = 0
    predictions = 0
    error = 0
    max_error = 0
    min_error = float('inf')

    with torch.no_grad():

        for batch in testloader:

            graph = batch_graph(batch, config)

            A = graph.A.to(device)
            y = graph.y.to(device)
            X = graph.X.to(device)

            z = gcn.encode(X, A)
            out = gcn.decode(z, A).view(-1)
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
