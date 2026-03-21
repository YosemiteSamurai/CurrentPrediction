# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import requests
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from graph import Graph, batch_graph
from gcn import GCN

wandb.login()

DATA_URL = "https://web.engr.oregonstate.edu/~jonesm25/datasets/dataset2.json"

#load data from link
response = requests.get(DATA_URL)
data = response.json()

data_frame = pd.DataFrame(data)
data_frame = data_frame.drop(columns=["ID", "PVT"])

design_col = data_frame['Design']
data_frame = data_frame.drop(columns=["Design"])

# Process variable encodings
# S = slow, T = mid, F = fast
SKEW_CODES = {'S':0,
              'T':1,
              'F':2}

data_frame['SkewL'] = data_frame['Skew'].apply(lambda x: SKEW_CODES[x[0]])
data_frame['SkewR'] = data_frame['Skew'].apply(lambda x: SKEW_CODES[x[1]])
data_frame = data_frame.drop(columns=["Skew"])

OPTION_CODES = {'LP': 0,
              'bulk': 1,
                'HP': 2}
data_frame['Option'] = data_frame['Option'].apply(lambda x: OPTION_CODES[x])

label_df = data_frame.loc[:, "I_vdd":"I_target":1]
# label_std = torch.from_numpy(label_df.std().values)
label_mean = label_df.mean().values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_frame)
data_frame = pd.DataFrame(scaled_data, columns=data_frame.columns)

data_frame['Design'] = design_col
# Un-standardize labels for now, but could be useful to work more on this
data_frame.loc[:, "I_vdd":"I_target":1] = label_df

print("===================Data loaded successfully===============")
print("head")
print(data_frame.head())

print("infomation")
print(data_frame.info())
print("size of DF")
print(data_frame.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class circuit_dataset(Dataset):
  def __init__(self, data_frame, config):
    """
    data_frame is the full pandas data frame
    We store it and build graphs inside __getitem__
    """
    self.df = data_frame.reset_index(drop=True)
    self.config = config

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    """
    For each index:
    - take one row
    - convert it into graph tensors
    - return (adj_M, X, Y)
    """

    row = self.df.iloc[idx]

    design = row['Design']
    model = getattr(models, self.config.model + '_' + design)

    return model(row, design)


class NMAELoss(nn.L1Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        # TODO this is hard coded for a specific model
        self.norm = [1/label_mean[0]]*2 + [1/label_mean[1]]*2 + [1/label_mean[2]]*2 + [1/label_mean[3]]*2 + [1/label_mean[4]]*4

    def forward(self, y_pred, y_true):
        # Ensure that the input tensors have the same shape
        # assert y_pred.shape == y_true.shape, "Shape of y_pred and y_true must be the same"

        norm = torch.tensor(self.norm * int(y_pred.shape[0] / len(self.norm))).to(device)

        return F.l1_loss(torch.mul(y_pred, norm), torch.mul(y_true, norm))

class LogL1Loss(nn.L1Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Ensure that the input tensors have the same shape
        # assert y_pred.shape == y_true.shape, "Shape of y_pred and y_true must be the same"
        return F.l1_loss(torch.log10(y_pred), torch.log10(y_true))

def MAPELoss(y_pred, y_true):
    epsilon = 1e-8  # To avoid division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))

def LogMAPELoss(y_pred, y_true):
    return MAPELoss(torch.log(y_pred), torch.log(y_true))

# TODO move criterion to sweep function
# criterion = nn.MSELoss()
# criterion = NMAELoss()
# criterion = LogL1Loss()
criterion = MAPELoss

# Single training epoch
def train(gcn, optimizer, trainloader, config):
    for batch in trainloader:
        graph = batch_graph(batch, config)
        # TODO - do we need to update the graph parameters?
        # Having it in its own object might discard changes that we want to be permanent
        # Need to think on this a bit
        A = graph.A.to(device)
        y = graph.y.to(device)
        X = graph.X.to(device)

        optimizer.zero_grad()
        z = gcn.encode(X, A)
    
        out = gcn.decode(z, A).view(-1)
        loss = criterion(out, y)
        # print(loss)
        # print(y)
        # print(out)
        # print(z)
        # print(X)
        # print(A)
        # quit()
        loss.backward()
        optimizer.step()

    return gcn, optimizer, loss.item()

def test(gcn, testloader, config):
    total_loss = 0
    batches = 0
    predictions = 0
    error = 0
    max_error = 0
    min_error = 0

    # Evaluate the network on the test data
    with torch.no_grad():
        for batch in testloader:
            graph = batch_graph(batch, config)
            # TODO - do we need to update the graph parameters?
            # Having it in its own object might discard changes that we want to be permanent
            # Need to think on this a bit
            A = graph.A.to(device)
            y = graph.y.to(device)
            X = graph.X.to(device)

            z = gcn.encode(X, A)
            out = gcn.decode(z, A).view(-1)
            loss = criterion(out, y)

            total_loss += loss.item()
            batches += 1

            y_pred = torch.relu(out)
            error_tensor = torch.abs((y_pred - y) / (y + 1e-8))
            if error_tensor.max() > max_error:
                max_error = error_tensor.max()
            if error_tensor.min() > min_error:
                min_error = error_tensor.min()
            error += error_tensor.sum()
            predictions += len(y_pred)

    return total_loss / batches, error / predictions, max_error, min_error

def main():
    with wandb.init(project="535-final-c") as run:
        wandb.run.name = f"{run.config.hidden_dim}-width, {2 + run.config.layers}-layer"

        TEST_SIZE = run.config.test_size

        #split the dataframe
        train_df, test_df = train_test_split(
            data_frame,
            test_size=TEST_SIZE)

        #create datasets
        train_dataset = circuit_dataset(train_df, run.config)
        test_dataset = circuit_dataset(test_df, run.config)

        #create dataloaders
        trainloader = DataLoader(
            train_dataset,
            batch_size=run.config.batch_size,
            shuffle=True)

        testloader = DataLoader(
            test_dataset,
            batch_size=run.config.batch_size,
            shuffle=False)

        # Parameters
        hidden_dim = run.config.hidden_dim
        embedding_dim = train_dataset[0][1].shape[1]

        # Initialize the models
        gcn = GCN(embedding_dim, hidden_dim, embedding_dim, run.config.layers)
        gcn.to(device)
        optimizer = torch.optim.Adam(params=gcn.parameters(), lr=run.config.lr)

        for epoch in range(10):
            # trainloader, testloader = init_data(run.config)

            gcn, optimizer, trainloss = train(gcn, optimizer, trainloader, run.config)
            testloss, testMRE, maxRE, minRE = test(gcn, testloader, run.config)
            wandb.log({"Epoch": (epoch+1),
                    "Training Loss": trainloss,
                    "Avg Test Loss": testloss,
                    "Mean Relative Error": testMRE,
                    "Max Relative Error": maxRE,
                    "Min Relative Error": minRE,
                    })

            print(f"Finished Epoch {epoch+1}")
            print(f"Training Loss: {trainloss}")
            print(f"Avg Test Loss {testloss}")
            print(f"Mean Relative Error {testMRE}")

# Define the search space
sweep_configuration = {
    "method": "grid",
    # "metric": {"goal": "minimize", "name": "acc"},
    "parameters": {
        "batch_size": {"values": [32]},
        "model": {"values": ['block']},
        "lr": {"values": [.0001]},
        "layers": {"values": [3]},
        "hidden_dim": {"values": [64]},
        "test_size": {"values": [.3]},
    }
}

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="535-Final-c")

# Use the following instead for a local .py run:
if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=3)