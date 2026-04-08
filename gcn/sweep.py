# =============================================================================
# sweep.py -- Hyperparameter Sweep Entry Point
#
# Defines the W&B sweep configuration and the main() training run function.
# Data loading, the dataset class, loss functions, and the train/test loops
# all live in dataset.py. The GCN model architecture is in gcn.py.
#
# W&B hyperparameter sweep (sweep_configuration):
#   - Grid search over batch_size, model (block/split), lr, layers,
#     hidden_dim, and test_size
#   - Runs via wandb.agent, logging all metrics per epoch
# =============================================================================

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import wandb

from dataset import circuit_dataset, data_frame, device, train, test
from gcn import GCN

wandb.login()


def main():

    with wandb.init(project="535-final-c") as run:
        wandb.run.name = f"{run.config.hidden_dim}-width, {2 + run.config.layers}-layer, {run.config.heads}-heads"

        TEST_SIZE = run.config.test_size

        # split the dataframe
        train_df, test_df = train_test_split(
            data_frame,
            test_size=TEST_SIZE)

        # create datasets
        train_dataset = circuit_dataset(train_df, run.config)
        test_dataset = circuit_dataset(test_df, run.config)

        # create dataloaders
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

        # Initialize the model
        gcn = GCN(embedding_dim, hidden_dim, embedding_dim, run.config.layers, heads=run.config.heads)
        gcn.to(device)
        optimizer = torch.optim.Adam(params=gcn.parameters(), lr=run.config.lr)

        for epoch in range(10):
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
        "heads": {"values": [4]},
        "test_size": {"values": [.3]},
    }
}

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="535-Final-c")

# Use the following instead for a local .py run:
if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=3)
