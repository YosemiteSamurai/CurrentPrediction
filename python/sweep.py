# =============================================================================
# sweep.py -- Hyperparameter Sweep Entry Point
#
# Defines the hyperparameter configuration and the main() training run.
# Data loading, the dataset class, loss functions, and the train/test loops
# all live in dataset.py. The GCN model architecture is in gcn.py.
#
# Runs training directly with a plain config object. W&B is used for logging
# via wandb.init but does not require sweep/agent API access.
# =============================================================================

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from types import SimpleNamespace
import os
import torch
import wandb
from dataset import circuit_dataset, data_frame, device, train, test, label_log_mean, label_log_std, scaler
from gan import GAN

print("[sweep] base imports done", flush=True)
print("[sweep] dataset imported", flush=True)
print("[sweep] GCN imported", flush=True)

config = SimpleNamespace(

    batch_size       = 32,
    model            = 'block',
    lr               = 0.0001,
    layers           = 3,
    hidden_dim       = 64,
    heads            = 4,
    test_size        = 0.3,
    epochs           = 100,
    edges_per_graph  = 7,
    target_edge_idx  = 3,
)

def main(config):

    run = wandb.init(

        entity="yosemitesamurai",
        project="CurrentPrediction",
        config=vars(config),

    )

    run.name = f"{config.hidden_dim}-width, {2 + config.layers}-layer, {config.heads}-heads"
    print(f"Starting run: {run.name}", flush=True)
    print(f"Device: {device}", flush=True)

    train_df, test_df = train_test_split(

        data_frame,
        test_size=config.test_size)

    train_dataset = circuit_dataset(train_df, config)
    test_dataset = circuit_dataset(test_df, config)
    print(f"[sweep] datasets created: {len(train_dataset)} train, {len(test_dataset)} test", flush=True)

    trainloader = DataLoader(

        train_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    testloader = DataLoader(

        test_dataset,
        batch_size=config.batch_size,
        shuffle=False)

    embedding_dim = train_dataset[0][1].shape[1]
    gan = GAN(embedding_dim, config.hidden_dim, embedding_dim, config.layers, heads=config.heads)
    gcn.to(device)
    print(f"[sweep] model initialized, starting training...", flush=True)
    optimizer = torch.optim.Adam(params=gcn.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)

    for epoch in range(config.epochs):

        gcn, optimizer, trainloss = train(gcn, optimizer, trainloader, config)
        testloss, testMRE, maxRE, minRE = test(gcn, testloader, config)
        scheduler.step(testloss)
        gan.load_state_dict(checkpoint["model_state_dict"])
        gan.to(device)
        gan.eval()
        current_lr = optimizer.param_groups[0]['lr']

        run.log({"Epoch": epoch + 1,
                 "Training Loss": trainloss,
                 "Avg Test Loss": testloss,
                 "Mean Relative Error (I_target)": testMRE,
                 "Max Relative Error (I_target)": maxRE,
                 "Min Relative Error (I_target)": minRE,
                 "LR": current_lr,
                 })
        
        print(f"Finished Epoch {epoch+1}", flush=True)
        print(f"Training Loss: {trainloss}", flush=True)
        print(f"Avg Test Loss {testloss}", flush=True)
        print(f"Mean Relative Error (I_target): {testMRE}", flush=True)
        print(f"LR: {current_lr}", flush=True)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": gcn.state_dict(),
        "config": vars(config),
        "label_log_mean": label_log_mean,
        "label_log_std": label_log_std,
        "embedding_dim": embedding_dim,
        "scaler": scaler,
    }

    model_path = os.path.join(results_dir, "model.pt")
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}", flush=True)
    run.finish()

if __name__ == '__main__':
    main(config)
