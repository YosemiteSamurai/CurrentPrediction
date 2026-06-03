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

from types import SimpleNamespace
import json
import os
import sys
import argparse
import random
import time

print("[sweep] module loaded", flush=True)

# Accept dataset path from environment variable or command-line argument
def get_dataset_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset JSON file')
    parser.add_argument('--run-tag', type=str, default=None, help='Tag appended to saved artifacts for repeated runs')
    args, _ = parser.parse_known_args()
    env_dataset = os.environ.get('DATASET')
    if args.dataset:
        return args.dataset
    elif env_dataset:
        # If only a dataset name (not a path) is given, resolve to dataset/ directory
        if not os.path.isabs(env_dataset) and not os.path.exists(env_dataset):
            base = os.path.join(os.path.dirname(__file__), '..', 'dataset')
            candidate = os.path.join(base, env_dataset)
            if os.path.exists(candidate + '.json'):
                return candidate + '.json'
            elif os.path.exists(candidate):
                return candidate
        return env_dataset
    return None

print("[sweep] resolving dataset path...", flush=True)
DATASET_PATH = get_dataset_path()
print("[sweep] dataset path resolved", flush=True)

os.environ['DATASET_PATH'] = DATASET_PATH if DATASET_PATH else ''

print("[sweep] Starting sweep.py", flush=True)
print("[sweep] DATASET_PATH:", DATASET_PATH, flush=True)

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
    run_tag          = os.environ.get('RUN_TAG', '').strip(),
)

def split_train_test(df, test_size=0.3, seed=42):
    """Simple dataframe split without sklearn dependency."""
    n = len(df)
    if n == 0:
        return df, df

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_n = int(round(n * test_size))
    test_n = max(1, min(n - 1, test_n)) if n > 1 else 0

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def main(config):

    print("[sweep] importing torch...", flush=True)
    import torch
    from torch.utils.data import DataLoader
    print("[sweep] torch imports done", flush=True)

    print("[sweep] importing wandb...", flush=True)
    import wandb
    print("[sweep] wandb import done", flush=True)

    print("[sweep] importing dataset module...", flush=True)
    from dataset import circuit_dataset, data_frame, row_ids, device, train, test, label_log_mean, label_log_std, scaler
    print("[sweep] dataset import done", flush=True)

    print("[sweep] importing GAN...", flush=True)
    from gan import GAN
    print("[sweep] GAN import done", flush=True)

    run_tag = config.run_tag or os.environ.get('RUN_TAG', '').strip()
    run_suffix = f"_{run_tag}" if run_tag else ""

    run = wandb.init(

        entity="yosemitesamurai",
        project="CurrentPrediction",
        config=vars(config),

    )

    run.name = f"{config.hidden_dim}-width, {2 + config.layers}-layer, {config.heads}-heads"
    print(f"Starting run: {run.name}", flush=True)
    print(f"Device: {device}", flush=True)


    split_df = data_frame.copy()
    if row_ids is not None:
        split_df['ID'] = row_ids.to_numpy()
    train_df, test_df = split_train_test(split_df, test_size=config.test_size)

    # Save training IDs for membership inference ground-truth
    privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(privacy_dir, exist_ok=True)
    train_ids_path = os.path.join(privacy_dir, 'train_ids.npy')
    if 'ID' in train_df.columns:
        import numpy as np
        np.save(train_ids_path, train_df['ID'].to_numpy())
        print(f"[sweep] Saved training IDs to {train_ids_path}")
    else:
        print("[sweep] WARNING: No 'ID' column found in training data; train_ids.npy not saved.")

    # Keep ID out of model inputs; it is only needed for privacy membership labels.
    if 'ID' in train_df.columns:
        train_df = train_df.drop(columns=['ID'])  # original: train_df = train_df.drop(columns=['ID'])
    if 'ID' in test_df.columns:
        test_df = test_df.drop(columns=['ID'])  # original: test_df = test_df.drop(columns=['ID'])

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
    gcn = GAN(embedding_dim, config.hidden_dim, embedding_dim, config.layers, heads=config.heads)
    gcn.to(device)
    print(f"[sweep] model initialized, starting training...", flush=True)
    optimizer = torch.optim.Adam(params=gcn.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)

    training_started = time.perf_counter()
    epoch_times = []

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()

        gcn, optimizer, trainloss = train(gcn, optimizer, trainloader, config)
        testloss, testMRE, maxRE, minRE = test(gcn, testloader, config)
        scheduler.step(testloss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_seconds = time.perf_counter() - epoch_started

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
        print(f"Epoch time: {epoch_seconds:.2f}s", flush=True)
        epoch_times.append(epoch_seconds)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Determine process name for output naming
    dataset_path = os.environ.get('DATASET_PATH', '')
    if dataset_path:
        process_name = os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')
    else:
        process_name = 'unknown'

    checkpoint = {
        "model_state_dict": gcn.state_dict(),
        "config": vars(config),
        "label_log_mean": label_log_mean,
        "label_log_std": label_log_std,
        "embedding_dim": embedding_dim,
        "scaler": scaler,
    }

    model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}", flush=True)

    load_time_seconds = globals().get('DATASET_LOAD_SECONDS')
    if load_time_seconds is None:
        load_time_seconds = float('nan')
    metadata = {
        "process_name": process_name,
        "dataset_path": dataset_path,
        "run_tag": run_tag,
        "dataset_load_seconds": load_time_seconds,
        "training_seconds": time.perf_counter() - training_started,
        "epoch_seconds": epoch_times,
        "epochs": config.epochs,
        "test_size": config.test_size,
        "batch_size": config.batch_size,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "heads": config.heads,
    }
    metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"[sweep] Saved run metadata to {metadata_path}", flush=True)

    # Always run inference after training
    # Use the same dataset and model for prediction
    predict_py = os.path.join(os.path.dirname(__file__), "predict.py")
    output_csv = os.path.join(results_dir, f"predictions_{process_name}.csv")
    if run_tag:
        output_csv = os.path.join(results_dir, f"predictions_{process_name}{run_suffix}.csv")
    # Call predict.py with correct arguments
    import subprocess
    predict_cmd = [
        sys.executable, predict_py,
        "--input", dataset_path,
        "--output", output_csv,
        "--checkpoint", model_path
    ]
    print(f"\nRunning inference for privacy outputs...\n{' '.join(predict_cmd)}", flush=True)
    subprocess.run(predict_cmd, check=True)

    # Copy privacy-related files to privacy/
    privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(privacy_dir, exist_ok=True)
    import shutil
    # Copy predictions CSV
    privacy_csv = os.path.join(privacy_dir, f"predictions_{process_name}{run_suffix}.csv")
    shutil.copy2(output_csv, privacy_csv)
    print(f"[sweep] Copied {output_csv} to {privacy_csv}")
    # Copy inference_outputs_*.npz
    npz_name = f"inference_outputs_{process_name}{run_suffix}.npz"
    npz_src = os.path.join(results_dir, npz_name)
    npz_dst = os.path.join(privacy_dir, npz_name)
    if os.path.exists(npz_src):
        shutil.copy2(npz_src, npz_dst)
        print(f"[sweep] Copied {npz_src} to {npz_dst}")
    else:
        print(f"[sweep] WARNING: {npz_src} not found; not copied to privacy/")
    # Copy model_*.pt
    model_dst = os.path.join(privacy_dir, f"model_{process_name}{run_suffix}.pt")
    shutil.copy2(model_path, model_dst)
    print(f"[sweep] Copied {model_path} to {model_dst}")
    metadata_dst = os.path.join(privacy_dir, f"run_metadata_{process_name}{run_suffix}.json")
    shutil.copy2(metadata_path, metadata_dst)
    print(f"[sweep] Copied {metadata_path} to {metadata_dst}")
    print(f"[sweep] total training time: {time.perf_counter() - training_started:.2f}s", flush=True)
    run.finish()

if __name__ == '__main__':
    main(config)
