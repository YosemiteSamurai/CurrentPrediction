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
import subprocess

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

def run_inference_and_copy(process_name, run_suffix, dataset_path, model_path, metadata_path=None):
    """Run predict.py for a trained checkpoint and copy privacy artifacts."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(privacy_dir, exist_ok=True)

    # Use the same dataset and model for prediction
    predict_py = os.path.join(os.path.dirname(__file__), "predict.py")
    output_csv = os.path.join(privacy_dir, f"predictions_{process_name}{run_suffix}.csv")  # original: output_csv = os.path.join(results_dir, f"predictions_{process_name}{run_suffix}.csv")

    # Call predict.py with correct arguments
    import subprocess
    predict_cmd = [
        sys.executable, predict_py,
        "--input", dataset_path,
        "--output", output_csv,
        "--checkpoint", model_path
    ]
    inference_started = time.perf_counter()
    print(f"\nRunning inference for privacy outputs...\n{' '.join(predict_cmd)}", flush=True)
    subprocess.run(predict_cmd, check=True)
    inference_seconds = time.perf_counter() - inference_started

    # Keep only one copy of non-model artifacts in privacy/.
    import shutil

    # predictions CSV is already written to privacy_dir
    print(f"[sweep] predictions saved in privacy: {output_csv}")

    # inference_outputs_*.npz is written next to output_csv (privacy_dir)
    npz_name = f"inference_outputs_{process_name}{run_suffix}.npz"
    npz_path = os.path.join(privacy_dir, npz_name)
    if os.path.exists(npz_path):
        print(f"[sweep] inference outputs saved in privacy: {npz_path}")
    else:
        print(f"[sweep] WARNING: {npz_path} not found in privacy/")

    # Copy model_*.pt
    model_dst = os.path.join(privacy_dir, f"model_{process_name}{run_suffix}.pt")
    shutil.copy2(model_path, model_dst)
    print(f"[sweep] Copied {model_path} to {model_dst}")

    # run_metadata is stored in privacy only; just verify it exists.
    if metadata_path and os.path.exists(metadata_path):
        print(f"[sweep] metadata saved in privacy: {metadata_path}")
    else:
        print(f"[sweep] WARNING: metadata not found at {metadata_path}; skipping metadata copy")

    run_tag = run_suffix.lstrip('_') or 'baseline'
    process_with_tag = f"{process_name}_{run_tag}"
    privacy_attack_py = os.path.join(os.path.dirname(__file__), '..', 'privacy', 'privacy_attack.py')
    attack_cmd = [
        sys.executable,
        privacy_attack_py,
        '--process', process_name,
        '--run-tag', run_tag,
        '--privacy_dir', privacy_dir,
    ]
    attacks_started = time.perf_counter()
    print(f"\n[sweep] Running privacy attacks...\n{' '.join(attack_cmd)}", flush=True)
    subprocess.run(attack_cmd, check=True)
    attacks_seconds = time.perf_counter() - attacks_started

    expected = [
        f"embedding_inversion_{process_with_tag}.npz",
        f"edge_reconstruction_{process_with_tag}.npz",
        f"membership_inference_{process_with_tag}.npz",
    ]
    for name in expected:
        path = os.path.join(privacy_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[sweep] expected privacy artifact missing: {path}")

    return {
        "inference_seconds": inference_seconds,
        "attacks_seconds": attacks_seconds,
        "post_training_seconds": inference_seconds + attacks_seconds,
    }

def run_split_backend(privacy_mode):
    """Dispatch training to privacy/split backend for SL-based modes."""
    split_sweep = os.path.join(os.path.dirname(__file__), '..', 'privacy', 'split', 'sweep.py')
    split_cwd = os.path.dirname(split_sweep)
    env = os.environ.copy()
    env['PRIVACY_MODE'] = privacy_mode
    # Force split mode in split backend for sl/both.
    env['SPLIT_LEARNING'] = '1'
    cmd = [sys.executable, split_sweep]
    print(f"[sweep] delegating to split backend: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=split_cwd, env=env, check=True)

def train_with_dp(gcn, optimizer, trainloader, config, device, noise_multiplier, max_grad_norm):
    """DP-style training: per-batch clip + Gaussian noise (practical approximation)."""
    from graph import batch_graph
    import torch

    criterion = torch.nn.L1Loss()
    total_loss = 0.0
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

        torch.nn.utils.clip_grad_norm_(gcn.parameters(), max_grad_norm)
        if noise_multiplier > 0:
            for p in gcn.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * (noise_multiplier * max_grad_norm))

        optimizer.step()
        total_loss += loss.item()
        batches += 1

    return gcn, optimizer, (total_loss / max(batches, 1))

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
    privacy_mode = os.environ.get('PRIVACY_MODE', 'neither').strip().lower() or 'neither'
    if privacy_mode not in {'neither', 'dp', 'sl', 'both'}:
        raise ValueError(f"[sweep] Invalid PRIVACY_MODE='{privacy_mode}'. Use neither|dp|sl|both")
    print(f"[sweep] PRIVACY_MODE: {privacy_mode}", flush=True)

    # Determine process name and artifact paths for output naming
    dataset_path = os.environ.get('DATASET_PATH', '')
    if dataset_path:
        process_name = os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')
    else:
        process_name = 'unknown'
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(privacy_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
    metadata_path = os.path.join(privacy_dir, f"run_metadata_{process_name}{run_suffix}.json")  # original: metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")

    inference_only = os.environ.get('INFERENCE_ONLY', '').strip().lower() in {'1', 'true', 'yes'}
    if inference_only:
        if privacy_mode in {'sl', 'both'}:
            raise ValueError("[sweep] INFERENCE_ONLY is currently only supported for monolithic modes (neither/dp)")
        print(f"[sweep] INFERENCE_ONLY=1: skipping training and reusing checkpoint {model_path}", flush=True)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[sweep] INFERENCE_ONLY requested but checkpoint not found: {model_path}")
        if not dataset_path:
            raise ValueError("[sweep] INFERENCE_ONLY requested but DATASET_PATH is empty")
        run_inference_and_copy(process_name, run_suffix, dataset_path, model_path, metadata_path)
        return

    if privacy_mode in {'sl', 'both'}:
        run_split_backend(privacy_mode)
        return

    run = wandb.init(

        entity="yosemitesamurai",
        project="CurrentPrediction",
        config=vars(config),

    )

    run_name = f"{process_name}_{run_tag}" if run_tag else process_name
    run.name = run_name  # original: run.name = f"{config.hidden_dim}-width, {2 + config.layers}-layer, {config.heads}-heads"
    print(f"Starting run: {run.name}", flush=True)
    print(f"Device: {device}", flush=True)


    data_prep_started = time.perf_counter()

    split_df = data_frame.copy()
    if row_ids is not None:
        split_df['ID'] = row_ids.to_numpy()
    train_df, test_df = split_train_test(split_df, test_size=config.test_size)

    # Save training IDs for membership inference ground-truth
    privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(privacy_dir, exist_ok=True)
    train_ids_path = os.path.join(privacy_dir, f'train_ids_{process_name}{run_suffix}.npy')  # original: train_ids_path = os.path.join(privacy_dir, 'train_ids.npy')
    if 'ID' in train_df.columns:
        import numpy as np
        np.save(train_ids_path, train_df['ID'].to_numpy())
        print(f"[sweep] Saved training IDs to {train_ids_path}")
    else:
        print(f"[sweep] WARNING: No 'ID' column found in training data; {train_ids_path} not saved.")  # original: print("[sweep] WARNING: No 'ID' column found in training data; train_ids.npy not saved.")

    # Keep ID out of model inputs; it is only needed for privacy membership labels.
    if 'ID' in train_df.columns:
        train_df = train_df.drop(columns=['ID'])  # original: train_df = train_df.drop(columns=['ID'])
    if 'ID' in test_df.columns:
        test_df = test_df.drop(columns=['ID'])  # original: test_df = test_df.drop(columns=['ID'])

    train_dataset = circuit_dataset(train_df, config)
    test_dataset = circuit_dataset(test_df, config)
    data_prep_seconds = time.perf_counter() - data_prep_started
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
    dp_noise_multiplier = float(os.environ.get('DP_NOISE_MULTIPLIER', '0.6'))
    dp_max_grad_norm = float(os.environ.get('DP_MAX_GRAD_NORM', '1.0'))

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()

        if privacy_mode == 'dp':
            gcn, optimizer, trainloss = train_with_dp(
                gcn, optimizer, trainloader, config, device,
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
            )
        else:
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

    training_seconds = time.perf_counter() - training_started
    checkpoint_started = time.perf_counter()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")  # original: results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)  # original: os.makedirs(results_dir, exist_ok=True)

    # Determine process name for output naming
    dataset_path = os.environ.get('DATASET_PATH', '')  # original: dataset_path = os.environ.get('DATASET_PATH', '')
    if dataset_path:
        process_name = os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')  # original: process_name = os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')
    else:
        process_name = 'unknown'  # original: process_name = 'unknown'

    checkpoint = {
        "model_state_dict": gcn.state_dict(),
        "config": vars(config),
        "label_log_mean": label_log_mean,
        "label_log_std": label_log_std,
        "embedding_dim": embedding_dim,
        "scaler": scaler,
    }

    model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")  # original: model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}", flush=True)
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    load_time_seconds = globals().get('DATASET_LOAD_SECONDS')
    if load_time_seconds is None:
        load_time_seconds = float('nan')
    metadata = {
        "process_name": process_name,
        "dataset_path": dataset_path,
        "run_tag": run_tag,
        "privacy_mode": privacy_mode,
        "dataset_load_seconds": load_time_seconds,
        "data_prep_seconds": data_prep_seconds,
        "training_seconds": training_seconds,
        "epoch_seconds": epoch_times,
        "checkpoint_seconds": checkpoint_seconds,
        "epochs": config.epochs,
        "test_size": config.test_size,
        "batch_size": config.batch_size,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "training_samples_per_second": (len(train_dataset) * config.epochs) / max(training_seconds, 1e-12),
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "heads": config.heads,
    }
    if privacy_mode == 'dp':
        metadata["dp_noise_multiplier"] = dp_noise_multiplier
        metadata["dp_max_grad_norm"] = dp_max_grad_norm
    metadata_path = os.path.join(privacy_dir, f"run_metadata_{process_name}{run_suffix}.json")  # original: metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")  # original: metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")
    post_training_times = run_inference_and_copy(process_name, run_suffix, dataset_path, model_path, metadata_path)
    metadata.update(post_training_times)
    metadata["end_to_end_seconds"] = (
        metadata["data_prep_seconds"]
        + metadata["training_seconds"]
        + metadata["checkpoint_seconds"]
        + metadata["post_training_seconds"]
    )

    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"[sweep] Saved run metadata to {metadata_path}", flush=True)
    print(f"[sweep] total end-to-end time: {metadata['end_to_end_seconds']:.2f}s", flush=True)
    run.finish()

if __name__ == '__main__':
    main(config)
