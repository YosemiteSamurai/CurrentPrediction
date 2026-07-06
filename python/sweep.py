# =============================================================================
# sweep.py -- Hyperparameter Sweep / Training Entry Point
#
# Defines the hyperparameter configuration and the main() training run.
# Data loading, the dataset class, loss functions, and the train/test loops
# all live in dataset.py. The GAT model architecture is in gan.py. The
# split-learning encoder + foundry wrapper live in process_encoder.py and
# foundry.py respectively.
#
# A single in-process entry point handles all four privacy modes, selected
# via the PRIVACY_MODE environment variable:
#
#   neither : monolithic GAT over raw BSIM4 columns (v2.0 baseline).
#   dp      : monolithic GAT + DP-SGD (per-batch clip + Gaussian noise).
#   sl      : split learning -- a private ProcessEncoder on the foundry
#             side maps raw BSIM4 -> embedding; the GAT consumes the
#             embedding via models.block_2inv_public + assemble_block_2inv.
#   both    : split learning + DP-SGD on the design-house GAT.
#
# PRIVACY_MODE maps to two config switches: config.split_learning (sl/both)
# and config.dp_enabled (dp/both). The DP step lives inside dataset.train();
# the split path adds a second optimizer on the foundry side.
#
# Runs training directly with a plain config object. W&B is used for logging
# via wandb.init but does not require sweep/agent API access.
# =============================================================================

from types import SimpleNamespace
import json
import os
import sys
import argparse
import time
import subprocess


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
        # If only a dataset name (not a path) is given, resolve to datasets/ directory  # original: # If only a dataset name (not a path) is given, resolve to dataset/ directory
        if not os.path.isabs(env_dataset) and not os.path.exists(env_dataset):
            base = os.path.join(os.path.dirname(__file__), '..', 'datasets')  # original: base = os.path.join(os.path.dirname(__file__), '..', 'dataset')
            candidate = os.path.join(base, env_dataset)
            if os.path.exists(candidate + '.json'):
                return candidate + '.json'
            elif os.path.exists(candidate):
                return candidate
        return env_dataset
    return None

DATASET_PATH = get_dataset_path()

os.environ['DATASET_PATH'] = DATASET_PATH if DATASET_PATH else ''

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

    # v3.0 split-learning configuration. main() sets split_learning and
    # dp_enabled from PRIVACY_MODE; the inner knobs only matter when the
    # corresponding path is active.
    split_learning   = False,
    embed_dim        = 16,
    encoder_hidden   = 64,
    nodes_per_graph  = 6,
    pmos_offset      = 4,
    dp_enabled       = False,
    dp_noise_multiplier = float(os.environ.get('DP_NOISE_MULTIPLIER', '0.6')),
    dp_max_grad_norm = float(os.environ.get('DP_MAX_GRAD_NORM', '1.0')),
)

def _artifact(process_name, run_suffix, description, ext, sub=None):
    """Build a results/ artifact name: <process>_<tag>_<description>[_<sub>].<ext>.

    The tag comes from run_suffix ('_<tag>' or '') and defaults to 'baseline'
    when untagged, so every results/ artifact follows one naming convention.
    """
    tag = run_suffix.lstrip('_') or 'baseline'
    name = f"{process_name}_{tag}_{description}"
    if sub:
        name = f"{name}_{sub}"
    return f"{name}.{ext}"

def _save_train_ids(dataset_path, process_name, run_suffix, test_size, inputs_dir):
    """Reconstruct and save the training-set IDs for membership inference.

    Re-splits the raw dataset JSON with the same (test_size, random_state)
    used for the training split so the saved IDs match the rows the model
    actually trained on.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    train_ids_path = os.path.join(inputs_dir, f'train_ids_{process_name}{run_suffix}.npy')
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"[sweep] WARNING: dataset path not found; cannot save {train_ids_path}", flush=True)
        return

    with open(dataset_path, 'r') as file_handle:
        raw_data = json.load(file_handle)

    ids = np.array([row.get('ID') for row in raw_data])
    if any(v is None for v in ids):
        print(f"[sweep] WARNING: dataset has missing ID values; cannot save {train_ids_path}", flush=True)
        return

    _dummy = np.zeros(len(ids), dtype=np.int8)
    train_ids, _unused, _dummy_train, _dummy_test = train_test_split(
        ids,
        _dummy,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    np.save(train_ids_path, train_ids)
    print(f"[sweep] Saved training IDs to {train_ids_path}", flush=True)

def run_inference_and_copy(process_name, run_suffix, dataset_path, model_path, metadata_path=None):
    """Run predict.py for a trained checkpoint and copy privacy artifacts."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    inputs_dir = os.path.join(os.path.dirname(__file__), '..', 'attacks', 'inputs')  # original: privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(inputs_dir, exist_ok=True)  # original: os.makedirs(privacy_dir, exist_ok=True)

    # Use the same dataset and model for prediction
    predict_py = os.path.join(os.path.dirname(__file__), "predict.py")
    output_csv = os.path.join(results_dir, _artifact(process_name, run_suffix, 'predictions', 'csv'))  # original: output_csv = os.path.join(results_dir, f"predictions_{process_name}{run_suffix}.csv")

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

    # Keep only one copy of non-model artifacts.
    # predictions CSV + inference_outputs npz are written to results_dir by predict.py
    print(f"[sweep] predictions saved in results: {output_csv}")  # original: print(f"[sweep] predictions saved in privacy: {output_csv}")

    # inference npz is written next to output_csv (results_dir) by predict.py
    npz_name = _artifact(process_name, run_suffix, 'inference', 'npz')  # original: npz_name = f"inference_outputs_{process_name}{run_suffix}.npz"
    npz_path = os.path.join(results_dir, npz_name)  # original: npz_path = os.path.join(privacy_dir, npz_name)
    if os.path.exists(npz_path):
        print(f"[sweep] inference outputs saved in results: {npz_path}")  # original: print(f"[sweep] inference outputs saved in privacy: {npz_path}")
    else:
        print(f"[sweep] WARNING: {npz_path} not found in results/")  # original: print(f"[sweep] WARNING: {npz_path} not found in privacy/")

    # The model checkpoint already lives in results_dir; no extra copy needed.

    # run_metadata is stored in results; just verify it exists.
    if metadata_path and os.path.exists(metadata_path):
        print(f"[sweep] metadata saved in results: {metadata_path}")  # original: print(f"[sweep] metadata saved in privacy: {metadata_path}")
    else:
        print(f"[sweep] WARNING: metadata not found at {metadata_path}; skipping metadata copy")

    run_attacks = os.environ.get('RUN_ATTACKS', '').strip().lower() in {'1', 'true', 'yes'}
    if not run_attacks:
        print("[sweep] RUN_ATTACKS not set; skipping privacy attacks (inference outputs retained).", flush=True)
        return {
            "inference_seconds": inference_seconds,
            "attacks_seconds": 0.0,
            "post_training_seconds": inference_seconds,
        }

    run_tag = run_suffix.lstrip('_') or 'baseline'
    process_with_tag = f"{process_name}_{run_tag}"

    # Ensure the per-run-tag ground-truth artifacts exist before the attacks run
    # (original_embeddings_*, ground_truth_edges_*, membership_labels_*). Without
    # this, arbitrary run tags (e.g. 'test') fail with FileNotFoundError because
    # only run_all_validations.sh's fixed tags were ever prepared.
    validate_py = os.path.join(os.path.dirname(__file__), '..', 'attacks', 'validate_privacy_artifacts.py')
    validate_cmd = [
        sys.executable,
        validate_py,
        '--dataset', dataset_path,
        '--tag', run_tag,
        '--privacy-dir', inputs_dir,
        '--results-dir', results_dir,
    ]
    print(f"\n[sweep] Preparing privacy attack ground truths...\n{' '.join(validate_cmd)}", flush=True)
    subprocess.run(validate_cmd, check=True)

    privacy_attack_py = os.path.join(os.path.dirname(__file__), '..', 'attacks', 'privacy_attack.py')  # original: privacy_attack_py = os.path.join(os.path.dirname(__file__), '..', 'privacy', 'privacy_attack.py')
    attack_cmd = [
        sys.executable,
        privacy_attack_py,
        '--process', process_name,
        '--run-tag', run_tag,
        '--privacy_dir', inputs_dir,  # original: '--privacy_dir', privacy_dir,
        '--results_dir', results_dir,
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
        path = os.path.join(inputs_dir, name)  # original: path = os.path.join(privacy_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[sweep] expected privacy artifact missing: {path}")

    return {
        "inference_seconds": inference_seconds,
        "attacks_seconds": attacks_seconds,
        "post_training_seconds": inference_seconds + attacks_seconds,
    }

def main(config):

    import torch
    from torch.utils.data import DataLoader

    import wandb

    from dataset import (
        circuit_dataset, data_frame, data_frame_public, device,
        train, test, label_log_mean, label_log_std, scaler,
        public_scaler, process_table,
        private_pmos_scaler, private_nmos_scaler,
        PUBLIC_FEATURE_COLS, PRIVATE_FEATURE_COLS,
    )

    from gan import GAN
    from process_encoder import ProcessEncoder
    from foundry import Foundry

    from sklearn.model_selection import train_test_split

    run_tag = config.run_tag or os.environ.get('RUN_TAG', '').strip()
    run_suffix = f"_{run_tag}" if run_tag else ""
    privacy_mode = os.environ.get('PRIVACY_MODE', 'neither').strip().lower() or 'neither'
    if privacy_mode not in {'neither', 'dp', 'sl', 'both'}:
        raise ValueError(f"[sweep] Invalid PRIVACY_MODE='{privacy_mode}'. Use neither|dp|sl|both")

    # PRIVACY_MODE drives the two training switches: split learning (sl/both)
    # and DP-SGD on the design-house GAT (dp/both). The DP step itself lives
    # inside dataset.train(), gated by config.dp_enabled.
    config.split_learning = privacy_mode in {'sl', 'both'}
    config.dp_enabled = privacy_mode in {'dp', 'both'}
    print(f"[sweep] PRIVACY_MODE: {privacy_mode} "
          f"(split_learning={config.split_learning}, dp_enabled={config.dp_enabled})", flush=True)
    if config.dp_enabled:
        print(f"[sweep] DP-SGD: sigma={config.dp_noise_multiplier}, C={config.dp_max_grad_norm}", flush=True)

    # Determine process name and artifact paths for output naming
    dataset_path = os.environ.get('DATASET_PATH', '')
    if dataset_path:
        process_name = os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')
    else:
        process_name = 'unknown'
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    inputs_dir = os.path.join(os.path.dirname(__file__), '..', 'attacks', 'inputs')  # original: privacy_dir = os.path.join(os.path.dirname(__file__), '..', 'privacy')
    os.makedirs(inputs_dir, exist_ok=True)  # original: os.makedirs(privacy_dir, exist_ok=True)
    model_path = os.path.join(results_dir, _artifact(process_name, run_suffix, 'model', 'pt'))  # original: model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
    metadata_path = os.path.join(results_dir, _artifact(process_name, run_suffix, 'runmetadata', 'json'))  # original: metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")

    inference_only = os.environ.get('INFERENCE_ONLY', '').strip().lower() in {'1', 'true', 'yes'}
    if inference_only:
        if config.split_learning:
            raise ValueError("[sweep] INFERENCE_ONLY is currently only supported for monolithic modes (neither/dp)")
        print(f"[sweep] INFERENCE_ONLY=1: skipping training and reusing checkpoint {model_path}", flush=True)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[sweep] INFERENCE_ONLY requested but checkpoint not found: {model_path}")
        if not dataset_path:
            raise ValueError("[sweep] INFERENCE_ONLY requested but DATASET_PATH is empty")
        run_inference_and_copy(process_name, run_suffix, dataset_path, model_path, metadata_path)
        return

    wandb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run = wandb.init(

        entity="yosemitesamurai",
        project="CurrentPrediction",
        config=vars(config),
        dir=wandb_dir,

    )

    run_name = f"{process_name}_{run_tag}" if run_tag else process_name
    run.name = run_name  # original: run.name = f"{config.hidden_dim}-width, {2 + config.layers}-layer, {config.heads}-heads"
    print(f"Starting run: {run.name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"split_learning: {config.split_learning}", flush=True)


    data_prep_started = time.perf_counter()

    # Split mode trains the design-house GAT on public columns only; the
    # private BSIM4 columns cross the cut layer as a foundry embedding.
    df_to_use = data_frame_public if config.split_learning else data_frame
    train_df, test_df = train_test_split(
        df_to_use, test_size=config.test_size, random_state=42, shuffle=True)

    # Save training IDs for membership inference ground-truth (only when attacks will run).
    run_attacks = os.environ.get('RUN_ATTACKS', '').strip().lower() in {'1', 'true', 'yes'}
    if run_attacks:
        _save_train_ids(dataset_path, process_name, run_suffix, config.test_size, inputs_dir)
    else:
        print("[sweep] RUN_ATTACKS not set; skipping training-ID dump.", flush=True)

    train_dataset = circuit_dataset(train_df, config)
    test_dataset = circuit_dataset(test_df, config)
    data_prep_seconds = time.perf_counter() - data_prep_started

    trainloader = DataLoader(

        train_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    testloader = DataLoader(

        test_dataset,
        batch_size=config.batch_size,
        shuffle=False)

    # Sample a row to determine input dimensionality. For split mode this is
    # the public-feature dim including zero-padded embedding slots that the
    # foundry fills in via assemble_block_2inv.
    embedding_dim = train_dataset[0][1].shape[1]
    gan = GAN(embedding_dim, config.hidden_dim, embedding_dim, config.layers, heads=config.heads)
    gan.to(device)
    optimizer = torch.optim.Adam(params=gan.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)

    # Foundry side of the cut layer (split modes only). The private
    # ProcessEncoder maps raw BSIM4 -> embedding; the foundry owns its own
    # optimizer and (optionally) DP-clips its gradients in foundry.backward().
    foundry = None
    if config.split_learning:
        encoder = ProcessEncoder(
            n_pmos_params=15,
            n_nmos_params=18,
            embed_dim=config.embed_dim,
            hidden=config.encoder_hidden,
        )
        foundry = Foundry(
            encoder=encoder,
            process_table=process_table,
            pmos_scaler=private_pmos_scaler,
            nmos_scaler=private_nmos_scaler,
            lr=config.lr,
            device=device,
            dp_enabled=config.dp_enabled,
            dp_noise_multiplier=config.dp_noise_multiplier,
            dp_max_grad_norm=config.dp_max_grad_norm,
        )

    training_started = time.perf_counter()
    epoch_times = []

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()

        gan, optimizer, trainloss = train(gan, optimizer, trainloader, config, foundry=foundry)
        testloss, testMRE, maxRE, minRE = test(gan, testloader, config, foundry=foundry)
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

    # Common design-house payload. For split mode this is the strict side of
    # the cut: model weights + public_scaler only -- no global scaler (which
    # would carry BSIM4 means/stds), no foundry payload. For monolithic mode
    # it carries the original global scaler used at inference time.
    design_house_ckpt = {
        "model_state_dict": gan.state_dict(),
        "config": vars(config),
        "label_log_mean": label_log_mean,
        "label_log_std": label_log_std,
        "embedding_dim": embedding_dim,
        "public_feature_cols": list(PUBLIC_FEATURE_COLS),
        "private_feature_cols": list(PRIVATE_FEATURE_COLS),
        "split_learning": bool(config.split_learning),
    }
    if config.split_learning:
        design_house_ckpt["public_scaler"] = public_scaler
    else:
        design_house_ckpt["scaler"] = scaler

    if config.split_learning:
        # In-process convenience: bundle the foundry payload alongside the
        # design-house payload as model_<process>.pt. In a real two-party
        # deployment the foundry portion would be persisted under foundry
        # control and model_<process>.pt would be design-house-only.
        bundled = dict(design_house_ckpt)
        if foundry is not None:
            bundled["foundry_state_dict"] = foundry.state_dict()
        torch.save(bundled, model_path)
        print(f"Bundled (design-house + foundry) checkpoint saved to {model_path}", flush=True)

        # Strict design-house-only checkpoint, identical to the bundled one
        # minus foundry_state_dict; the artifact a real design-house
        # deployment would receive. Saving it here lets the user verify the
        # cut is real.
        dh_path = os.path.join(results_dir, _artifact(process_name, run_suffix, 'model', 'pt', sub='design_house'))  # original: dh_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}_design_house.pt")
        torch.save(design_house_ckpt, dh_path)
        print(f"Strict design-house checkpoint saved to {dh_path}", flush=True)

        if foundry is not None:
            foundry_path = os.path.join(results_dir, _artifact(process_name, run_suffix, 'model', 'pt', sub='foundry'))  # original: foundry_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}_foundry.pt")
            torch.save(foundry.state_dict(), foundry_path)
            print(f"Foundry-side checkpoint saved to {foundry_path}", flush=True)
    else:
        torch.save(design_house_ckpt, model_path)
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
        "split_learning": bool(config.split_learning),
        "dp_enabled": bool(config.dp_enabled),
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
    if config.dp_enabled:
        metadata["dp_noise_multiplier"] = float(config.dp_noise_multiplier)
        metadata["dp_max_grad_norm"] = float(config.dp_max_grad_norm)
    metadata_path = os.path.join(results_dir, _artifact(process_name, run_suffix, 'runmetadata', 'json'))  # original: metadata_path = os.path.join(results_dir, f"run_metadata_{process_name}{run_suffix}.json")
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
