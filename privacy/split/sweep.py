# =============================================================================
# sweep.py -- Training Entry Point
#
# Defines the hyperparameter configuration and the main() training run.
# Data loading, the dataset class, loss functions, and the train/test loops
# all live in dataset.py. The GAT model architecture is in gan.py. The
# split-learning encoder + foundry wrapper live in process_encoder.py and
# foundry.py respectively.
#
# Two paths share this file, gated by `config.split_learning`:
#
#   False (v2.0 monolithic): one optimizer over a GAT that consumes raw
#       BSIM4 columns directly via models.block_2inv. Reproduces the
#       GANv2.0 baseline numbers.
#
#   True  (v3.0 split):      a private ProcessEncoder on the foundry side
#       maps raw BSIM4 -> 16-dim embedding; the embedding crosses the cut
#       layer; the GAT consumes the embedding instead of raw BSIM4 via
#       models.block_2inv_public + models.assemble_block_2inv. Two
#       independent optimizers (one per side) update via the standard
#       SplitNN detach/reattach idiom (Vepakomma et al., 2018).
#
# An env var override (SPLIT_LEARNING=0 to run monolithic, 1 to run split)
# lets the SLURM submitter pick the path without editing this file.
# Default is the v3.0 split path.
# =============================================================================

import sys
import os
# Ensure privacy/split/ is resolved first so 'from dataset import ...' picks up
# privacy/split/dataset.py rather than python/dataset.py (which is on sys.path
# because SLURM sets CWD to the python/ directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from types import SimpleNamespace
import json
import time
import subprocess
import shutil
import numpy as np
import torch
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

print("[sweep] base imports done", flush=True)
print("[sweep] dataset imported", flush=True)
print("[sweep] GAN imported", flush=True)

def _get_process_name_from_env():
    dataset_path = os.environ.get('DATASET_PATH', '').strip()
    if dataset_path:
        return os.path.splitext(os.path.basename(dataset_path))[0].replace('dataset_', '')
    dataset_name = os.environ.get('DATASET', '').strip()
    if dataset_name:
        return os.path.splitext(os.path.basename(dataset_name))[0].replace('dataset_', '')
    return 'unknown'


def _resolve_dataset_path(project_root):
    dataset_path = os.environ.get('DATASET_PATH', '').strip()
    if dataset_path and os.path.exists(dataset_path):
        return dataset_path
    dataset_name = os.environ.get('DATASET', '').strip()
    if dataset_name:
        candidate = dataset_name
        if not os.path.isabs(candidate):
            candidate = os.path.join(project_root, 'dataset', candidate)
        if not candidate.endswith('.json'):
            candidate += '.json'
        if os.path.exists(candidate):
            return candidate
    return ''


def _save_train_ids(dataset_path, process_name, run_suffix, test_size, privacy_dir):
    train_ids_path = os.path.join(privacy_dir, f'train_ids_{process_name}{run_suffix}.npy')
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


def _run_inference_and_attacks(project_root, process_name, run_suffix, dataset_path, model_path, privacy_dir):
    if not dataset_path:
        print("[sweep] WARNING: DATASET_PATH not set; skipping split inference/attacks", flush=True)
        return {
            "inference_seconds": float('nan'),
            "attacks_seconds": float('nan'),
            "post_training_seconds": float('nan'),
        }

    split_predict_py = os.path.join(project_root, 'privacy', 'split', 'predict.py')
    output_csv = os.path.join(privacy_dir, f"predictions_{process_name}{run_suffix}.csv")
    inference_started = time.perf_counter()
    predict_cmd = [
        sys.executable,
        split_predict_py,
        '--input', dataset_path,
        '--output', output_csv,
        '--checkpoint', model_path,
    ]
    print(f"\n[sweep] Running split inference...\n{' '.join(predict_cmd)}", flush=True)
    subprocess.run(predict_cmd, cwd=os.path.dirname(split_predict_py), check=True)
    inference_seconds = time.perf_counter() - inference_started

    privacy_attack_py = os.path.join(project_root, 'privacy', 'privacy_attack.py')
    run_tag = run_suffix.lstrip('_') or 'baseline'
    process_with_tag = f"{process_name}_{run_tag}"
    attacks_started = time.perf_counter()
    attack_cmd = [
        sys.executable,
        privacy_attack_py,
        '--process', process_name,
        '--run-tag', run_tag,
        '--privacy_dir', privacy_dir,
    ]
    print(f"\n[sweep] Running privacy attacks...\n{' '.join(attack_cmd)}", flush=True)
    subprocess.run(attack_cmd, cwd=os.path.join(project_root, 'privacy'), check=True)
    attacks_seconds = time.perf_counter() - attacks_started

    expected = [
        f"inference_outputs_{process_with_tag}.npz",
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

    # v3.0 split-learning configuration. Toggle the master switch to swap
    # the entire training path; the inner knobs only matter when split is
    # active.
    split_learning   = True,
    embed_dim        = 16,
    encoder_hidden   = 64,
    nodes_per_graph  = 6,
    pmos_offset      = 4,

    # DP controls (used when PRIVACY_MODE=both in split backend).
    dp_enabled       = False,
    dp_noise_multiplier = float(os.environ.get('DP_NOISE_MULTIPLIER', '0.6')),
    dp_max_grad_norm = float(os.environ.get('DP_MAX_GRAD_NORM', '1.0')),
)

# Env-var override so the same sbatch can run either path without
# modifying source.
_env_split = os.environ.get('SPLIT_LEARNING')
if _env_split is not None:
    config.split_learning = _env_split not in ('0', '', 'false', 'False')


def main(config):

    run_tag = os.environ.get('RUN_TAG', '').strip()
    run_suffix = f"_{run_tag}" if run_tag else ""
    process_name = _get_process_name_from_env()
    privacy_mode = os.environ.get('PRIVACY_MODE', 'sl').strip().lower() or 'sl'

    if privacy_mode == 'both':
        config.dp_enabled = True
        print(f"[sweep] PRIVACY_MODE=both: enabling split-learning + DP (sigma={config.dp_noise_multiplier}, C={config.dp_max_grad_norm})", flush=True)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(project_root, 'results')
    privacy_dir = os.path.join(project_root, 'privacy')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(privacy_dir, exist_ok=True)
    training_started = time.perf_counter()
    data_prep_started = time.perf_counter()

    run = wandb.init(

        entity="yosemitesamurai",
        project="CurrentPrediction",
        config=vars(config),

    )

    mode = "split" if config.split_learning else "monolithic"
    run.name = f"{process_name}_{run_tag}" if run_tag else process_name
    print(f"Starting run: {run.name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"split_learning: {config.split_learning}", flush=True)

    df_to_use = data_frame_public if config.split_learning else data_frame

    train_df, test_df = train_test_split(

        df_to_use,
        test_size=config.test_size,
        random_state=42,
        shuffle=True)

    dataset_path = _resolve_dataset_path(project_root)
    _save_train_ids(dataset_path, process_name, run_suffix, config.test_size, privacy_dir)

    data_prep_seconds = time.perf_counter() - data_prep_started

    train_dataset = circuit_dataset(train_df, config)
    test_dataset = circuit_dataset(test_df, config)
    print(f"[sweep] datasets created: {len(train_dataset)} train, "
          f"{len(test_dataset)} test", flush=True)

    trainloader = DataLoader(

        train_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    testloader = DataLoader(

        test_dataset,
        batch_size=config.batch_size,
        shuffle=False)

    # Sample a row to determine input dimensionality. For split mode this
    # is the public-feature dim including zero-padded embedding slots; the
    # GAT then consumes the same shape after assemble_block_2inv fills in
    # the M1 / M2 embedding slots from the foundry.
    sample_X = train_dataset[0][1]
    embedding_dim = sample_X.shape[1]
    print(f"[sweep] node feature dim: {embedding_dim}", flush=True)

    gan = GAN(embedding_dim, config.hidden_dim, embedding_dim,
              config.layers, heads=config.heads)
    gan.to(device)
    print(f"[sweep] model initialized, starting training...", flush=True)
    optimizer = torch.optim.Adam(params=gan.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)
    epoch_times = []

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
        print(f"[sweep] foundry initialised: {len(process_table)} "
              f"process(es), embed_dim={config.embed_dim}", flush=True)

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()

        gan, optimizer, trainloss = train(
            gan, optimizer, trainloader, config, foundry=foundry)
        testloss, testMRE, maxRE, minRE = test(
            gan, testloader, config, foundry=foundry)
        scheduler.step(testloss)
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
        epoch_seconds = time.perf_counter() - epoch_started
        print(f"Epoch time: {epoch_seconds:.2f}s", flush=True)
        epoch_times.append(epoch_seconds)

    training_loop_seconds = time.perf_counter() - training_started
    checkpoint_started = time.perf_counter()

    # Common design-house payload. For v3.0 this is the strict side of the
    # cut: model weights + public_scaler only -- no global scaler (which
    # would carry BSIM4 means/stds), no foundry payload.
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
        # v2.0: the original global scaler is the only one used at
        # inference time.
        design_house_ckpt["scaler"] = scaler

    if config.split_learning:
        # In-process convenience: bundle the foundry payload alongside the
        # design-house payload as `model_split.pt`. In a real two-party
        # deployment, the foundry portion would be persisted under
        # foundry control and `model_split.pt` would be design-house-only.
        bundled = dict(design_house_ckpt)
        if foundry is not None:
            bundled["foundry_state_dict"] = foundry.state_dict()

        bundled_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
        torch.save(bundled, bundled_path)
        print(f"Bundled (design-house + foundry) checkpoint saved to "
              f"{bundled_path}", flush=True)
        shutil.copy2(bundled_path, os.path.join(privacy_dir, os.path.basename(bundled_path)))
        print(f"[sweep] Copied bundled checkpoint to privacy/", flush=True)

        # Strict design-house-only checkpoint, identical to the bundled
        # one minus foundry_state_dict. This is the artifact a real
        # design-house deployment would receive; saving it here lets the
        # user verify the cut is real.
        dh_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}_design_house.pt")
        torch.save(design_house_ckpt, dh_path)
        print(f"Strict design-house checkpoint saved to {dh_path}",
              flush=True)
        shutil.copy2(dh_path, os.path.join(privacy_dir, os.path.basename(dh_path)))
        print(f"[sweep] Copied design-house checkpoint to privacy/", flush=True)

        if foundry is not None:
            foundry_path = os.path.join(results_dir,
                                        f"model_{process_name}{run_suffix}_foundry.pt")
            torch.save(foundry.state_dict(), foundry_path)
            print(f"Foundry-side checkpoint saved to {foundry_path}",
                  flush=True)
            shutil.copy2(foundry_path, os.path.join(privacy_dir, os.path.basename(foundry_path)))
            print(f"[sweep] Copied foundry checkpoint to privacy/", flush=True)
    else:
        v2_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
        torch.save(design_house_ckpt, v2_path)
        print(f"Model saved to {v2_path}", flush=True)
        shutil.copy2(v2_path, os.path.join(privacy_dir, os.path.basename(v2_path)))
        print(f"[sweep] Copied model checkpoint to privacy/", flush=True)

    checkpoint_seconds = time.perf_counter() - checkpoint_started

    metadata = {
        "process_name": process_name,
        "run_tag": run_tag,
        "privacy_mode": privacy_mode,
        "split_learning": bool(config.split_learning),
        "dp_enabled": bool(config.dp_enabled),
        "dp_noise_multiplier": float(config.dp_noise_multiplier),
        "dp_max_grad_norm": float(config.dp_max_grad_norm),
        "data_prep_seconds": data_prep_seconds,
        "training_seconds": training_loop_seconds,
        "epoch_seconds": epoch_times,
        "checkpoint_seconds": checkpoint_seconds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "training_samples_per_second": (len(train_dataset) * config.epochs) / max(training_loop_seconds, 1e-12),
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "heads": config.heads,
    }
    metadata_path = os.path.join(privacy_dir, f"run_metadata_{process_name}{run_suffix}.json")
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"[sweep] Saved run metadata to {metadata_path}", flush=True)

    primary_model_path = os.path.join(results_dir, f"model_{process_name}{run_suffix}.pt")
    post_training_times = _run_inference_and_attacks(
        project_root=project_root,
        process_name=process_name,
        run_suffix=run_suffix,
        dataset_path=dataset_path,
        model_path=primary_model_path,
        privacy_dir=privacy_dir,
    )
    metadata.update(post_training_times)
    metadata["end_to_end_seconds"] = (
        metadata["data_prep_seconds"]
        + metadata["training_seconds"]
        + metadata["checkpoint_seconds"]
        + metadata["post_training_seconds"]
    )
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"[sweep] Updated run metadata with post-training timing at {metadata_path}", flush=True)

    run.finish()

if __name__ == '__main__':
    main(config)
