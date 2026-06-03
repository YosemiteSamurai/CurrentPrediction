# python/dp_sgd.py
# =============================================================================
# dp_sgd.py - DP-SGD for the GATv2 (GAN) model, Path B (manual per-sample loop)
#
# HOW THIS INTEGRATES
#   dp_train() has the SAME signature as dataset.train(gcn, optimizer,
#   trainloader, config). It is a drop-in replacement: in the training loop,
#   call dp_train(...) instead of train(...) when DP is enabled. It reuses the
#   exact same per-sample graph construction (graph.Graph) and the same
#   encode/decode/target-edge-mask flow as dataset.train(), so the model and
#   data path are unchanged. The only difference is per-sample gradient
#   clipping + Gaussian noise (DP-SGD).
#
#   Required on config:
#     batch_size, edges_per_graph, target_edge_idx  (already present)
#     dp_epsilon, dp_delta, max_grad_norm            (new DP knobs)
#     sigma                                          (set once via compute_sigma)
#
# WHY PATH B (not Opacus PrivacyEngine):
#   Opacus's per-sample-grad uses a functorch fallback that calls
#   layer.forward(x) with one positional arg; GATv2Conv.forward(x, edge_index)
#   needs two, so the fallback crashes. Path B sidesteps that: split the batch
#   into individual circuits, get each sample's grad via autograd.grad, clip,
#   accumulate, add noise once, then optimizer.step(). We still use Opacus's
#   accountant (get_noise_multiplier) so the (epsilon, delta) bookkeeping
#   matches what Opacus / GAP would compute.
# =============================================================================

import torch
import torch.nn as nn
from opacus.accountants.utils import get_noise_multiplier
from graph import Graph

criterion = nn.L1Loss()


def compute_sigma(config, n_train):
    """Noise multiplier calibrated to (dp_epsilon, dp_delta) under Poisson
    subsampling. Call once before training and store as config.sigma."""
    return get_noise_multiplier(
        target_epsilon=config.dp_epsilon,
        target_delta=config.dp_delta,
        sample_rate=config.batch_size / n_train,
        epochs=config.epochs,
    )


def dp_train(gcn, optimizer, trainloader, config):
    """One epoch of DP-SGD. Drop-in for dataset.train().

    Returns (gcn, optimizer, avg_loss). Also returns clipping stats on
    config._last_clip_frac for quick diagnostics."""
    device = next(gcn.parameters()).device
    sigma = config.sigma
    C = config.max_grad_norm
    params = list(gcn.parameters())

    gcn.train()
    total_loss = 0.0
    batches = 0
    norms_over_C = 0
    norms_seen = 0

    for batch in trainloader:
        # default_collate stacks the per-sample (edges, X) tuples, exactly like
        # graph.batch_graph reads them: batch[0][i] = sample i edges, [1][i] = X.
        n_in_batch = len(batch[0])
        accumulated = [torch.zeros_like(p) for p in params]
        batch_loss = 0.0

        for i in range(n_in_batch):
            graph = Graph(batch[0][i], batch[1][i], config)
            A = graph.A.to(device)
            X = graph.X.to(device)
            y = graph.y.to(device)

            z = gcn.encode(X, A)
            out = gcn.decode(z, A).view(-1)
            n_edges = y.shape[0]
            mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
            mask[config.target_edge_idx::config.edges_per_graph] = True
            loss = criterion(out[mask], y[mask])
            batch_loss += loss.item()

            # this one sample's gradient w.r.t. all params
            per_sample_grads = torch.autograd.grad(
                loss, params, retain_graph=False, create_graph=False)

            # global L2 norm across all params, then clip to C
            total_sq = sum((g.detach() ** 2).sum() for g in per_sample_grads)
            s_norm = torch.sqrt(total_sq + 1e-12)
            norms_seen += 1
            if s_norm.item() > C:
                norms_over_C += 1
            scale = (C / (s_norm + 1e-12)).clamp(max=1.0)

            for acc, g in zip(accumulated, per_sample_grads):
                acc.add_(g.detach() * scale)

        # add Gaussian noise once to the SUM of clipped grads, then divide by
        # the EXPECTED batch size (config.batch_size), not n_in_batch.
        for acc in accumulated:
            acc.add_(torch.randn_like(acc) * sigma * C)
        for p, acc in zip(params, accumulated):
            p.grad = acc / config.batch_size
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += batch_loss / n_in_batch
        batches += 1

    config._last_clip_frac = norms_over_C / max(norms_seen, 1)
    return gcn, optimizer, total_loss / batches
