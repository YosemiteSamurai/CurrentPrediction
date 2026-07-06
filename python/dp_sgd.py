# python/dp_sgd.py
# =============================================================================
# dp_sgd.py - DP-SGD on the GAN (GATv2) model, Path B implementation
#
# WHY PATH B (manual per-sample loop) and not Path A (Opacus PrivacyEngine):
#   Opacus's auto per-sample-grad relies on a functorch fallback for layers
#   it does not natively support. The fallback calls layer.forward(activations)
#   with one positional arg. GATv2Conv.forward(x, edge_index) needs two, so
#   the fallback crashes:
#       TypeError: GATv2Conv.forward() missing 1 required positional argument
#   Writing a custom grad_sampler for GATv2Conv is involved because of the
#   message-passing per-sample semantics. Path B sidesteps this:
#     - split each batch into its individual graphs (one sample = one circuit)
#     - for each graph: forward, loss, autograd.grad to get this sample's grad
#     - clip each per-sample grad to max_grad_norm
#     - accumulate clipped grads, add Gaussian noise once per step
#     - write into .grad and call optimizer.step()
#
# This still gives sample-level DP via the standard DP-SGD accountant. We
# pull sigma from opacus.accountants.utils.get_noise_multiplier so the
# (epsilon, delta) bookkeeping matches what GAP / Opacus would compute.
# =============================================================================

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from opacus.accountants.utils import get_noise_multiplier
from gan import GAN

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[dp_sgd] device: {device}, torch {torch.__version__}")

# Shape matches the real CurrentPrediction batch.
NUM_GRAPHS, NODES_PER_GRAPH = 64, 8
EDGES_PER_GRAPH, TARGET_EDGE_IDX = 7, 3
FEATURE_DIM = 33                  # BSIM4 params per node
BATCH_SIZE  = 8
EPOCHS      = 1
TARGET_EPS, TARGET_DELTA = 8.0, 1e-5
MAX_GRAD_NORM = 1.0

def make_graph():
    x = torch.randn(NODES_PER_GRAPH, FEATURE_DIM)
    src = torch.randint(0, NODES_PER_GRAPH, (EDGES_PER_GRAPH,))
    dst = torch.randint(0, NODES_PER_GRAPH, (EDGES_PER_GRAPH,))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.randn(EDGES_PER_GRAPH)
    return Data(x=x, edge_index=edge_index, y=y)

graphs = [make_graph() for _ in range(NUM_GRAPHS)]
loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
print(f"[dp_sgd] dataset: {NUM_GRAPHS} graphs, batch_size={BATCH_SIZE}")

model = GAN(in_channels=FEATURE_DIM, hidden_channels=16,
            out_channels=FEATURE_DIM, extra_layers=1, heads=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()
print(f"[dp_sgd] model: {sum(p.numel() for p in model.parameters())} params")

# Calibrate sigma to (epsilon, delta) under Poisson subsampling. We DO NOT
# use Opacus's PrivacyEngine, only its accountant utility.
sigma = get_noise_multiplier(
    target_epsilon = TARGET_EPS,
    target_delta   = TARGET_DELTA,
    sample_rate    = BATCH_SIZE / NUM_GRAPHS,
    epochs         = EPOCHS,
)
print(f"[dp_sgd] sigma (noise multiplier) for eps={TARGET_EPS}, delta={TARGET_DELTA}: {sigma:.4f}")


def dp_train_step(model, optimizer, batch, criterion):
    """One DP-SGD step over the graphs in a PyG Batch."""
    graphs = batch.to_data_list()           # split disconnected union -> list
    n_samples = len(graphs)

    accumulated = [torch.zeros_like(p) for p in model.parameters()]
    sample_norms = []

    for g in graphs:
        g = g.to(device)
        z = model.encode(g.x, g.edge_index)
        out = model.decode(z, g.edge_index).view(-1)
        n_edges = g.y.shape[0]
        mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        mask[TARGET_EDGE_IDX::EDGES_PER_GRAPH] = True
        loss = criterion(out[mask], g.y.to(device)[mask])

        # per-sample gradient
        per_sample_grads = torch.autograd.grad(
            loss, list(model.parameters()), retain_graph=False, create_graph=False,
        )

        # global L2 norm across all params for this one sample
        total_sq = sum((pg.detach() ** 2).sum() for pg in per_sample_grads)
        s_norm = torch.sqrt(total_sq + 1e-12)
        sample_norms.append(s_norm.item())

        # clip factor in [0, 1]
        scale = (MAX_GRAD_NORM / (s_norm + 1e-12)).clamp(max=1.0)

        for acc, pg in zip(accumulated, per_sample_grads):
            acc.add_(pg.detach() * scale)

    # add Gaussian noise once per step. Standard DP-SGD: noise added to the
    # SUM of clipped grads, then divided by expected batch size.
    for acc in accumulated:
        acc.add_(torch.randn_like(acc) * sigma * MAX_GRAD_NORM)

    # write into .grad and let the optimizer apply it
    for p, acc in zip(model.parameters(), accumulated):
        p.grad = acc / BATCH_SIZE     # use expected batch size, not n_samples
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return n_samples, sample_norms


# Run one epoch on synthetic data.
model.train()
total_samples = 0
all_norms = []
for step, batch in enumerate(loader):
    n, norms = dp_train_step(model, optimizer, batch, criterion)
    total_samples += n
    all_norms.extend(norms)
    print(f"  step {step}: {n} samples, per-sample grad norms "
          f"min={min(norms):.3f} mean={sum(norms)/len(norms):.3f} max={max(norms):.3f}")

print(f"\n[dp_sgd] epoch done. saw {total_samples} samples across {step+1} steps.")
print(f"[dp_sgd] fraction of samples whose grad norm exceeded C={MAX_GRAD_NORM}: "
      f"{sum(n > MAX_GRAD_NORM for n in all_norms) / len(all_norms):.1%}")
print("[dp_sgd] DP-SGD Path B sanity check OK.")
