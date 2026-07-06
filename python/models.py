# =============================================================================
# models.py -- Circuit-to-Graph Encoders
#
# Translates raw SPICE simulation data rows into (edges, X) graph tensors
# for specific circuit topologies. Implements variants for the 2-inverter chain (2inv):
#   - block_2inv (6 nodes): Coarse-grained model grouping PMOS+NMOS of each inverter into a 
#     single transistor node (M1, M2). Topology: Vi -> M1 -> M2 -> Vo, with VDD and GND rails.
#   - split_2inv (8 nodes): Fine-grained model separating each inverter into individual P/N
#     transistor nodes (P1/N1, P2/N2), with extra edges to handle duplicated currents at junctions.
#
# Both models build node feature matrix X containing:
#   - Transistor dimensions: W, L per device
#   - SPICE model parameters: 15 PMOS params (VTH0, TOX, U0, ...) and 18 NMOS params
#   - Global conditions: temperature, process skew (L/R), technology size, device option (LP/HP/bulk)
#
# Edge weights are the simulated branch currents (I_vdd, I_in, I_out, I_gnd, I_target).
#
# v3.0 split-learning additions:
#   - block_2inv_public: same topology as block_2inv but the BSIM4 slots
#     are placeholder zeros. The foundry's encoder produces 16-dim PMOS
#     and NMOS embeddings; assemble_block_2inv writes them into the M1/M2
#     slots after batching. The design-house data has no raw BSIM4 values,
#     so the GAT consumes the embeddings instead.
#   - assemble_block_2inv: post-batch helper that stitches the encoder
#     embeddings into the merged graph's feature matrix at the M1 / M2
#     positions of every sample.
# =============================================================================

import numpy as np
import torch

# Default embedding dim assumed by block_2inv_public + assemble_block_2inv.
# Kept here as a constant so the layout (W/L/VDD slots, embedding slots,
# globals) stays consistent across the dataset / model / training code
# even when the encoder's actual embed_dim differs at run time.
DEFAULT_EMBED_DIM = 16
# Number of per-node W/L/VDD slots before the embedding region.
PMOS_OFFSET = 4
# Number of global condition slots appended after the embedding region.
NUM_GLOBAL_SLOTS = 5

def _f(data, key):
    # Support both pandas Series (data.index) and dict (data.keys())
    if hasattr(data, 'index'):
        exists = key in data.index
    else:
        exists = key in data
    return float(data[key]) if exists else 0.0

def block_2inv(data, design):

    NODE_MAP = {
      "Vi": 0,
      "M1": 1,
      "VDD": 2,
      "M2": 3,
      "Vo": 4,
      "GND": 5}

    NUM_NODES = len(NODE_MAP)

    Vi = NODE_MAP["Vi"]
    M1 = NODE_MAP["M1"]
    VDD = NODE_MAP["VDD"]
    M2 = NODE_MAP["M2"]
    Vo = NODE_MAP["Vo"]
    GND = NODE_MAP["GND"]

    I_t = data['I_target']
    I_in = data['I_in']
    I_out = data['I_out']
    I_vdd = data['I_vdd']
    I_gnd = data['I_gnd']

    edges = [(VDD, M1, I_vdd),
             (VDD, M2, I_vdd), # duplicated over junction
             (Vi, M1, I_in),
             (M1, M2, I_t),
             (M1, GND, I_gnd),
             (M2, GND, I_gnd), # duplicated over junction
             (M2, Vo, I_out)
            ]

    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = _f(data, 'Size')
    option = _f(data, 'Option')
    general = [temp, skewl, skewr, size, option]
    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']

    NUM_SPECIFIC = 4 + len(pFields) + len(nFields)
    x = [0]*NUM_SPECIFIC + general
    X = np.repeat([x], NUM_NODES, axis=0)

    X[M1, 0] = data['WP1']
    X[M1, 1] = data['WN1']
    X[M1, 2] = data['L1']
    X[M2, 0] = data['WP2']
    X[M2, 1] = data['WN2']
    X[M2, 2] = data['L2']
    X[VDD, 3] = data['VDD']

    for i in range(len(pFields)):

        X[M1, 3 + i] = _f(data, pFields[i] + '_P')
        X[M2, 3 + i] = _f(data, pFields[i] + '_P')

    for i in range(len(nFields)):

        X[M1, 3 + len(pFields) + i] = _f(data, nFields[i] + '_N')
        X[M2, 3 + len(pFields) + i] = _f(data, nFields[i] + '_N')

    edges = torch.tensor(edges).to(torch.float)
    X = torch.tensor(X).to(torch.float)

    return edges, X

def split_2inv(data, design):

    NODE_MAP = {
      "Vi": 0,
      "N1": 1,
      "P1": 2,
      "VDD": 3,
      "N2": 4,
      "P2": 5,
      "Vo": 6,
      "GND": 7
    }

    NUM_NODES = len(NODE_MAP)

    Vi = NODE_MAP["Vi"]
    N1 = NODE_MAP["N1"]
    P1 = NODE_MAP["P1"]
    VDD = NODE_MAP["VDD"]
    N2 = NODE_MAP["N2"]
    P2 = NODE_MAP["P2"]
    Vo = NODE_MAP["Vo"]
    GND = NODE_MAP["GND"]

    I_t = data['I_target']
    I_in = data['I_in']
    I_out = data['I_out']
    I_vdd = data['I_vdd']
    I_gnd = data['I_gnd']

    edges =[(VDD, P1, I_vdd),
            (VDD, P2, I_vdd),
            (N1, GND, I_gnd),
            (N2, GND, I_gnd),
            (Vi, P1, I_in),
            (Vi, N1, I_in),
            (P2, Vo, I_out),
            (N2, Vo, I_out),
            (P1, P2, I_t),
            (P1, N2, I_t),
            (N1, P2, I_t),
            (N1, N2, I_t),
            ]

    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = _f(data, 'Size')
    option = _f(data, 'Option')
    general = [temp, skewl, skewr, size, option]
    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']

    NUM_SPECIFIC = 3 + max(len(pFields), len(nFields))
    x = [0]*NUM_SPECIFIC + general
    X = np.repeat([x], NUM_NODES, axis=0)

    X[P1, 0] = data['WP1']
    X[P1, 1] = data['L1']
    X[N1, 0] = data['WN1']
    X[N1, 1] = data['L1']
    X[P2, 0] = data['WP2']
    X[P2, 1] = data['L2']
    X[N2, 0] = data['WN2']
    X[N2, 1] = data['L2']

    for i in range(len(pFields)):

        X[P1, 3 + i] = _f(data, pFields[i] + '_P')
        X[P2, 3 + i] = _f(data, pFields[i] + '_P')

    for i in range(len(nFields)):
        
        X[N1, 3 + i] = _f(data, nFields[i] + '_N')
        X[N2, 3 + i] = _f(data, nFields[i] + '_N')

    X[VDD, 2] = data['VDD']
    edges = torch.tensor(edges).to(torch.float)
    X = torch.tensor(X.astype(float)).to(torch.float)

    return edges, X


# =============================================================================
# v3.0 split-learning helpers -- block_2inv_public + assemble_block_2inv
#
# The public-feature builder mirrors block_2inv's topology but reserves
# zero-padded slots for the PMOS and NMOS embeddings instead of raw BSIM4
# values. Layout per node:
#
#     [WP, WN, L, VDD,                        # 4 per-node slots
#      pmos_embed_0 ... pmos_embed_(D-1),     # D zero-padded slots
#      nmos_embed_0 ... nmos_embed_(D-1),     # D zero-padded slots
#      Temp, SkewL, SkewR, Size, Option]      # 5 global slots
#
# After batching, models.assemble_block_2inv() writes the foundry's
# embeddings into the M1 (idx 1) and M2 (idx 3) slot ranges of every
# sample.
# =============================================================================


def block_2inv_public(data, design, embed_dim=DEFAULT_EMBED_DIM):

    NODE_MAP = {
      "Vi": 0,
      "M1": 1,
      "VDD": 2,
      "M2": 3,
      "Vo": 4,
      "GND": 5}

    NUM_NODES = len(NODE_MAP)

    Vi = NODE_MAP["Vi"]
    M1 = NODE_MAP["M1"]
    VDD = NODE_MAP["VDD"]
    M2 = NODE_MAP["M2"]
    Vo = NODE_MAP["Vo"]
    GND = NODE_MAP["GND"]

    I_t = data['I_target']
    I_in = data['I_in']
    I_out = data['I_out']
    I_vdd = data['I_vdd']
    I_gnd = data['I_gnd']

    edges = [(VDD, M1, I_vdd),
             (VDD, M2, I_vdd),
             (Vi, M1, I_in),
             (M1, M2, I_t),
             (M1, GND, I_gnd),
             (M2, GND, I_gnd),
             (M2, Vo, I_out)
            ]

    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = _f(data, 'Size')
    option = _f(data, 'Option')
    general = [temp, skewl, skewr, size, option]

    # Layout: 4 per-node slots + 2*embed_dim placeholder slots + 5 globals.
    public_specific = PMOS_OFFSET + 2 * embed_dim
    x = [0.0] * public_specific + general
    X = np.repeat([x], NUM_NODES, axis=0).astype(float)

    X[M1, 0] = data['WP1']
    X[M1, 1] = data['WN1']
    X[M1, 2] = data['L1']
    X[M2, 0] = data['WP2']
    X[M2, 1] = data['WN2']
    X[M2, 2] = data['L2']
    X[VDD, 3] = data['VDD']

    edges = torch.tensor(edges).to(torch.float)
    X = torch.tensor(X).to(torch.float)

    return edges, X


def assemble_block_2inv(X, z_pmos, z_nmos,
                        num_nodes_per_graph=6,
                        pmos_offset=PMOS_OFFSET,
                        m1_idx=1, m2_idx=3):
    """Stitch foundry embeddings into a merged batch's feature matrix.

    X       : merged node feature matrix produced by Graph.merge over a
              batch of block_2inv_public outputs. Shape:
              (batch_size * num_nodes_per_graph, feat_dim).
    z_pmos  : (batch_size, embed_dim) PMOS embeddings from the foundry.
    z_nmos  : (batch_size, embed_dim) NMOS embeddings from the foundry.

    Writes z_pmos[i] into the PMOS slot range of M1 and M2 for graph i,
    and z_nmos[i] into the NMOS slot range. Returns a NEW tensor that
    participates in autograd through z_pmos / z_nmos so loss.backward()
    populates their .grad fields.
    """

    embed_dim = z_pmos.shape[-1]
    if z_nmos.shape[-1] != embed_dim:
        raise ValueError(
            f"PMOS / NMOS embed_dim mismatch: {embed_dim} vs {z_nmos.shape[-1]}")

    pmos_slot = slice(pmos_offset, pmos_offset + embed_dim)
    nmos_slot = slice(pmos_offset + embed_dim, pmos_offset + 2 * embed_dim)

    batch_size = z_pmos.shape[0]

    # Clone so we don't mutate the merged graph's buffer in place. The
    # clone produces a non-leaf tensor; the indexed writes from
    # z_pmos / z_nmos make the result depend on those tensors so autograd
    # populates their gradients during loss.backward().
    X_out = X.clone()

    # Move embeddings to X's device if they aren't already there.
    if z_pmos.device != X_out.device:
        z_pmos = z_pmos.to(X_out.device)
        z_nmos = z_nmos.to(X_out.device)

    # Cast embeddings to X's dtype (X is float32 from torch.tensor(...).to(torch.float)).
    if z_pmos.dtype != X_out.dtype:
        z_pmos = z_pmos.to(X_out.dtype)
        z_nmos = z_nmos.to(X_out.dtype)

    # Build flat row indices for all M1 and M2 nodes across the batch and
    # do the writes in one shot per slot range. This is faster than a
    # per-sample Python loop and keeps the autograd graph compact.
    base = torch.arange(batch_size, device=X_out.device) * num_nodes_per_graph
    m1_rows = base + m1_idx
    m2_rows = base + m2_idx

    X_out[m1_rows, pmos_slot] = z_pmos
    X_out[m2_rows, pmos_slot] = z_pmos
    X_out[m1_rows, nmos_slot] = z_nmos
    X_out[m2_rows, nmos_slot] = z_nmos

    return X_out
