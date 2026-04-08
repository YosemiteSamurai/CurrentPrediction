# =============================================================================
# models.py -- Circuit-to-Graph Encoders
#
# Translates raw SPICE simulation data rows into (edges, X) graph tensors
# for specific circuit topologies. Currently implements two variants for the
# 2-inverter chain (2inv) design:
#
# block_2inv (6 nodes): Coarse-grained model grouping PMOS+NMOS of each
#   inverter into a single transistor node (M1, M2). Topology:
#   Vi -> M1 -> M2 -> Vo, with VDD and GND rails.
#
# split_2inv (8 nodes): Fine-grained model separating each inverter into
#   individual P/N transistor nodes (P1/N1, P2/N2), with extra edges to
#   handle duplicated currents at junctions.
#
# Both models build node feature matrix X containing:
#   - Transistor dimensions: W, L per device
#   - SPICE model parameters: 15 PMOS params (VTH0, TOX, U0, ...) and
#     18 NMOS params
#   - Global conditions: temperature, process skew (L/R), technology size,
#     device option (LP/HP/bulk)
#
# Edge weights are the simulated branch currents (I_vdd, I_in, I_out,
# I_gnd, I_target).
# =============================================================================

import numpy as np
import torch

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

    # Build a sparse adjacency matrix
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
    size = data['Size']
    option = data['Option']
    general = [temp, skewl, skewr, size, option]
    
    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']

    NUM_SPECIFIC = 4 + len(pFields) + len(nFields)
    x = [0]*NUM_SPECIFIC + general
    X = np.repeat([x], NUM_NODES, axis=0)

    # Dimensions
    X[M1, 0] = data['WP1']
    X[M1, 1] = data['WN1']
    X[M1, 2] = data['L1']

    X[M2, 0] = data['WP2']
    X[M2, 1] = data['WN2']
    X[M2, 2] = data['L2']

    # VDD
    X[VDD, 3] = data['VDD']

    # Other transistor factors
    for i in range(len(pFields)):
        X[M1, 3 + i] = data[pFields[i] + '_P']
        X[M2, 3 + i] = data[pFields[i] + '_P']

    for i in range(len(nFields)):
        X[M1, 3 + len(pFields) + i] = data[nFields[i] + '_N']
        X[M2, 3 + len(pFields) + i] = data[nFields[i] + '_N']

    edges = torch.tensor(edges).to(torch.float)
    # print(len(X[1]))
    X = torch.tensor(X).to(torch.float)

    return edges, X#, targets

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

    # Build a sparse adjacency matrix
    I_t = data['I_target']
    I_in = data['I_in']
    I_out = data['I_out']
    I_vdd = data['I_vdd']
    I_gnd = data['I_gnd']

    edges =[(VDD, P1, I_vdd),
            (VDD, P2, I_vdd), # duplicated over junction
            (N1, GND, I_gnd),
            (N2, GND, I_gnd), # duplicated over junction
            (Vi, P1, I_in),
            (Vi, N1, I_in), # duplicated over junction
            (P2, Vo, I_out),
            (N2, Vo, I_out),
            (P1, P2, I_t),
            (P1, N2, I_t), # duplicated over junction
            (N1, P2, I_t), # duplicated over junction
            (N1, N2, I_t), # duplicated over junction
            ]

    # Build a feature matrix
    # General parameters
    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = data['Size']
    option = data['Option']
    general = [temp, skewl, skewr, size, option]

    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']

    NUM_SPECIFIC = 3 + max(len(pFields), len(nFields))
    x = [0]*NUM_SPECIFIC + general
    X = np.repeat([x], NUM_NODES, axis=0)

    # Dimensions
    X[P1, 0] = data['WP1']
    X[P1, 1] = data['L1']

    X[N1, 0] = data['WN1']
    X[N1, 1] = data['L1']

    X[P2, 0] = data['WP2']
    X[P2, 1] = data['L2']

    X[N2, 0] = data['WN2']
    X[N2, 1] = data['L2']

    # Other transistor factors

    for i in range(len(pFields)):

        X[P1, 3 + i] = data[pFields[i] + '_P']
        X[P2, 3 + i] = data[pFields[i] + '_P']

    for i in range(len(nFields)):
        
        X[N1, 3 + i] = data[nFields[i] + '_N']
        X[N2, 3 + i] = data[nFields[i] + '_N']

    # VDD
    X[VDD, 2] = data['VDD']

    edges = torch.tensor(edges).to(torch.float)
    X = torch.tensor(X.astype(float)).to(torch.float)

    return edges, X
