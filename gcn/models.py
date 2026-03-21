import numpy as np
import torch

# All models are currently assuming the 2inv design

# 2inv design with a block modelling paradigm
# Transistors are blocked into one node
# TODO - currently, junctions duplicate current, which breaks conservation, but this simplified version might be solvable
  # this would require junction nodes, though
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
    # targets = []

    # Build a feature matrix
    # General parameters
    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = data['Size']
    option = data['Option']
    general = [temp, skewl, skewr, size, option]
    
    # Node-specific parameters
    # pFields = data.filter(regex=("*_P")).columns.to_list()
    # nFields = data.filter(regex=("*_N")).columns.to_list()

    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']


    # # Node-specific parameters
    # NUM_SPECIFIC = 4
    # x = [0]*NUM_SPECIFIC + general
    # X = np.repeat([x], NUM_NODES, axis=0)

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

    # print(f"edges: {edges.shape}\n{edges}")
    # print(f"features: {X.shape}\n{X}")

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
      "GND": 7}

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
    # targets = []

    # Build a feature matrix
    # General parameters
    temp = data['Temp']
    skewl = data['SkewL']
    skewr = data['SkewR']
    size = data['Size']
    option = data['Option']
    general = [temp, skewl, skewr, size, option]

    # Node-specific parameters
    # pFields = data.filter(regex=("*_P")).columns.to_list()
    # nFields = data.filter(regex=("*_N")).columns.to_list()

    pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
               'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
               'CGDO']
    nFields = pFields + ['PCLM', 'K2', 'DVT2']

    # # Cut fields that have a smaller std than the transistor dimensions
    # pFields = ['U0', 'VSAT', 'NDEP', 'NF', 'ETA0', 'RDSW']
    # nFields = pFields + ['VTH0', 'PCLM', 'K2']

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


# def joint_2inv(data):
    # NODE_MAP = {
    #   "Vi": 0,
    #   "M1": 1,
    #   "VDD": 2,
    #   "M2": 3,
    #   "Vo": 4,
    #   "GND": 5,
    #   "J1": 6,
    #   "J2": 7}
    # NUM_NODES = len(NODE_MAP)

    # Vi = NODE_MAP["Vi"]
    # M1 = NODE_MAP["M1"]
    # VDD = NODE_MAP["VDD"]
    # M2 = NODE_MAP["M2"]
    # Vo = NODE_MAP["Vo"]
    # GND = NODE_MAP["GND"]
    # JP = data["J1"]
    # JN = data["J2"]

    # # Build a sparse adjacency matrix
    # I_t = data['I_target']
    # I_in = data['I_in']
    # I_out = data['I_out']
    # I_vdd = data['I_vdd']
    # I_gnd = data['I_gnd']

    # edges = [(VDD, JP, I_vdd),
    #          (JN, GND, I_gnd)
    #          (Vi, M1, I_in),
    #          (M2, Vo, I_out),
    #          (M1, M2, I_t),
    #          (JP, M1, -1),
    #          (JP, M2, -1),
    #          (M1, JN, -1),
    #          (M2, JN, -1),             
    #         ]

    # # Build a feature matrix
    # # General parameters
    # general = [data['Temp']]
    # skewl = data['SkewL']
    # skewr = data['SkewR']
    # size = data['Size']
    # option = data['Option']
    # general += [skewl] + [skewr] + [size] + [option]

    # # Node-specific parameters
    # # pFields = data.filter(regex=("*_P")).columns.to_list()
    # # nFields = data.filter(regex=("*_N")).columns.to_list()

    # pFields = ['VTH0', 'TOX', 'TOXP', 'TOXM', 'U0', 'UC', 'VSAT',
    #            'XJ', 'NDEP', 'NF', 'ETA0', 'VOFF', 'RDSW', 'CGSO',
    #            'CGDO']
    # nFields = pFields + ['PCLM', 'K2', 'DVT2']

    # # # Cut fields that have a smaller std than the transistor dimensions
    # # pFields = ['U0', 'VSAT', 'NDEP', 'NF', 'ETA0', 'RDSW']
    # # nFields = pFields + ['VTH0', 'PCLM', 'K2']

    # NUM_SPECIFIC = 3 + max(len(pFields), len(nFields))
    # x = [0]*NUM_SPECIFIC + general
    # X = np.repeat([x], NUM_NODES, axis=0)

    # # Dimensions
    # X[P1, 0] = data['WP1']
    # X[P1, 1] = data['L1']

    # X[N1, 0] = data['WN1']
    # X[N1, 1] = data['L1']

    # X[P2, 0] = data['WP2']
    # X[P2, 1] = data['L2']

    # X[N2, 0] = data['WN2']
    # X[N2, 1] = data['L2']

    # # Other transistor factors

    # for i in range(len(pFields)):
    #     X[P1, 3 + i] = data[pFields[i] + '_P']
    #     X[P2, 3 + i] = data[pFields[i] + '_P']

    # for i in range(len(nFields)):
    #     X[N1, 3 + i] = data[nFields[i] + '_N']
    #     X[N2, 3 + i] = data[nFields[i] + '_N']

    # # VDD
    # X[VDD, 2] = data['VDD']

    # edges = torch.tensor(edges).to(torch.float)
    # X = torch.tensor(X.astype(float)).to(torch.float)

    # return edges, X

