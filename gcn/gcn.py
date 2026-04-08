# =============================================================================
# gcn.py -- GAT Model Architecture
#
# Defines the GCN class using PyTorch Geometric's GATv2Conv layers.
# GATv2Conv learns a separate attention weight per neighbor per node, making
# it significantly more expressive than GCNConv for heterogeneous graphs like
# circuits where node roles (VDD, transistor, GND) differ greatly.
#
# encode(x, edge_index): Passes node features through a GATv2Conv input layer,
#   ELU activations, N optional hidden GAT layers, and a final GAT layer to
#   produce node embeddings. Hidden layers use multi-head attention with heads
#   averaged; the output layer concatenates heads to produce richer embeddings.
#
# decode(z, edge_label_index): Predicts each edge's current value using a
#   two-layer MLP over the concatenation of the two endpoint embeddings along
#   with their element-wise difference and product. This gives the decoder
#   explicit signal about the directional relationship between endpoints.
#
# Depth is controlled by extra_layers; heads controls multi-head attention.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, extra_layers, heads=4):
        super(GCN, self).__init__()

        # Input layer: heads averaged so output is hidden_channels
        self.conv0 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False)

        # Hidden layers: heads averaged so shape stays hidden_channels
        extra = []
        for _ in range(extra_layers):
            extra.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False))
        self.extra = nn.ModuleList(extra)

        # Output layer: heads concatenated -> embedding_dim = out_channels * heads
        self.convF = GATv2Conv(hidden_channels, out_channels, heads=heads, concat=True)
        embedding_dim = out_channels * heads

        # MLP decoder: takes [z_u || z_v || z_u - z_v || z_u * z_v] -> scalar
        mlp_input_dim = embedding_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.elu(x)
        for layer in self.extra:
            x = layer(x, edge_index)
            x = F.elu(x)
        x = self.convF(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        z_u = z[edge_label_index[0]]
        z_v = z[edge_label_index[1]]
        edge_features = torch.cat([z_u, z_v, z_u - z_v, z_u * z_v], dim=-1)
        return self.mlp(edge_features).squeeze(-1)