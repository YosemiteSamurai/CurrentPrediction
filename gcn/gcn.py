import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, extra_layers, fc=False):
        super(GCN, self).__init__()
        self.conv0 = GCNConv(in_channels, hidden_channels)
        self.convF = GCNConv(hidden_channels, out_channels) #if not fc else GCNConv(hidden_channels, hidden_channels)

        extra = []
        for i in range(extra_layers):
            extra.append(GCNConv(hidden_channels, hidden_channels))

        self.extra = nn.ModuleList(extra)

        # if fc:


    def encode(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.relu(x)
        for layer in self.extra:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.convF(x, edge_index)
        return x
 
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
 
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()