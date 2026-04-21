# =============================================================================
# graph.py -- Graph Data Structure
#
# Provides a Graph class that holds the circuit graph representation:
#   - A: COO-format sparse edge index (2xE tensor of source/dest node indices)
#   - y: Edge weight labels (simulated branch current values)
#   - X: Node feature matrix
#
# Key methods:
#   - add_edge: Appends a directed edge with its current label.
#   - merge(graph): Combines two graphs into one by offsetting the second
#     graph's node indices by self.num_nodes, enabling batch training.
#   - batch_graph(batch, config): Converts a raw DataLoader batch into a single
#     merged Graph object -- one graph per sample, all concatenated for
#     parallel GPU processing.
# =============================================================================

import torch

class Graph():

    def __init__(self, edges, features, config):

        self.A = [[], []]
        self.A_blank = [[], []]
        self.y = []

        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])

        self.A = torch.tensor(self.A)
        self.y = torch.tensor(self.y)
        self.X = features

        self.num_nodes = self.X.shape[0]

    def add_edge(self, start, end, weight):

        self.A[0].append(int(start))
        self.A[1].append(int(end))
        self.y.append(weight)

    def merge(self, graph):

        self.A = torch.cat((self.A, graph.A + self.num_nodes), dim=1)
        self.y = torch.cat((self.y, graph.y), dim=0)
        self.X = torch.cat((self.X, graph.X), dim=0)
        self.num_nodes += graph.num_nodes

def batch_graph(batch, config):

    graphs = []

    for i in range(len(batch[0])):
        edges = batch[0][i]
        features = batch[1][i]
        graphs.append(Graph(edges, features, config))

    batch_graph = graphs[0]
    for graph in graphs[1:]:
        batch_graph.merge(graph)

    return batch_graph
