# TODO - go through torch_geometric docs
#   and determine if any of this work is redundant with library functions

import torch

class Graph():
    def __init__(self, edges, features, config):
        # Implement as a mask?
        # self.target_edges = models[config.model]['target']
        # self.target_labels = models[config.model]['target']

        self.A = [[], []]
        self.A_blank = [[], []]
        self.y = []

        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])

        self.A = torch.tensor(self.A)
        self.y = torch.tensor(self.y)
        self.X = features

        # print(f"A: {self.A.shape}\n{self.A}")
        # print(f"y: {self.y.shape}\n{self.y}")
        # print(f"X: {self.X.shape}\n{self.X}")

        self.num_nodes = self.X.shape[0]

        # if config.log_current:
        #     self.y = torch.log(self.y)
        #     self.y = torch.abs(self.y)

    def add_edge(self, start, end, weight):
        self.A[0].append(int(start))
        self.A[1].append(int(end))
        self.y.append(weight)

    def merge(self, graph):
        # print(self.A.shape)
        # print(f"X: {self.X.shape}")
        self.A = torch.cat((self.A, graph.A + self.num_nodes), dim=1)

        # print(f"A[0]: {self.A[0].shape}\n{self.A[0]}")
        # print(f"A[1]: {self.A[1].shape}\n{self.A[1]}")

        self.y = torch.cat((self.y, graph.y), dim=0)

        self.X = torch.cat((self.X, graph.X), dim=0)
        # print(self.A.shape)
        # print(f"X: {self.X.shape}")

        self.num_nodes += graph.num_nodes

def batch_graph(batch, config):
    graphs = []
    # print(len(batch))
    # print(f"batch[0]: {batch[0].shape}\n{batch[0]}")
    # print(f"batch[1]: {batch[1].shape)\n{batch[1]}")
    for i in range(len(batch[0])):
        edges = batch[0][i]
        features = batch[1][i]
        graphs.append(Graph(edges, features, config))

    batch_graph = graphs[0]
    for graph in graphs[1:]:
        batch_graph.merge(graph)

    return batch_graph