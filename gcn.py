import math
import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class MyGATConv(PyG.GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(MyGATConv, self).__init__(in_channels, out_channels, heads=heads,
                                        concat=concat, negative_slope=negative_slope, dropout=dropout, bias=bias)
        self.att = nn.Parameter(torch.Tensor(1, 1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i, edge_weight):
        # Compute attention coefficients.
        x_j = x_j.view(x_j.size(0), x_j.size(1), self.heads, self.out_channels)

        x_i = x_i.view(x_i.size(0), x_i.size(1), self.heads, self.out_channels)
        edge_weight = edge_weight.view(-1, 1, 1)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = alpha * edge_weight

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geometric.utils.softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(x_j.size(0), x_j.size(1), self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(aggr_out.size(
                0), aggr_out.size(1), self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=-2)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        heads = 4
        self.conv1 = MyGATConv(in_channels=in_channels,
                               out_channels=16, heads=heads, concat=False)

        self.conv2 = MyGATConv(in_channels=16,
                               out_channels=out_channels, heads=heads, concat=False)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']

        # swap node to dim 0
        X = X.permute(1, 0, 2)

        conv1 = self.conv1(
            (X, X[res_n_id[0]]), edge_index[0], edge_weight=edge_weight[0], size=size[0])

        X = F.leaky_relu(conv1)

        conv2 = self.conv2(
            (X, X[res_n_id[1]]), edge_index[1], edge_weight=edge_weight[1], size=size[1])

        X = F.leaky_relu(conv2)

        X = X.permute(1, 0, 2)
        return X


class MySAGEConv(PyG.SAGEConv):
    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super(MySAGEConv, self).__init__(in_channels, out_channels,
                                         normalize=normalize, concat=concat, bias=bias, **kwargs)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out, x, res_n_id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class SAGENet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGENet, self).__init__()
        self.conv1 = MySAGEConv(
            in_channels, 16, normalize=False, concat=True)
        self.conv2 = MySAGEConv(
            16, out_channels, normalize=False, concat=True)

    def forward(self, X, g):
        # swap node to dim 0
        X = X.permute(1, 0, 2)

        # conv1 = self.conv1(X, g.edge_index, edge_weight=g.edge_attr * g.edge_norm)
        conv1 = self.conv1(X, g.edge_index, edge_weight=g.edge_attr)

        X = F.leaky_relu(conv1)

        # conv2 = self.conv2(X, g.edge_index, edge_weight=g.edge_attr * g.edge_norm)
        conv2 = self.conv2(X, g.edge_index, edge_weight=g.edge_attr)

        X = F.leaky_relu(conv2)
        X = X.permute(1, 0, 2)
        return X
