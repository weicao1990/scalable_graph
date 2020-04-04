import math
import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()

    def forward(self, X, edge_index, edge_weight, data_flow):
        pass


class MySAGEConv(PyG.SAGEConv):
    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super(MySAGEConv, self).__init__(in_channels, out_channels,
                                         normalize=normalize, concat=concat, bias=bias, **kwargs)
    def message(self, x_j, edge_weight):
        print('x_j', x_j.size())
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out, x, res_n_id):
        # TODO: this triggers a CUDA error
        # print(aggr_out.size())
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            # TODO: to check the consistency
            # print((x[0][res_n_id] - aggr_out).abs().sum())
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

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

    def forward(self, X, g, pretrain=False):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']
        
        # swap node to dim 0
        X = X.permute(1, 0, 2)

        loop_index_list = [
            res_n_id[i].unsqueeze(0).repeat(2, 1).to(device=edge_index[0].device) for i in range(2)
        ]

        if pretrain:
            conv1 = self.conv1(
                (X, None), loop_index_list[0], edge_weight=None, size=size[0], res_n_id=res_n_id[0])
        else:
            conv1 = self.conv1(
                (X, None), edge_index[0], edge_weight=edge_weight[0], size=size[0], res_n_id=res_n_id[0])
        
        X = F.leaky_relu(conv1)

        if pretrain:
            conv2 = self.conv2(
                (X, None), loop_index_list[1], edge_weight=None, size=size[1], res_n_id=res_n_id[1])
        else:
            conv2 = self.conv2(
                (X, None), edge_index[1], edge_weight=edge_weight[1], size=size[1], res_n_id=res_n_id[1])


        X = F.leaky_relu(conv2)
        X = X.permute(1, 0, 2)
        return X
