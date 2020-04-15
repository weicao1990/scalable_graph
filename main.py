import os
import time
import argparse
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader

from graph_saint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import EarlyStopping

from argparse import Namespace

from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from tgcn import TGCN
from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj


parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
parser.add_argument('--log-name', type=str, default='default',
                    help='Experiment name to log')
parser.add_argument('--log-dir', type=str, default='./logs',
                    help='Path to log dir')
parser.add_argument('--gpus', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('-m', "--model", choices=['tgcn', 'stgcn', 'gwnet'],
                    help='Choose Spatial-Temporal model', default='stgcn')
parser.add_argument('-d', "--dataset", choices=["metr", "nyc-bike"],
                    help='Choose dataset', default='metr')
parser.add_argument('-t', "--gcn-type", choices=['sage', 'graph', 'gat'],
                    help='Choose GCN Conv Type', default='gat')
parser.add_argument('-batchsize', type=int, default=32,
                    help='Training batch size')
parser.add_argument('-epochs', type=int, default=1000,
                    help='Training epochs')
parser.add_argument('-l', '--loss-criterion', choices=['mse', 'mae'],
                    help='Choose loss criterion', default='mse')
parser.add_argument('-num-timesteps-input', type=int, default=12,
                    help='Num of input timesteps')
parser.add_argument('-num-timesteps-output', type=int, default=3,
                    help='Num of output timesteps for forecasting')
parser.add_argument('-early-stop-rounds', type=int, default=30,
                    help='Earlystop rounds when validation loss does not decrease')

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

model = TGCN
log_name = args.log_name
log_dir = args.log_dir
gpus = args.gpus

loss_criterion = {'mse': nn.MSELoss(reduction='none'), 'mae': nn.L1Loss(reduction='none')}\
    .get(args.loss_criterion)
gcn_type = args.gcn_type
batch_size = args.batchsize
epochs = args.epochs
num_timesteps_input = args.num_timesteps_input
num_timesteps_output = args.num_timesteps_output
early_stop_rounds = args.early_stop_rounds


class SaintDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, shuffle):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.graph_sampler = self.make_graph_sampler()

        self.length = self.get_length()

    def make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
        ) .to('cpu')

        loader = GraphSAINTNodeSampler(
            graph, batch_size=10, walk_length=2, num_steps=5, sample_coverage=1000, num_workers=0, sequential_sample=not self.shuffle
        )

        return loader

    def __iter__(self):
        for data in self.graph_sampler:
            X, y = self.X[:, data.node_idx], self.y[:, data.node_idx]

            subset = TensorDataset(X, y)
            indices = np.arange(len(subset))

            if self.shuffle:
                np.random.shuffle(indices)

            num_batches = (
                len(subset) + self.batch_size - 1
            ) // self.batch_size

            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size

                yield X[indices[start: end]], y[indices[start: end]], data, indices[start: end]

    def get_length(self):
        length = 0
        for data in self.graph_sampler:
            X, y = self.X[:, data.node_idx], self.y[:, data.node_idx]
            length += (X.size(0) + self.batch_size - 1) // self.batch_size
        return length

    def __len__(self):
        return self.length


class WrapperNet(pl.LightningModule):
    # NOTE: pl module is supposed to only have ``hparams`` parameter
    def __init__(self, hparams):
        super(WrapperNet, self).__init__()

        self.hparams = hparams
        self.net = model(**vars(hparams))

        self.register_buffer(
            'edge_index', torch.LongTensor(2, hparams.num_edges))
        self.register_buffer(
            'edge_weight', torch.FloatTensor(hparams.num_edges))

        self.epoch_count = 0

    def init_graph(self, edge_index, edge_weight):
        self.edge_index.copy_(edge_index)
        self.edge_weight.copy_(edge_weight)

    def init_data(self, training_input, training_target, val_input, val_target, test_input, test_target):
        print('preparing data...')
        self.training_input = training_input
        self.training_target = training_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target

    def make_sample_dataloader(self, X, y, shuffle):
        # return a data loader based on saint
        dataset = SaintDataset(
            X, y, self.edge_index, self.edge_weight,
            num_nodes=self.hparams.num_nodes, batch_size=batch_size, shuffle=shuffle
        )
        return DataLoader(dataset, batch_size=None)

    def train_dataloader(self):
        return self.make_sample_dataloader(self.training_input, self.training_target, shuffle=True)

    def val_dataloader(self):
        return self.make_sample_dataloader(self.val_input, self.val_target, shuffle=False)

    def forward(self, X, g):
        return self.net(X, g)

    def training_step(self, batch, batch_idx):
        X, y, g, rows = batch
        g = g.to(X.device)
        y_hat = self(X, g)
        assert(y.size() == y_hat.size())
        loss = loss_criterion(y_hat, y)
        loss = (loss * g.node_norm.view(1, -1, 1)).sum()
        # loss = loss.mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        X, y, g, rows = batch

        g = g.to(X.device)
        y_hat = self(X, g)
        assert(y.size() == y_hat.size())

        out_dim = y.size(-1)

        index_ptr = torch.cartesian_prod(
            torch.arange(rows.size(0)),
            torch.arange(g.node_idx.size(0)),
            torch.arange(out_dim)
        )

        label = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g.node_idx[index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g.node_idx[index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y_hat[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

        return {'label': label, 'pred': pred}

    def validation_epoch_end(self, outputs):
        pred = pd.concat([x['pred'] for x in outputs], axis=0)
        label = pd.concat([x['label'] for x in outputs], axis=0)

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

        loss = np.mean((pred.values - label.values) ** 2)

        return {'log': {'val_loss': loss}, 'progress_bar': {'val_loss': loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    start_time = time.time()
    print('Arguments:')
    print(args)
    torch.manual_seed(7)

    if args.dataset == "metr":
        A, X, means, stds = load_metr_la_data()
    else:
        A, X, means, stds = load_nyc_sharing_bike_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A = torch.from_numpy(A)
    sparse_A = A.to_sparse()
    edge_index = sparse_A._indices()
    edge_weight = sparse_A._values()

    print('Total nodes: ', A.size(0))
    print('Average degree: ', edge_index.size(1) / A.size(0))

    contains_self_loops = torch_geometric.utils.contains_self_loops(edge_index)
    print('Contains self loops: ', contains_self_loops)
    if torch_geometric.utils.contains_self_loops(edge_index):
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(edge_index, edge_weight)

    hparams = Namespace(**{
        'num_nodes': A.shape[0],
        'num_edges': edge_weight.shape[0],
        'num_features': training_input.shape[3],
        'num_timesteps_input': num_timesteps_input,
        'num_timesteps_output': num_timesteps_output,
        'gcn_type': gcn_type,
    })

    net = WrapperNet(hparams)

    net.init_data(
        training_input, training_target,
        val_input, val_target,
        test_input, test_target
    )

    net.init_graph(edge_index, edge_weight)

    early_stop_callback = EarlyStopping(patience=early_stop_rounds)
    logger = TestTubeLogger(save_dir=log_dir, name=log_name)

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        distributed_backend='ddp',
        early_stop_callback=early_stop_callback,
        logger=logger,
    )
    trainer.fit(net)
    print('Training time {}'.format(time.time() - start_time))
