import os
import time
import argparse
import json
import math
import torch
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from torch_geometric.data import Data, Batch, NeighborSampler, ClusterData, ClusterLoader
import torch.nn as nn
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tgcn import TGCN
from sandwich import Sandwich

from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY


class STConfig(BaseConfig):
    def __init__(self):
        super(STConfig, self).__init__()
        # 1. Reset base config variables:
        self.max_epochs = 1000
        self.early_stop_epochs = 30

        # 2. set spatial-temporal config variables:
        self.model = 'sandwich'  # choices: tgcn, stgcn, gwnet
        self.dataset = 'metr'  # choices: metr, nyc
        # choices: ./data/METR-LA, ./data/NYC-Sharing-Bike
        self.data_dir = './data/METR-LA'
        self.gcn = 'gat'  # choices: sage, gat

        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.batch_size = 32
        self.normalize = 'none'
        self.num_timesteps_input = 12  # the length of the input time-series sequence
        self.num_timesteps_output = 3  # the length of the output time-series sequence
        self.lr = 1e-3  # the learning rate
        self.rep_eval = 3  # do evaluation for multiple times

        # pretrained ckpt for krnn, use 'none' to ignore it
        self.pretrain_ckpt = 'none'
        self.use_residual = True


def get_model_class(model):
    return {
        'tgcn': TGCN,
        'sandwich': Sandwich
    }.get(model)


class NeighborSampleDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, shuffle=False, use_dist_sampler=False, rep_eval=None):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        # number of repeats to run evaluation, set to None for training mode
        self.rep_eval = rep_eval

        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
        ).to('cpu')

        graph_sampler = NeighborSampler(
            # graph, size=[5, 5], num_hops=2, batch_size=100, shuffle=self.shuffle, add_self_loops=True
            graph, size=[10, 15], num_hops=2, batch_size=250, shuffle=self.shuffle, add_self_loops=True
        )

        return graph_sampler

    def get_subgraph(self, data_flow):
        sub_graph = {
            'edge_index': [block.edge_index for block in data_flow],
            'edge_weight': [self.edge_weight[block.e_id] for block in data_flow],
            'size': [block.size for block in data_flow],
            'res_n_id': [block.res_n_id for block in data_flow],
            'cent_n_id': data_flow[-1].n_id[data_flow[-1].res_n_id],
            'graph_n_id': data_flow[0].n_id
        }

        return sub_graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            # decide random seeds for graph sampler

            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self.epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            for data_flow in self.graph_sampler():
                g = self.get_subgraph(data_flow)
                X, y = self.X[:, g['graph_n_id']], self.y[:, g['cent_n_id']]
                dataset_len = X.size(0)
                indices = list(range(dataset_len))

                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self.epoch)
                        indices = torch.randperm(
                            dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size -
                            1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0

        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            for data_flow in self.graph_sampler():
                if self.use_dist_sampler and dist.is_initialized():
                    dataset_len = self.X.size(0)
                    world_size = dist.get_world_size()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                else:
                    num_samples_per_node = self.X.size(0)
                length += (num_samples_per_node +
                        self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch


class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_class = get_model_class(config.model)
        self.net = model_class(**vars(config))

    def forward(self, X, g):
        return self.net(X, g)


class SpatialTemporalTask(BasePytorchTask):
    def __init__(self, config):
        super(SpatialTemporalTask, self).__init__(config)
        self.log('Intialize {}'.format(self.__class__))

        self.init_data()
        self.loss_func = nn.MSELoss()

        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))

    def init_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.config.data_dir

        if self.config.dataset == "metr":
            A, X, means, stds = load_metr_la_data(data_dir)
        else:
            A, X, means, stds = load_nyc_sharing_bike_data(data_dir)

        split_line1 = int(X.shape[2] * 0.6)
        split_line2 = int(X.shape[2] * 0.8)
        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]

        self.training_input, self.training_target = generate_dataset(train_original_data,
                                                                     num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
                                                                     )
        self.val_input, self.val_target = generate_dataset(val_original_data,
                                                           num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
                                                           )
        self.test_input, self.test_target = generate_dataset(test_original_data,
                                                             num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
                                                             )

        self.A = torch.from_numpy(A)
        self.sparse_A = self.A.to_sparse()
        self.edge_index = self.sparse_A._indices()
        self.edge_weight = self.sparse_A._values()

        contains_self_loops = torch_geometric.utils.contains_self_loops(
            self.edge_index)
        self.log('Contains self loops: {}, but we add them.'.format(
            contains_self_loops))
        if not contains_self_loops:
            self.edge_index, self.edge_weight = torch_geometric.utils.add_self_loops(
                self.edge_index, self.edge_weight,
                num_nodes=self.A.shape[0]
            )

        # set config attributes for model initialization
        self.config.num_nodes = self.A.shape[0]
        self.config.num_edges = self.edge_weight.shape[0]
        self.config.num_features = self.training_input.shape[3]
        self.log('Total nodes: {}'.format(self.config.num_nodes))
        self.log('Average degree: {:.3f}'.format(
            self.config.num_edges / self.config.num_nodes))

    def make_sample_dataloader(self, X, y, batch_size, shuffle=False, use_dist_sampler=False, rep_eval=None):
        # return a data loader based on neighbor sampling
        dataset = NeighborSampleDataset(
            X, y, self.edge_index, self.edge_weight, self.config.num_nodes, batch_size,
            shuffle=shuffle, use_dist_sampler=use_dist_sampler, rep_eval=rep_eval
        )

        return DataLoader(dataset, batch_size=None)

    def build_train_dataloader(self):
        return self.make_sample_dataloader(
            self.training_input, self.training_target, batch_size=self.config.batch_size, shuffle=True, use_dist_sampler=True
        )

    def build_val_dataloader(self):
        # use a small batch size to test the normalization methods (BN/LN)
        return self.make_sample_dataloader(self.val_input, self.val_target, batch_size=8, rep_eval=self.config.rep_eval)

    def build_test_dataloader(self):
        return self.make_sample_dataloader(self.test_input, self.test_target, batch_size=8, rep_eval=self.config.rep_eval)

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def load_pretrain_ckpt(self):
        if self.config.pretrain_ckpt == 'none':
            return
        state_dict = torch.load(self.config.pretrain_ckpt)['model']

        if self.has_parallel_wrapper(self.model):
            model = self.model.module.net.gru1.seq2seq
        else:
            model = self.model.net.gru1.seq2seq

        for name, param in model.named_parameters():
            name = 'net.seq2seq.{}'.format(name)
            param.data.copy_(state_dict[name])
            # # if krnn is pretrained, we then freeze it
            param.requires_grad = False

    def train_step(self, batch, batch_idx):
        X, y, g, rows = batch
        # debug distributed sampler
        if batch_idx == 0:
            self.log('train batch {} indices: {}'.format(batch_idx, rows))
            self.log('train batch {} g.cent_n_id: {}'.format(batch_idx, g['cent_n_id']))
            self.log('train batch {} g.graph_n_id: {}'.format(batch_idx, g['graph_n_id']))

        y_hat = self.model(X, g)
        assert(y.size() == y_hat.size())
        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        return {
            LOSS_KEY: loss,
            BAR_KEY: {'train_loss': loss_i},
            SCALAR_LOG_KEY: {'train_loss': loss_i}
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y, g, rows = batch
        # debug repetitive evaluation
        if batch_idx == 0:
            self.log('{} batch {} indices: {}'.format(tag, batch_idx, rows))
            self.log('{} batch {} g.cent_n_id: {}'.format(tag, batch_idx, g['cent_n_id']))
            self.log('{} batch {} g.graph_n_id: {}'.format(tag, batch_idx, g['graph_n_id']))

        y_hat = self.model(X, g)
        assert(y.size() == y_hat.size())

        out_dim = y.size(-1)

        index_ptr = torch.cartesian_prod(
            torch.arange(rows.size(0)),
            torch.arange(g['cent_n_id'].size(0)),
            torch.arange(out_dim)
        )

        label = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g['cent_n_id'][index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g['cent_n_id'][index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y_hat[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

        return {
            'label': label,
            'pred': pred,
        }

    def eval_epoch_end(self, outputs, tag):
        pred = pd.concat([x['pred'] for x in outputs], axis=0)
        label = pd.concat([x['label'] for x in outputs], axis=0)

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

        loss = np.mean((pred.values - label.values) ** 2)

        out = {
            BAR_KEY: {'{}_loss'.format(tag): loss},
            SCALAR_LOG_KEY: {'{}_loss'.format(tag): loss},
            VAL_SCORE_KEY: -loss,  # a larger score corresponds to a better model
        }

        return out

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')


if __name__ == '__main__':
    start_time = time.time()

    # build argument parser and config
    config = STConfig()
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    # build task
    task = SpatialTemporalTask(config)

    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    task.set_random_seed()
    net = WrapperNet(task.config)
    task.init_model_and_optimizer(net)
    task.load_pretrain_ckpt()

    if not task.config.skip_train:
        task.fit()

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    task.log('Training time {}s'.format(time.time() - start_time))
