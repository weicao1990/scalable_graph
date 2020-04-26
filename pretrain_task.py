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

from krnn import KRNN

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
        self.dataset = 'metr'  # choices: metr, nyc
        # choices: ./data/METR-LA, ./data/NYC-Sharing-Bike
        self.data_dir = './data/METR-LA'
        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.batch_size = 32
        self.num_timesteps_input = 12  # the length of the input time-series sequence
        self.num_timesteps_output = 3  # the length of the output time-series sequence
        self.lr = 1e-3  # the learning rate


class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_class = KRNN
        self.net = model_class(
            config.num_nodes,
            config.num_features,
            config.num_timesteps_input,
            config.num_timesteps_output,
        )
        self.register_buffer('node_idx', torch.arange(config.num_nodes))

    def forward(self, X):
        return self.net(X, self.node_idx)


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

        # set config attributes for model initialization
        self.config.num_nodes = self.A.shape[0]
        self.config.num_features = self.training_input.shape[3]

    def make_sample_dataloader(self, X, y, batch_size, shuffle=False, use_dist_sampler=False, rep_eval=None):
        dataset = TensorDataset(X, y)

        if use_dist_sampler and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def build_train_dataloader(self):
        return self.make_sample_dataloader(
            self.training_input, self.training_target, batch_size=self.config.batch_size, shuffle=True, use_dist_sampler=True
        )

    def build_val_dataloader(self):
        # use a small batch size to test the normalization methods (BN/LN)
        return self.make_sample_dataloader(self.val_input, self.val_target, batch_size=self.config.batch_size)

    def build_test_dataloader(self):
        return self.make_sample_dataloader(self.test_input, self.test_target, batch_size=self.config.batch_size)

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        assert(y.size() == y_hat.size())
        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        # # debug distributed sampler
        # if batch_idx == 0:
        #     self.log('indices: {}'.format(rows))
        #     self.log('g.cent_n_id: {}'.format(g['cent_n_id']))

        return {
            LOSS_KEY: loss,
            BAR_KEY: {'train_loss': loss_i},
            SCALAR_LOG_KEY: {'train_loss': loss_i}
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y = batch

        y_hat = self.model(X)
        assert(y.size() == y_hat.size())

        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        return {'loss': loss_i}

    def eval_epoch_end(self, outputs, tag):
        loss = np.mean([x['loss'] for x in outputs])

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
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Pretrain')
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

    if task.config.skip_train:
        task.init_model_and_optimizer(net)
    else:
        task.fit(net)

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    task.log('Training time {}s'.format(time.time() - start_time))
