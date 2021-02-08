#!/usr/bin/env python
# coding: utf-8

""" Learning Koopman Invariant Subspace
 (c) Naoya Takeishi, 2017.
 takeishi@ailab.t.u-tokyo.ac.jp
"""

import numpy as np
np.random.seed(1234567890)

from argparse import ArgumentParser
from os import path
import time

from lkis import TimeSeriesBatchMaker, KoopmanInvariantSubspaceLearner
from losses import combined_loss
from torch.optim import SGD

# -- Parse arguments
t = time.time()
parser = ArgumentParser(description='Learning Koopman Invariant Subspace (Now with PyTorch!)')
parser.add_argument("--name", "-n", type=str, default=f"lkis-{int(time.time())}", help="name of experiment")
parser.add_argument("--data-path", type=str, default="./train.npy", help="time-series data to model")
parser.add_argument("--epochs", "-e", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--num-batches", "-b", type=int, default=1, help="how many batchs for break the data up into")
parser.add_argument("--gpu", action="store_true", default=False, help="use a GPU or no")
parser.add_argument("--intermediate-observable", "-i", type=int, default=-1, help="intermediate dimensional observation space")
parser.add_argument("--save-model", "-m", type=str, default="./", help="where to save you model")
parser.add_argument("--save-training-plot", "-p", type=str, default="./", help="where to save plotting")
parser.add_argument("--max-lag", "-l", type=int, default=-1, help="maximum_lag")
parser.add_argument("--state-space", "-s", type=int, default=1)
parser.add_argument("--alpha", "-a", type=float, default=1.0, help="value to score the reconstruction loss by")
if __name__ == "__main__":
    cli_args = parser.parse_args()
    data_path = cli_args.data_path
    print(f"Loading training data from {data_path}")
    data_train = np.load(data_path)
    if len(data_train.shape) == 1:
        data_train = data_train.reshape(-1, 1)
    print(f"Loaded a dataset with dimension: {data_train.shape}")
    delay = cli_args.max_lag if cli_args.max_lag > 0 else (data_train.shape[0] // 10)
    samples_per_batch = (data_train.shape[0] - delay) // cli_args.num_batches

    batch_iterator = TimeSeriesBatchMaker(
        y=data_train,
        batch_size=samples_per_batch,
        max_lag=delay
    )

    lkis = KoopmanInvariantSubspaceLearner(
        observable_dim=data_train.shape[1],
        latent_dim=cli_args.state_space,
        intermediate_observable=cli_args.intermediate_observable,
        delay=delay
    )

    optimizer = SGD(lkis.parameters(), lr=0.001)
    losses = []
    for epoch in range(cli_args.epochs):
        loss = 0
        for b in range(cli_args.num_batches):
            time_delayed_ys, y_true = next(batch_iterator)

            g_pred, y_pred = lkis(time_delayed_ys)
            g_0 = g_pred[:-1]
            g_1 = g_pred[1:]

            batch_loss = combined_loss(y_pred=y_pred, y_true=y_true, g_0=g_0, g_1=g_1)

            batch_loss.backward()

            optimizer.step()

            loss += batch_loss.item()

        # compute the epoch training loss
        loss = loss / (cli_args.num_batches * samples_per_batch)

        # display the epoch training loss
        print(f"epoch : {epoch + 1}/{cli_args.epochs}, loss = {loss:.6f}")
        losses.append(loss)
