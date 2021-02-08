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
from torch import device, save, manual_seed
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

# -- Parse arguments
t = time.time()
parser = ArgumentParser(description='Learning Koopman Invariant Subspace (Now with PyTorch!)')
parser.add_argument("--name", "-n", type=str, default=f"lkis-{int(time.time())}", help="name of experiment")
parser.add_argument("--data-path", type=str, default="./train.npy", help="time-series data to model")
parser.add_argument("--epochs", "-e", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--num-batches", "-b", type=int, default=1, help="how many batchs for break the data up into")
parser.add_argument("--gpu", action="store_true", default=False, help="use a GPU or no")
parser.add_argument("--intermediate-observable", "-i", type=int, default=-1, help="intermediate dimensional observation space")
parser.add_argument("--save-model", "-m", action="store_true", default=False, help="whether or not you want the model saved to $name$.torch.mdl")
parser.add_argument("--save-training-plot", "-p", action="store_true", default=False, help="where to save plotting")
parser.add_argument("--max-lag", "-l", type=int, default=-1, help="maximum_lag")
parser.add_argument("--state-space", "-s", type=int, default=1, help="dimensionality of the underlying state space")
parser.add_argument("--alpha", "-a", type=float, default=1.0, help="value to score the reconstruction loss by")
parser.add_argument("--learning-rate", "-r", type=float, default=0.001, help="Optimizer learning rate")
parser.add_argument("--validation-data-path", "-v", type=str, default="")

if __name__ == "__main__":
    # grab the command line arguments
    cli_args = parser.parse_args()
    manual_seed(216)

    # find and load the training data
    data_path = cli_args.data_path
    print(f"Loading training data from {data_path}")
    data_train = np.load(data_path)
    if len(data_train.shape) == 1:
        data_train = data_train.reshape(-1, 1)
    print(f"Loaded a dataset with dimension: {data_train.shape}")
    validate = cli_args.validation_data_path != ""
    data_val = None
    if validate:
        data_path = cli_args.validation_data_path
        print(f"Loading validation data from {data_path}")
        data_val = np.load(data_path)

    # process the delay either set by the user or is set to one 10th of the data
    delay = cli_args.max_lag if cli_args.max_lag > 0 else (data_train.shape[0] // 10)

    # based on the number of batches, delay, and size of the data compute the samples per batch
    samples_per_batch = (data_train.shape[0] - delay) // cli_args.num_batches

    # construct the data preparer
    batch_iterator = TimeSeriesBatchMaker(
        y=data_train,
        batch_size=samples_per_batch,
        max_lag=delay
    )
    if validate:
        val_batch_iterator = TimeSeriesBatchMaker(
            y=data_val,
            max_lag=delay
        )

    # construct the end-to-end model
    lkis = KoopmanInvariantSubspaceLearner(
        observable_dim=data_train.shape[1],
        latent_dim=cli_args.state_space,
        intermediate_observable=cli_args.intermediate_observable,
        delay=delay
    )

    if cli_args.gpu:
        device = device("cuda")

    # initialize the optimizer
    optimizer = SGD(lkis.parameters(), lr=cli_args.learning_rate)
    losses = []
    val_losses = []
    for epoch in range(cli_args.epochs):
        loss = 0
        for b in range(cli_args.num_batches):
            optimizer.zero_grad()
            time_delayed_ys, y_true = next(batch_iterator)

            if cli_args.gpu:
                time_delayed_ys.to(device)
                y_true.to(device)

            g_pred, y_pred = lkis(time_delayed_ys)
            g_0 = g_pred[:-1]
            g_1 = g_pred[1:]

            batch_loss = combined_loss(y_pred=y_pred, y_true=y_true, g_0=g_0, g_1=g_1)

            batch_loss.backward()

            optimizer.step()

            loss += batch_loss.item()

        # display the epoch training loss
        print(f"epoch : {epoch + 1}/{cli_args.epochs}, loss = {loss:.6f}")
        losses.append(loss)

        if validate:
            y_time_delayed_val, y_true = next(val_batch_iterator)
            if cli_args.gpu:
                y_time_delayed_val.to(device)
                y_true.to(device)

            g_pred, y_pred = lkis(y_time_delayed_val)
            g_0 = g_pred[:-1]
            g_1 = g_pred[1:]

            batch_loss = combined_loss(y_pred=y_pred, y_true=y_true, g_0=g_0, g_1=g_1)
            val_loss = batch_loss.item()
            print(f"\tval-loss  = {val_loss:.6f}")
            val_losses.append(val_loss)

    if cli_args.save_model:
        save(lkis, f"{cli_args.name}.torch.mdl")

    if cli_args.save_training_plot:
        sns.lineplot(x=list(range(cli_args.epochs)), y=losses, label="training loss")
        if validate:
            sns.lineplot(x=list(range(cli_args.epochs)), y=val_losses, label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Combined Reconstruction and DMD Loss")
        plt.title(f"Training Loss for {cli_args.name}")
        plt.savefig(f"{cli_args.name}-training-loss.png")
