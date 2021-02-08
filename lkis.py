#!/usr/bin/env python
# coding: utf-8
import numpy as np
from torch import from_numpy
from torch.nn import Module, Linear, PReLU, BatchNorm1d

from typing import Tuple
from logging import warning


class TimeSeriesBatchMaker(object):
    def __init__(self, y:np.ndarray, max_lag:int = 1, batch_size:int = 0) -> None:
        print(y.shape[0])
        print(max_lag)
        print(batch_size)
        self.maximum_lag = min(max_lag, y.shape[0]-1)
        self.start_ = self.maximum_lag
        self.batch_size = y.shape[0] if batch_size == 0 else min(batch_size, y.shape[0] - max_lag)
        self.samples = y.copy()

    def __next__(self, as_numpy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        We're effectively training an auto-encoder of sorts, just more convoluted.

        Basically what our network is trying to predict is y_t from Y_[T-L:T] = [y_t, y_t-1, y_t-2, ..., y_t-L]
        where L is the maximum lag. Note this is what the end-to-end network is trying to predict,
        however we're also subjecting it to another property, namely that it has to learn a intermediate representation
        g_t that also has to have the relationship that g_t+1 = A * g_t.

        With how this relates to this function. Assuming we are given a data set we need to extract the appropriate
        inputs and outputs associated with the values we described above
        """

        xs = []  # this will be where the time delayed y values will be stored
        ys = []  # this is where the batch ground truth will go

        """
        First we need to iterator that will count off the size of the batch or the dataset
        """
        T = min((self.start_ + self.batch_size), self.samples.shape[0])
        for t in range(self.start_, T):
            ys.append(self.samples[t])  # add the value that we will be trying to reconstruct later
            x = [self.samples[t-l] for l in range(self.maximum_lag)]  # this is the vector that will be used as the input
            x = np.hstack(x).reshape(1, -1)
            xs.append(x)

        # Now we might want to restart the batches so here we'll add some simple logic
        if T >= self.samples.shape[0]:
            self.start_ = self.maximum_lag
        else:
            self.start_ += self.batch_size

        xs = np.vstack(xs)
        ys = np.vstack(ys)
        if as_numpy:
            return xs, ys

        return from_numpy(xs).float(), from_numpy(ys).float()


class LinearDelayEmbedder(Module):
    """
    LinearDelayEmbedder that tries to "reconstruct the latent state space
    """
    def __init__(self, observable_dim:int, delay:int, latent_dim:int) -> None:
        super().__init__()
        self.linear_embedder = Linear(in_features=observable_dim*delay, out_features=latent_dim)

    def forward(self, x):
        return self.linear_embedder(x)


class Observer(Module):
    """
    Observer (using PRelu) that maps from the linear delay embedder into the state-space
    """
    def __init__(self, latent_dim:int, observable_dim:int):
        super().__init__()
        n_hidden = ((latent_dim + observable_dim) // 2) + ((latent_dim + observable_dim) % 2)
        self.linear_1 = Linear(latent_dim, n_hidden)
        self.prelu = PReLU()
        self.batch_norm = BatchNorm1d(n_hidden)
        self.linear_2 = Linear(n_hidden, observable_dim)

    def forward(self, x):
        x_hat = self.linear_1(x)
        x_hat = self.prelu(x_hat)
        x_hat = self.batch_norm(x_hat)
        return self.linear_2(x_hat)


class KoopmanInvariantSubspaceLearner(Module):
    def __init__(self, observable_dim: int, latent_dim: int, delay: int, intermediate_observable:int=-1):
        super().__init__()

        self.linear_delay_embedder = LinearDelayEmbedder(
            observable_dim=observable_dim,
            delay=delay,
            latent_dim=latent_dim
        )
        self.intermediate_observable_dim = intermediate_observable if intermediate_observable > 0 else latent_dim
        if intermediate_observable <= 0:
            warning(f"Overwriting intermediate_observable dimension from {intermediate_observable} "
                  f"to {self.intermediate_observable_dim}")
        self.observer = Observer(latent_dim=latent_dim, observable_dim=self.intermediate_observable_dim)
        self.reconstructor = Observer(latent_dim=self.intermediate_observable_dim, observable_dim=observable_dim)

    def forward(self, y):
        x_hat = self.linear_delay_embedder(y)
        g = self.observer(x_hat)
        y_hat = self.reconstructor(g)

        return (g, y_hat)

