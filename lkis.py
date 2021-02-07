#!/usr/bin/env python
# coding: utf-8
import numpy as np
from torch import from_numpy
from torch.nn import Module, Linear, PReLU

from typing import Tuple

class TimeSeriesBatchMaker(object):
    def __init__(self, y:np.ndarray, max_lag:int = 1, batch_size:int = 0 ) -> None:
        self.maximum_lag = min(max_lag, y.shape[0])
        self.start_ = self.maximum_lag
        self.batch_size = y.shape[0] if batch_size == 0 else min(batch_size, y.shape[0] - max_lag)
        self.samples = y.copy()

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        ys = []
        for _ in range(
                min(
                    self.samples.shape[0] - self.start,
                    self.batch_size
                )
        ):
            y_flattened_t = [self.samples[self.start_ - j] for j in range(self.maximum_lag)]
            self.start_ += 1
            ys.append(np.hstack(y_flattened_t))

        if self.start_ == self.samples[0]:
            self.start_ = 0

        ys = np.vstack(ys)

        batch_y0 = from_numpy(ys[:-1])
        batch_y1 = from_numpy(ys[1:])

        return (batch_y0, batch_y1)

class LinearDelayEmbedder(Module):
    """
    LinearDelayEmbedder that tries to "reconstruct the latent state space
    """
    def __init__(self, observable_dim:int, delay:int, latent_dim:int) -> None:
        super.__init__()
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
        self.linear_2 = Linear(n_hidden, observable_dim)

    def forward(self, x):
        x_hat = self.linear_1(x)
        x_hat = self.prelu(x_hat)
        return self.linear_2(x_hat)


class DynamicalAutoEncoder(Module):
    def __init__(self, observable_dim: int, latent_dim: int, delay: int, intermediate_observable:int=-1):
        super().__init__()

        self.linear_delay_embedder = LinearDelayEmbedder(
            observable_dim=observable_dim,
            delay=delay,
            latent_dim=latent_dim
        )
        self.intermediate_observable_dim = intermediate_observable if intermediate_observable > 0 else latent_dim
        if intermediate_observable <= 0:
            print(f"Overwriting intermediate_observable dimension from {intermediate_observable} "
                  f"to {self.intermediate_observable_dim}")
        self.observer = Observer(latent_dim=latent_dim, observable_dim=self.intermediate_observable_dim)
        self.reconstructor = Observer(latent_dim=self.intermediate_observable_dim, observable_dim=observable_dim)

    def forward(self, y):
        x_hat = self.encoder(y)
        g = self.observer(x_hat)
        y_hat = self.reconstructor(g)

        return (g, y_hat)

