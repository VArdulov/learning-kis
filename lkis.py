#!/usr/bin/env python
# coding: utf-8

""" Learning Koopman Invariant Subspace
 (c) Naoya Takeishi, 2017.
 takeishi@ailab.t.u-tokyo.ac.jp

 (c) Victor Ardulov, 2021.
 ardulov@usc.edu
"""

import numpy as np
from scipy import linalg
from torch import (
    matmul,
    inverse,
    transpose
)
# from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, PReLU, BatchNorm1d
from torch.nn.functional import mse_loss
# from chainer import link
# from chainer import Variable
# from chainer import Chain
# from chainer import dataset
# from chainer import reporter as reporter_module
# from chainer import training
# from chainer import initializers
# from chainer.training import extensions
# import chainer.functions as F
# import chainer.links as L


# ==========

def ls_solution(g0, g1):
    """
    Get least-squares solution matrix for regression from rows of g0
    to rows of g1. Both g0 and g1 are torch Variable.
    """
    g0_t = transpose(g0)
    if g0.shape[0] >= g0.shape[1]:
        g0_H = inverse(matmul(g0_t, g0))
        g0_pinv = matmul(g0_H, g0t)
    else:
        g0_H = inverse(matmul(g0, g0_t))
        g0_pinv = matmul(g0_t, g0_H)

    K = transpose(matmul(g0_pinv, g1))
    return K


def dmd(y0, y1, eps=1e-6):
    """
    Perform Dynamic Mode Decomposition (DMD)

    inputs:
    ---
    y0: numpy.ndarray
    y1: numpy.ndarray
    """
    # transpose inputs
    Y0 = y0.T
    Y1 = y1.T

    # perform the singular valud decomposition (SVD)
    U, S, Vh, = linalg.svd(Y0, full_matrices=False)

    # find non-zero (epsilon thresholded) singular values
    r = len(np.where(S >= eps)[0])

    # take the left eigen-vectors upto r
    U = U[:, :r]

    invS = np.diag(1. / S[:r])
    V = Vh.conj().T[:, :r]
    M = np.dot(np.dot(Y1, V), invS)
    A = np.dot(U.conj().T, M)

    lam, z_til, w_til = linalg.eig(A, left=True)
    w = np.dot(np.dot(M, w_til), np.diag(1. / lam)) + 1j * np.zeros(z_til.shape)
    z = np.dot(U, z_til) + 1j * np.zeros(z_til.shape)
    for i in range(w.shape[1]):
        z[:, i] = z[:, i] / np.dot(w[:, i].conj(), z[:, i])
    return lam, w, z

class DelayPairDataLoader():
    def __init__(self, values, dim_delay, n_lag=1):
        if isinstance(values, list):
            self.values = values
        else:
            self.values = [values, ]
        self.lens = tuple(value.shape[0] - (dim_delay - 1) * n_lag - 1 for value in self.values)
        self.a_s = [0 for i in range(sum(self.lens))]
        for i in range(sum(self.lens)):
            for j in range(len(self.values)):
                if i >= sum(self.lens[0:j]):
                    self.a_s[i] = j
        self.dim_delay = dim_delay
        self.n_lag = n_lag

    def __len__(self):
        return sum(self.lens)

    def get_example(self, i):
        tau = self.n_lag
        k = self.dim_delay
        a = self.a_s[i]
        start = i - sum(self.lens[0:a])
        end = start + (k-1) * tau + 1
        return (
            self.values[a][start:end:tau],
            self.values[a][start + 1:end + 1:tau]
        )


class Encoder(Module):
    def __init__(self, dimy, delay, dim_emb):
        # super(Encoder, self).__init__(l1 = L.Linear(dimy*delay, dim_emb))
        super().__init__()
        self.l1 = Linear(in_features=(dimy * delay), out_features=dim_emb)

    def __call__(self, x):
        return self.l1(x)


class Observable(Module):
    def __init__(self, dim_g, dim_emb):
        n_hidden = ((dim_g + dim_emb) // 2) + ((dim_g + dim_emb) % 2)

        super().__init__()
        self.l1 = Linear(dim_emb, n_hidden)
        self.p1 = PReLU()
        self.b1 = BatchNorm1d(num_features=n_hidden)
        self.l2 = Linear(in_features=n_hidden, out_features=dim_g)

        self.dim_g = dim_g

    def __call__(self, x):
        x_hat = self.l1(x)
        x_hat = self.p1(x_hat)
        x_hat = self.b1(x_hat)
        return self.l2(x_hat)


class Reconstructor(Module):
    def __init__(self, dim_y, dim_g):
        n_hidden = ((dim_y + dim_g) // 2) + ((dim_y + dim_g) % 2)

        super().__init__()
        self.l1 = Linear(dim_g, n_hidden)
        self.p1 = PReLU()
        self.b1 = BatchNorm1d(num_features=n_hidden)
        self.l2 = Linear(n_hidden, dim_y)

    def __call__(self, x):
        # The nonlinearlity of Reconstructor is realized by p1 (PReLU function),
        # so eliminating p1 from calculation makes Reconstructor linear.
        x_hat = self.l1(x)
        x_hat = self.p1(x_hat)
        x_hat = self.b1(x_hat)
        return self.l2(x_hat)


class Network(Module):
    def __init__(self, dim_emb, dim_g, dim_y):
        super().__init__()
        self.b1 = BatchNorm1d(dim_emb)
        self.g = Observable(dim_g, dim_emb)
        self.h = Reconstructor(dim_y, dim_g)

    def __call__(self, y0, y1, phi):
        x0 = self.b(phi(y0))
        g0 = self.g(x0)
        h0 = self.h(g0)

        x1 = self.b(phi(y1))
        g1 = self.g(x1)
        h1 = self.h(g1)

        return g0, g1, h0, h1


class CombinedLoss():
    def __init__(self, phi, net, alpha=1.0, decay=0.9):
        self.phi = phi
        self.net = net
        self.alpha = alpha
        self.decay = decay

    def __call__(self, y0, y1):
        g0, g1, h0, h1 = self.net(y0, y1, phi=self.phi)
        g1_pred = matmul(ls_solution(g0, g1), g0)
        loss1 = mse_loss(g1_pred, g1)
        loss2 = mse_loss(h0, transpose(y0, axes=(1, 0, 2))[-1]) # reconstruction loss
        loss3 = mse_loss(h1, F.transpose(y1, axes=(1, 0, 2))[-1])
        loss = loss1 + self.alpha * 0.5 * (loss2 + loss3)

        return loss


# # ==========
#
# class Updater(training.StandardUpdater):
#     def update_core(self):
#         batch = self._iterators['main'].next()
#         in_arrays = self.converter(batch, self.device)
#         in_vars = tuple(Variable(x) for x in in_arrays)
#         for optimizer in self._optimizers.values():
#             optimizer.update(self.loss_func, *in_vars)
#
#
# # ==========
#
# class Evaluator(extensions.Evaluator):
#     def __init__(self, iterator, target, converter=dataset.convert.concat_examples,
#                  device=None, eval_hook=None, eval_func=None, trigger=(1, 'epoch')):
#         if isinstance(iterator, dataset.iterator.Iterator):
#             iterator = {'main': iterator}
#         self._iterators = iterator
#
#         if isinstance(target, link.Link):
#             target = {'main': target}
#         self._targets = target
#
#         self.converter = converter
#         self.device = device
#         self.eval_hook = eval_hook
#         self.eval_func = eval_func
#         self.trigger = trigger
#
#     def evaluate(self):
#         iterator = self._iterators['main']
#         target = self._targets['main']
#         eval_func = self.eval_func or target
#
#         if self.eval_hook:
#             self.eval_hook(self)
#
#         if hasattr(iterator, 'reset'):
#             iterator.reset()
#             it = iterator
#         else:
#             it = copy.copy(iterator)
#
#         summary = reporter_module.DictSummary()
#         for batch in it:
#             observation = {}
#             with reporter_module.report_scope(observation):
#                 in_arrays = self.converter(batch, self.device)
#                 in_vars = tuple(Variable(x, volatile='on')
#                                 for x in in_arrays)
#                 eval_func(*in_vars, train=False)
#                 summary.add(observation)
#
#         return summary.compute_mean()
