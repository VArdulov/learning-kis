from torch import Tensor, matmul, pinverse, pow
from torch.nn.functional import mse_loss
from torch.linalg import norm


def rss_loss(g_0: Tensor, g_1: Tensor):
    g_0_pinv = pinverse(g_0.T)
    A_hat = matmul(g_1.T, g_0_pinv)
    g_1_hat = matmul(A_hat, g_0.T)
    return pow(norm(g_1.T - g_1_hat, ord="fro"), 2)


def combined_loss(y_pred, y_true, g_0, g_1, alpha=1.0):
    rss = rss_loss(g_0, g_1)
    mse = mse_loss(y_true, y_pred)
    return rss + (alpha * mse)
