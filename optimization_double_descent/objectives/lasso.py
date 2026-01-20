import numpy as np
from .base import Objective


class Lasso(Objective):
    """
    Lasso objective:
        (1/2n) ||Xw - y||^2 + λ ||w||_1
    """

    def __init__(self, lam: float):
        self.lam = lam

    def loss(self, X, y, w):
        r = X @ w - y
        return 0.5 * np.mean(r ** 2) + self.lam * np.sum(np.abs(w))

    def grad(self, X, y, w):
        n = X.shape[0]
        return X.T @ (X @ w - y) / n + self.lam * np.sign(w)
