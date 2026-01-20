import numpy as np
from .base import Objective


class Ridge(Objective):
    """
    Ridge regression objective:
        (1/2n) ||Xw - y||^2 + (λ/2) ||w||^2
    """

    def __init__(self, lam: float):
        self.lam = lam

    def loss(self, X, y, w):
        r = X @ w - y
        return 0.5 * np.mean(r ** 2) + 0.5 * self.lam * np.sum(w ** 2)

    def grad(self, X, y, w):
        n = X.shape[0]
        return X.T @ (X @ w - y) / n + self.lam * w
