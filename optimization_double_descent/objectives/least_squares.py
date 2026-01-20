import numpy as np
from .base import Objective


class LeastSquares(Objective):
    """
    Least squares objective:
        (1/2n) ||Xw - y||^2
    """

    def loss(self, X, y, w):
        r = X @ w - y
        return 0.5 * np.mean(r ** 2)

    def grad(self, X, y, w):
        n = X.shape[0]
        return X.T @ (X @ w - y) / n
