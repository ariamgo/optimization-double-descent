import numpy as np
from .base import Solver
from ..objectives.least_squares import LeastSquares
from ..objectives.ridge import Ridge


class ClosedFormSolver(Solver):
    """
    Closed-form solver for least squares and ridge regression.
    """

    def solve(self, model, objective, X, y):
        if isinstance(objective, LeastSquares):
            model.w = np.linalg.pinv(X) @ y
            return None

        if isinstance(objective, Ridge):
            n, p = X.shape
            A = X.T @ X / n + objective.lam * np.eye(p)
            b = X.T @ y / n
            model.w = np.linalg.solve(A, b)
            return None

        raise TypeError(
            "ClosedFormSolver supports only LeastSquares and Ridge objectives."
        )
