import numpy as np
from .base import Solver


class GradientDescent(Solver):
    def __init__(
        self,
        lr=1e-2,
        max_iter=10_000,
        tol=1e-6,
        record_history=False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.record_history = record_history

    def solve(self, model, objective, X, y):
        history = []

        for _ in range(self.max_iter):
            grad = objective.grad(X, y, model.w)
            step = self.lr * grad
            model.w -= step

            if self.record_history:
                history.append({
                    "loss": objective.loss(X, y, model.w),
                    "norm": model.norm(),
                })

            if np.linalg.norm(grad) < self.tol:
                break

        return history if self.record_history else None
