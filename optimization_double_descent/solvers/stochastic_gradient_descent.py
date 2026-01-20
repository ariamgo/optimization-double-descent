import numpy as np
from .base import Solver


class StochasticGradientDescent(Solver):
    def __init__(
        self,
        lr=1e-2,
        batch_size=32,
        epochs=100,
        shuffle=True,
        record_history=False,
        seed=None,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.record_history = record_history
        self.seed = seed

    def solve(self, model, objective, X, y):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        history = []

        for _ in range(self.epochs):
            idx = rng.permutation(n) if self.shuffle else np.arange(n)

            for i in range(0, n, self.batch_size):
                batch = idx[i : i + self.batch_size]
                Xb, yb = X[batch], y[batch]

                grad = objective.grad(Xb, yb, model.w)
                model.w -= self.lr * grad

            if self.record_history:
                history.append({
                    "loss": objective.loss(X, y, model.w),
                    "norm": model.norm(),
                })

        return history if self.record_history else None
