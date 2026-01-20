import numpy as np


class RegressionDataset:
    def __init__(
        self,
        n_samples,
        n_features,
        noise_std=0.0,
        test_ratio=0.3,
        correlation="iid",
        rho=0.0,
        seed=42,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_std = noise_std
        self.test_ratio = test_ratio
        self.correlation = correlation
        self.rho = rho
        self.seed = seed

    def generate(self):
        rng = np.random.default_rng(self.seed)

        X = self._design_matrix(rng)

        w_true = rng.normal(
            0.0, 1.0 / np.sqrt(self.n_features), size=self.n_features
        )

        y = X @ w_true
        if self.noise_std > 0:
            y += rng.normal(0.0, self.noise_std, size=self.n_samples)

        n_test = int(self.test_ratio * self.n_samples)
        idx = rng.permutation(self.n_samples)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]
        self.w_true = w_true

        return self.X_train, self.y_train, self.X_test, self.y_test

    def _design_matrix(self, rng):
        if self.correlation == "iid":
            return rng.normal(0.0, 1.0, size=(self.n_samples, self.n_features))

        if self.correlation == "toeplitz":
            idx = np.arange(self.n_features)
            cov = self.rho ** np.abs(idx[:, None] - idx[None, :])
            L = np.linalg.cholesky(cov)
            Z = rng.normal(size=(self.n_samples, self.n_features))
            return Z @ L.T

        raise ValueError(f"Unknown correlation type: {self.correlation}")

    # ---- Diagnostics ----
    def condition_number(self):
        s = np.linalg.svd(self.X_train, compute_uv=False)
        return s.max() / s.min()

    def effective_rank(self, tol=1e-10):
        s = np.linalg.svd(self.X_train, compute_uv=False)
        return np.sum(s > tol * s.max())
