import numpy as np

class RegressionDataset:
    def __init__(
        self,
        n_samples,
        n_features,
        noise_std=0.1,
        test_ratio=0.3,
        correlation="iid",   # "iid", "toeplitz"
        rho=0.0,
        seed=42,
    ):
        """
        Initialize a RegressionDataset object.

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset.
        n_features : int
            The number of features in the dataset.
        noise_std : float, default=0.1
            The standard deviation of the noise in the dataset.
        test_ratio : float, default=0.3
            The ratio of test samples to total samples.
        correlation : str, default="iid"
            The type of correlation between features. Can be "iid" or "toeplitz".
        rho : float, default=0.0
            The correlation coefficient between features in the "toeplitz" correlation type.
        seed : int, default=42
            The seed used to generate the dataset.

        Returns
        -------
        None
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_std = noise_std
        self.test_ratio = test_ratio
        self.correlation = correlation
        self.rho = rho
        self.seed = seed

        self._generate()

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

    def _generate(self):
        rng = np.random.default_rng(self.seed)

        X = self._design_matrix(rng)

        # Ground-truth weights (energy-normalized)
        w_true = rng.normal(
            0.0, 1.0 / np.sqrt(self.n_features), size=self.n_features
        )

        noise = rng.normal(0.0, self.noise_std, size=self.n_samples)
        y = X @ w_true + noise

        n_test = int(self.test_ratio * self.n_samples)
        idx = rng.permutation(self.n_samples)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]
        self.w_true = w_true

    # --------------------
    # Diagnostics
    # --------------------
    def condition_number(self):
        XtX = self.X_train.T @ self.X_train
        s = np.linalg.svd(XtX, compute_uv=False)
        return s.max() / s.min()

    def effective_rank(self, tol=1e-10):
        s = np.linalg.svd(self.X_train, compute_uv=False)
        return np.sum(s > tol * s.max())
