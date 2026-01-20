import numpy as np


class LinearModel:
    """
    Linear model f(x) = x^T w with no intercept.

    This class only represents the hypothesis.
    Training logic is handled by solvers / optimizers.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.reset()

    def reset(self):
        """
        Reset parameters to zero.
        """
        self.w = np.zeros(self.n_features)

    def initialize(self, seed: int | None = None, scale: float = 1e-2):
        """
        Randomly initialize parameters.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.
        scale : float
            Standard deviation of initialization.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.w = rng.normal(0.0, scale, size=self.n_features)
        else:
            self.w = np.random.normal(0.0, scale, size=self.n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for input matrix X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected X with {self.n_features} features, "
                f"got {X.shape[1]}"
            )

        return X @ self.w

    def norm(self) -> float:
        """
        Return the Euclidean norm of the parameter vector.
        """
        return np.linalg.norm(self.w)
