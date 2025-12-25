import numpy as np

class OLS:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w = np.linalg.pinv(X) @ y
        return self

    def predict(self, X):
        return X @ self.w

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def train_error(self):
        return self.mse(self.X, self.y)

class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        p = X.shape[1]
        A = X.T @ X + self.lam * np.eye(p)
        self.w = np.linalg.solve(A, X.T @ y)
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        return X @ self.w

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

from sklearn.linear_model import LinearRegression, Ridge as SkRidge

def fit_sklearn_ols(X, y):
    model = LinearRegression(
        fit_intercept=False,
        copy_X=True,
    )
    model.fit(X, y)
    return model

def fit_sklearn_ridge(X, y, alpha):
    model = SkRidge(
        alpha=alpha,
        fit_intercept=False,
        solver="auto",
    )
    model.fit(X, y)
    return model
