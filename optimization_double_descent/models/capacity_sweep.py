import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from optimization_double_descent.data.synthetic import RegressionDataset
from optimization_double_descent.models.regression import OLS


def capacity_sweep(
    n_samples=100,
    p_min=5,
    p_max=300,
    p_step=5,
    noise_std=0.1,
    correlation="iid",
    rho=0.0,
    seed=42,
):
    """
    Perform a capacity sweep on a regression problem.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples in the dataset.
    p_min : int, default=5
        The minimum number of features in the sweep.
    p_max : int, default=300
        The maximum number of features in the sweep.
    p_step : int, default=5
        The step size of the feature grid.
    noise_std : float, default=0.1
        The standard deviation of the noise in the dataset.
    correlation : str, default="iid"
        The type of correlation between features. Can be "iid" or "toeplitz".
    rho : float, default=0.0
        The correlation coefficient between features in the "toeplitz" correlation type.
    seed : int, default=42
        The seed used to generate the dataset.

    Returns
    -------
    results : list of dict
        A list of dictionaries containing the results of the sweep.
        Each dictionary contains the number of features, the training and test mean squared errors of the OLS model,
        the norm of the weights of the OLS model, whether the OLS model interpolates, the condition number of the dataset,
        and the effective rank of the dataset.
    """
    p_grid = np.arange(p_min, p_max + 1, p_step)

    results = []

    for p in tqdm(p_grid, desc="Capacity sweep"):
        ds = RegressionDataset(
            n_samples=n_samples,
            n_features=p,
            noise_std=noise_std,
            correlation=correlation,
            rho=rho,
            seed=seed,
        )

        # ------------------
        # Exact OLS
        # ------------------
        ols = OLS().fit(ds.X_train, ds.y_train)

        train_mse_ours = ols.mse(ds.X_train, ds.y_train)
        test_mse_ours = ols.mse(ds.X_test, ds.y_test)
        weight_norm_ours = np.linalg.norm(ols.w)

        # ------------------
        # sklearn OLS
        # ------------------
        sk = LinearRegression(
            fit_intercept=False,
            copy_X=True,
        )
        sk.fit(ds.X_train, ds.y_train)

        train_pred_sk = sk.predict(ds.X_train)
        test_pred_sk = sk.predict(ds.X_test)

        train_mse_sk = np.mean((train_pred_sk - ds.y_train) ** 2)
        test_mse_sk = np.mean((test_pred_sk - ds.y_test) ** 2)
        weight_norm_sk = np.linalg.norm(sk.coef_)

        # ------------------
        # Diagnostics
        # ------------------
        cond = ds.condition_number()
        eff_rank = ds.effective_rank()

        results.append({
            "p": p,
            "train_mse_ours": train_mse_ours,
            "test_mse_ours": test_mse_ours,
            "weight_norm_ours": weight_norm_ours,
            "interpolates_ours": train_mse_ours < 1e-10,

            "train_mse_sklearn": train_mse_sk,
            "test_mse_sklearn": test_mse_sk,
            "weight_norm_sklearn": weight_norm_sk,
            "interpolates_sklearn": train_mse_sk < 1e-10,

            "condition_number": cond,
            "effective_rank": eff_rank,
        })

    return results

import matplotlib.pyplot as plt

def unpack_results(results):
    keys = results[0].keys()
    out = {k: np.array([r[k] for r in results]) for k in keys}
    return out

def plot_double_descent(
    results,
    n_train,
    title_suffix="",
):
    data = {k: np.array([r[k] for r in results]) for k in results[0]}
    p = data["p"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # =========================
    # 1. Train/Test MSE
    # =========================
    ax = axes[0]
    ax.plot(p, data["train_mse_ours"], label="Train MSE (ours)", linestyle="--")
    ax.plot(p, data["test_mse_ours"], label="Test MSE (ours)")
    ax.plot(p, data["train_mse_sklearn"], label="Train MSE (sklearn)", linestyle="--", alpha=0.7)
    ax.plot(p, data["test_mse_sklearn"], label="Test MSE (sklearn)", alpha=0.7)

    ax.axvline(n_train, color="k", linestyle=":", label="Interpolation threshold")
    ax.set_yscale("log")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(f"Double Descent: Error Curves {title_suffix}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # =========================
    # 2. Weight norm
    # =========================
    ax = axes[1]
    ax.plot(p, data["weight_norm_ours"], label="‖w‖ (ours)")
    ax.plot(p, data["weight_norm_sklearn"], label="‖w‖ (sklearn)", alpha=0.7)

    ax.axvline(n_train, color="k", linestyle=":")
    ax.set_yscale("log")
    ax.set_ylabel("Weight norm (log scale)")
    ax.set_title("Parameter Explosion at Interpolation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # =========================
    # 3. Conditioning
    # =========================
    ax = axes[2]
    ax.plot(p, data["condition_number"], label="cond(XᵀX)")
    ax.axvline(n_train, color="k", linestyle=":")

    ax.set_yscale("log")
    ax.set_xlabel("Number of features (p)")
    ax.set_ylabel("Condition number (log scale)")
    ax.set_title("Design Matrix Conditioning")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()