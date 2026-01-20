import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def _extract(results: List[Dict[str, Any]], key: str):
    return np.array([r[key] for r in results])

def plot_train_test_error(results, ax=None):
    p = _extract(results, "p")
    train_err = _extract(results, "train_error")
    test_err = _extract(results, "test_error")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(p, train_err, marker=".", label="Train error",)
    ax.plot(p, test_err, marker=".", label="Test error")

    ax.set_xlabel("Model capacity (p)")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Double descent")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def plot_parameter_norm(results, ax=None, logy=True):
    p = _extract(results, "p")
    norm = _extract(results, "norm")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(p, norm, marker=".")

    ax.set_xlabel("Model capacity (p)")
    ax.set_ylabel(r"$\|w\|_2$")
    ax.set_title("Parameter norm vs capacity")

    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    return ax

def plot_condition_number(results, ax=None, logy=True):
    p = _extract(results, "p")
    cond = _extract(results, "condition_number")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(p, cond, marker=".")

    ax.set_xlabel("Model capacity (p)")
    ax.set_ylabel("Condition number")
    ax.set_title("Design matrix conditioning")

    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    return ax

def plot_effective_rank(results, ax=None):
    p = _extract(results, "p")
    rank = _extract(results, "effective_rank")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(p, rank, marker="o")

    ax.set_xlabel("Model capacity (p)")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank vs capacity")
    ax.grid(True, alpha=0.3)

    return ax

def plot_test_error_multiple(experiments, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for exp, label in zip(experiments, labels):
        p = _extract(exp["results"], "p")
        test_err = _extract(exp["results"], "test_error")
        ax.plot(p, test_err, marker="o", label=label)

    ax.set_xlabel("Model capacity (p)")
    ax.set_ylabel("Test error")
    ax.set_title("Test error comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

