import numpy as np
from typing import Iterable, Dict, Any
from tqdm import tqdm
from joblib import Parallel, delayed
from contextlib import contextmanager
import joblib


@contextmanager
def tqdm_joblib(tqdm_object):
    
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

class DoubleDescentExperiment:
    """
    Runs a double descent experiment by sweeping model capacity p
    at fixed sample size n (parallelized).
    """

    def __init__(
        self,
        dataset_cls,
        model_cls,
        objective,
        solver,
        n_samples: int,
        p_grid: Iterable[int],
        noise_std: float = 0.0,
        test_ratio: float = 0.3,
        correlation: str = "iid",
        rho: float = 0.0,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.objective = objective
        self.solver = solver

        self.n_samples = n_samples
        self.p_grid = list(p_grid)
        self.noise_std = noise_std
        self.test_ratio = test_ratio
        self.correlation = correlation
        self.rho = rho
        self.seed = seed
        self.n_jobs = n_jobs

        self.results = []

    
    # Single experiment
    def _run_single(self, p: int) -> Dict[str, Any]:
        dataset = self.dataset_cls(
            n_samples=self.n_samples,
            n_features=p,
            noise_std=self.noise_std,
            test_ratio=self.test_ratio,
            correlation=self.correlation,
            rho=self.rho,
            seed=self.seed + p,  # deterministic but unique
        )

        Xtr, ytr, Xte, yte = dataset.generate()

        model = self.model_cls(n_features=p)
        model.reset()

        self.solver.solve(model, self.objective, Xtr, ytr)

        return {
            "p": p,
            "n_train": Xtr.shape[0],
            "train_error": self.objective.loss(Xtr, ytr, model.w),
            "test_error": self.objective.loss(Xte, yte, model.w),
            "norm": model.norm(),
            "condition_number": dataset.condition_number(),
            "effective_rank": dataset.effective_rank(),
        }

    # --------------------------------------------------
    # Parallel run with tqdm
    # --------------------------------------------------
    def run(self) -> Dict[str, Any]:

        with tqdm_joblib(
            tqdm(total=len(self.p_grid), desc="Sweeping p")
        ):
            self.results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
            )(
                delayed(self._run_single)(p)
                for p in self.p_grid
            )

        return {
            "config": self._config(),
            "results": self.results,
        }

    def _config(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "p_grid": self.p_grid,
            "noise_std": self.noise_std,
            "test_ratio": self.test_ratio,
            "correlation": self.correlation,
            "rho": self.rho,
            "objective": type(self.objective).__name__,
            "solver": type(self.solver).__name__,
            "n_jobs": self.n_jobs,
        }
