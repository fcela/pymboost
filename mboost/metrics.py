from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mboost.api.glmboost import GLMBoostModel
from mboost.families.gaussian import Gaussian


@dataclass
class AICResult:
    value: float
    mstop: int
    df: float
    aic_path: np.ndarray
    df_path: np.ndarray
    corrected: bool

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return (
            f"{self.value:.6f}\n"
            f"Optimal number of boosting iterations: {self.mstop}\n"
            f"Degrees of freedom (for mstop = {self.mstop}): {self.df:.6f}"
        )


def _smoother_matrix(model: GLMBoostModel, learner_idx: int) -> np.ndarray:
    learner = model.prepared_learners[learner_idx]
    x = learner.design
    w = model.sample_weights
    penalty = learner.penalty_matrix
    lhs = np.einsum("ni,n,nj->ij", x, w, x) + penalty
    rhs = (x.T * w).astype(np.float64)
    coef_map = np.linalg.solve(lhs, rhs)
    return np.einsum("ik,kj->ij", x, coef_map)


def hatvalues(model: GLMBoostModel) -> np.ndarray:
    if not isinstance(model.family, Gaussian):
        raise NotImplementedError("hatvalues are currently implemented only for Gaussian models")

    n = model.y.shape[0]
    hat = np.zeros((n, n), dtype=np.float64)
    trace_path = np.zeros(model.mstop + 1, dtype=np.float64)
    smoothers = {
        idx: _smoother_matrix(model, idx)
        for idx in set(model.path.selected)
    }
    identity = np.eye(n, dtype=np.float64)

    for step, idx in enumerate(model.path.selected, start=1):
        smoother = smoothers[idx]
        hat = hat + model.control.nu * np.einsum("ik,kj->ij", smoother, identity - hat)
        trace_path[step] = float(np.trace(hat))

    diag = np.diag(hat).copy()
    diag.setflags(write=False)
    return diag, trace_path


def AIC(model: GLMBoostModel, method: str = "corrected") -> AICResult:
    if method != "corrected":
        raise NotImplementedError("only corrected AIC is currently implemented")
    if not isinstance(model.family, Gaussian):
        raise NotImplementedError("corrected AIC is currently implemented only for Gaussian models")

    _, trace_path = hatvalues(model)
    sumw = float(np.sum(model.sample_weights))
    risk_path = model.risk_[1:]
    df_path = trace_path[1:]
    aic_path = np.log(risk_path / sumw) + (1.0 + df_path / sumw) / (1.0 - (df_path + 2.0) / sumw)
    best_idx = int(np.argmin(aic_path))
    best_mstop = best_idx + 1
    return AICResult(
        value=float(aic_path[best_idx]),
        mstop=best_mstop,
        df=float(df_path[best_idx]),
        aic_path=aic_path.copy(),
        df_path=df_path.copy(),
        corrected=True,
    )
