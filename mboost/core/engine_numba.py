from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def weighted_mean_numba(y: np.ndarray, weights: np.ndarray) -> float:
    total_weight = 0.0
    total_value = 0.0
    for i in range(y.shape[0]):
        total_weight += weights[i]
        total_value += weights[i] * y[i]
    return total_value / total_weight


@nb.njit(cache=True)
def componentwise_linear_fit_numba(
    x: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray,
    penalties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_learners = x.shape
    betas = np.empty(n_learners, dtype=np.float64)
    sse = np.empty(n_learners, dtype=np.float64)

    for j in range(n_learners):
        numer = 0.0
        denom = penalties[j]
        for i in range(n_samples):
            w = weights[i]
            xij = x[i, j]
            numer += w * xij * u[i]
            denom += w * xij * xij

        beta = 0.0
        if denom > 0.0:
            beta = numer / denom
        betas[j] = beta

        err = 0.0
        for i in range(n_samples):
            resid = u[i] - x[i, j] * beta
            err += weights[i] * resid * resid
        sse[j] = err

    return betas, sse


@nb.njit(cache=True)
def linear_predictor_update_numba(
    current: np.ndarray,
    x_column: np.ndarray,
    beta: float,
    nu: float,
) -> np.ndarray:
    updated = np.empty_like(current)
    for i in range(current.shape[0]):
        updated[i] = current[i] + nu * x_column[i] * beta
    return updated
