from __future__ import annotations

import numba as nb
import numpy as np

from mboost.core.engine_numba import weighted_mean_numba

from .base import Family


@nb.njit(cache=True)
def expectile_negative_gradient_numba(y: np.ndarray, f: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        diff = y[i] - f[i]
        if diff >= 0.0:
            out[i] = 2.0 * tau * diff
        else:
            out[i] = 2.0 * (1.0 - tau) * diff
    return out


@nb.njit(cache=True)
def expectile_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray, tau: float) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        diff = y[i] - f[i]
        if diff >= 0.0:
            total += weights[i] * tau * diff * diff
        else:
            total += weights[i] * (1.0 - tau) * diff * diff
    return total


class Expectile(Family):
    def __init__(self, tau: float = 0.5):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be strictly between 0 and 1")
        self.tau = float(tau)

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        return float(weighted_mean_numba(y, weights))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return expectile_negative_gradient_numba(y, f, self.tau)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(expectile_risk_numba(y, f, weights, self.tau))
