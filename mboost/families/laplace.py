from __future__ import annotations

import numba as nb
import numpy as np
from scipy.optimize import minimize_scalar

from .base import Family


def _laplace_offset(y: np.ndarray, weights: np.ndarray) -> float:
    lower = float(np.min(y))
    upper = float(np.max(y))

    def objective(value: float) -> float:
        return float(np.sum(weights * np.abs(y - value)))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    return float(result.x)


@nb.njit(cache=True)
def laplace_negative_gradient_numba(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        diff = y[i] - f[i]
        if diff > 0.0:
            out[i] = 1.0
        elif diff < 0.0:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out


@nb.njit(cache=True)
def laplace_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        total += weights[i] * abs(y[i] - f[i])
    return total


class Laplace(Family):
    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        return _laplace_offset(y, weights)

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return laplace_negative_gradient_numba(y, f)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(laplace_risk_numba(y, f, weights))
