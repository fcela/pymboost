from __future__ import annotations

import numba as nb
import numpy as np

from mboost.core.engine_numba import weighted_mean_numba

from .base import Family


@nb.njit(cache=True)
def gaussian_negative_gradient_numba(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = y[i] - f[i]
    return out


@nb.njit(cache=True)
def gaussian_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        resid = y[i] - f[i]
        total += weights[i] * resid * resid
    return total


class Gaussian(Family):
    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        return float(weighted_mean_numba(y, weights))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return gaussian_negative_gradient_numba(y, f)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(gaussian_risk_numba(y, f, weights))
