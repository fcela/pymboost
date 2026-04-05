from __future__ import annotations

import math

import numba as nb
import numpy as np

from mboost.core.engine_numba import weighted_mean_numba

from .base import Family


@nb.njit(cache=True)
def poisson_negative_gradient_numba(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = y[i] - np.exp(f[i])
    return out


@nb.njit(cache=True)
def poisson_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        total += weights[i] * (np.exp(f[i]) - y[i] * f[i] + math.lgamma(y[i] + 1.0))
    return total


class Poisson(Family):
    def response(self, f: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(f, dtype=np.float64))

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        mean = max(float(weighted_mean_numba(y, weights)), 1e-12)
        return float(np.log(mean))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return poisson_negative_gradient_numba(y, f)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(poisson_risk_numba(y, f, weights))
