from __future__ import annotations

import numba as nb
import numpy as np

from .base import Family


def _weighted_quantile(y: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    weights = np.asarray(weights, dtype=np.float64)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must sum to a positive value")

    rounded = np.rint(weights).astype(np.int64)
    if np.allclose(weights, rounded, atol=1e-12, rtol=0.0):
        replicated = np.repeat(np.asarray(y, dtype=np.float64), rounded)
        if replicated.size == 0:
            raise ValueError("weights must contain at least one positive entry")
        return float(np.quantile(replicated, q, method="linear"))

    order = np.argsort(y, kind="mergesort")
    y_sorted = np.asarray(y[order], dtype=np.float64)
    w_sorted = np.asarray(weights[order], dtype=np.float64)
    cumulative = np.cumsum(w_sorted) - 0.5 * w_sorted
    cumulative /= float(np.sum(w_sorted))
    return float(np.interp(q, cumulative, y_sorted))


@nb.njit(cache=True)
def quantile_negative_gradient_numba(y: np.ndarray, f: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        if y[i] - f[i] >= 0.0:
            out[i] = tau
        else:
            out[i] = -(1.0 - tau)
    return out


@nb.njit(cache=True)
def quantile_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray, tau: float) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        diff = y[i] - f[i]
        if diff >= 0.0:
            total += weights[i] * tau * diff
        else:
            total += weights[i] * (tau - 1.0) * diff
    return total


class Quantile(Family):
    def __init__(self, tau: float = 0.5, qoffset: float = 0.5):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be strictly between 0 and 1")
        if not 0.0 <= qoffset <= 1.0:
            raise ValueError("qoffset must be between 0 and 1")
        self.tau = float(tau)
        self.qoffset = float(qoffset)

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        return _weighted_quantile(y, weights, self.qoffset)

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return quantile_negative_gradient_numba(y, f, self.tau)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(quantile_risk_numba(y, f, weights, self.tau))
