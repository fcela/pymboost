from __future__ import annotations

import numba as nb
import numpy as np

from mboost.core.engine_numba import weighted_mean_numba

from .base import Family


@nb.njit(cache=True)
def binomial_negative_gradient_numba(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        y_pm = 2.0 * y[i] - 1.0
        margin = 2.0 * y_pm * f[i]
        if margin >= 0.0:
            exp_neg_margin = np.exp(-margin)
            out[i] = 2.0 * y_pm * exp_neg_margin / (np.log(2.0) * (1.0 + exp_neg_margin))
        else:
            exp_margin = np.exp(margin)
            out[i] = 2.0 * y_pm / (np.log(2.0) * (1.0 + exp_margin))
    return out


@nb.njit(cache=True)
def binomial_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        f_clip = min(max(f[i], -36.0), 36.0)
        exp_pos = np.exp(f_clip)
        exp_neg = np.exp(-f_clip)
        p = exp_pos / (exp_pos + exp_neg)
        y01 = y[i]
        total += weights[i] * (-y01 * np.log(p) - (1.0 - y01) * np.log(1.0 - p))
    return total


class Binomial(Family):
    def response(self, f: np.ndarray) -> np.ndarray:
        f_arr = np.asarray(f, dtype=np.float64)
        margin = np.clip(2.0 * f_arr, -36.0, 36.0)
        return 1.0 / (1.0 + np.exp(-margin))

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        mean = float(weighted_mean_numba(y, weights))
        mean = min(max(mean, 1e-12), 1.0 - 1e-12)
        return float(0.5 * np.log(mean / (1.0 - mean)))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return binomial_negative_gradient_numba(y, f)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(binomial_risk_numba(y, f, weights))
