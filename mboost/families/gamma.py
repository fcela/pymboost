from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.optimize import minimize_scalar

from .base import Family


@nb.njit(cache=True)
def gamma_negative_gradient_numba(y: np.ndarray, f: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = sigma * y[i] * np.exp(-f[i]) - sigma
    return out


@nb.njit(cache=True)
def gamma_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray, sigma: float) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        yi = y[i]
        log_y = math.log(yi) if yi > 0.0 else -np.inf
        total += weights[i] * (
            math.lgamma(sigma)
            + sigma * yi * np.exp(-f[i])
            - sigma * log_y
            - sigma * math.log(sigma)
            + sigma * f[i]
        )
    return total


class GammaReg(Family):
    def __init__(self):
        self.sigma = 1.0

    def response(self, f: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(f, dtype=np.float64))

    def _check_y(self, y: np.ndarray) -> None:
        if np.any(y < 0.0):
            raise ValueError("response is not positive but 'family = GammaReg()'")

    def _sigma_risk(self, sigma: float, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(gamma_risk_numba(y, f, weights, sigma))

    def calibrate(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> None:
        self._check_y(y)
        result = minimize_scalar(
            lambda sigma: self._sigma_risk(float(sigma), y, f, weights),
            bounds=(0.0, 100.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        self.sigma = float(result.x)

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        self._check_y(y)
        upper = float(np.max(y * y)) if y.size else 0.0
        result = minimize_scalar(
            lambda value: self._sigma_risk(self.sigma, y, np.full_like(y, value), weights),
            bounds=(0.0, upper),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(result.x)

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        self._check_y(y)
        return gamma_negative_gradient_numba(y, f, self.sigma)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        self._check_y(y)
        return float(gamma_risk_numba(y, f, weights, self.sigma))
