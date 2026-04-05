from __future__ import annotations

import numba as nb
import numpy as np
from scipy.optimize import minimize_scalar

from .base import Family


def _resolve_huber_delta(y: np.ndarray, reference_fit: np.ndarray, delta: float | None) -> float:
    if delta is not None:
        return float(delta)
    resolved = float(np.median(np.abs(y - reference_fit)))
    return max(resolved, 1e-12)


def _huber_offset(
    y: np.ndarray,
    weights: np.ndarray,
    delta: float | None,
    reference_fit: np.ndarray,
) -> float:
    lower = float(np.min(y))
    upper = float(np.max(y))
    local_delta = _resolve_huber_delta(y, reference_fit, delta)

    def objective(value: float) -> float:
        resid = np.abs(y - value)
        quadratic = resid < local_delta
        loss = np.where(
            quadratic,
            0.5 * resid * resid,
            local_delta * (resid - 0.5 * local_delta),
        )
        return float(np.sum(weights * loss))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(result.x)


@nb.njit(cache=True)
def huber_negative_gradient_numba(y: np.ndarray, f: np.ndarray, delta: float) -> np.ndarray:
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        diff = y[i] - f[i]
        if diff > delta:
            out[i] = delta
        elif diff < -delta:
            out[i] = -delta
        else:
            out[i] = diff
    return out


@nb.njit(cache=True)
def huber_risk_numba(y: np.ndarray, f: np.ndarray, weights: np.ndarray, delta: float) -> float:
    total = 0.0
    for i in range(y.shape[0]):
        resid = abs(y[i] - f[i])
        if resid < delta:
            total += weights[i] * 0.5 * resid * resid
        else:
            total += weights[i] * delta * (resid - 0.5 * delta)
    return total


class Huber(Family):
    def __init__(self, d: float | None = None):
        if d is not None and d <= 0.0:
            raise ValueError("d must be positive when provided")
        self.d = None if d is None else float(d)
        self._reference_fit: np.ndarray | None = None

    def _current_reference(self, y: np.ndarray) -> np.ndarray:
        if self._reference_fit is None or self._reference_fit.shape != y.shape:
            self._reference_fit = np.zeros_like(y)
        return self._reference_fit

    def _delta(self, y: np.ndarray) -> float:
        return _resolve_huber_delta(y, self._current_reference(y), self.d)

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        self._reference_fit = np.zeros_like(y)
        offset = _huber_offset(y, weights, self.d, self._reference_fit)
        self._reference_fit = np.full_like(y, offset)
        return offset

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        delta = self._delta(y)
        self._reference_fit = f.copy()
        return huber_negative_gradient_numba(y, f, delta)

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        return float(huber_risk_numba(y, f, weights, self._delta(y)))
