from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Family(ABC):
    def calibrate(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> None:
        return None

    def response(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(f, dtype=np.float64)

    @abstractmethod
    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        raise NotImplementedError
