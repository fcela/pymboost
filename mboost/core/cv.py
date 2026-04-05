from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CVRiskResult:
    risk: np.ndarray
    fold_risk: np.ndarray
    folds: np.ndarray

    @property
    def best_mstop(self) -> int:
        return int(np.argmin(self.risk))


def cv(
    n_samples: int | np.ndarray,
    *,
    folds: int | None = None,
    type: str | None = None,
    B: int | None = None,
    fraction: float = 0.5,
    prob: float | None = None,
    shuffle: bool = False,
    random_state: int = 0,
    strata=None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    sample_fraction = fraction if prob is None else prob

    if np.isscalar(n_samples):
        n_obs = int(n_samples)
        resolved_type = "kfold" if type is None else type
        if resolved_type == "kfold":
            resolved_folds = 5 if folds is None else int(folds)
            if resolved_folds <= 1:
                raise ValueError("folds must be greater than 1")
            if n_obs < resolved_folds:
                raise ValueError("n_samples must be at least as large as folds")
            assignments = np.arange(n_obs) % resolved_folds
            if shuffle:
                rng.shuffle(assignments)
            return assignments
        if resolved_type not in {"subsampling", "bootstrap"}:
            raise ValueError("type must be one of {'kfold', 'subsampling', 'bootstrap'}")
        resolved_B = int(folds) if B is None and folds is not None else B
        if resolved_B is None:
            resolved_B = 5
        if resolved_B <= 0:
            raise ValueError("B must be positive")
        if not 0.0 < sample_fraction < 1.0 and resolved_type == "subsampling":
            raise ValueError("fraction must be strictly between 0 and 1 for subsampling")
        masks = np.zeros((n_obs, resolved_B), dtype=np.float64)
        for b in range(resolved_B):
            if resolved_type == "subsampling":
                train_size = max(1, int(np.floor(sample_fraction * n_obs)))
                train_idx = rng.choice(n_obs, size=train_size, replace=False)
                holdout = np.ones(n_obs, dtype=np.float64)
                holdout[train_idx] = 0.0
                masks[:, b] = holdout
            else:
                draw = rng.choice(n_obs, size=n_obs, replace=True)
                in_bag = np.bincount(draw, minlength=n_obs) > 0
                masks[:, b] = (~in_bag).astype(np.float64)
        return masks

    weights = np.asarray(n_samples, dtype=np.float64).reshape(-1)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weights.size == 0:
        raise ValueError("weights must be non-empty")

    resolved_type = "bootstrap" if type is None else type
    if resolved_type not in {"kfold", "subsampling", "bootstrap"}:
        raise ValueError("type must be one of {'kfold', 'subsampling', 'bootstrap'}")
    resolved_B = B if B is not None else (int(folds) if folds is not None else (10 if resolved_type == "kfold" else 25))
    if resolved_B <= 0:
        raise ValueError("B must be positive")
    if not 0.0 < sample_fraction < 1.0 and resolved_type == "subsampling":
        raise ValueError("prob must be strictly between 0 and 1 for subsampling")

    if strata is None:
        strata_arr = np.zeros(weights.shape[0], dtype=np.int64)
    else:
        strata_arr = np.asarray(strata)
        if strata_arr.shape[0] != weights.shape[0]:
            raise ValueError("strata must have the same length as weights")

    out = np.zeros((weights.shape[0], resolved_B), dtype=np.float64)

    for stratum in np.unique(strata_arr):
        idx = np.where(strata_arr == stratum)[0]
        local_weights = np.asarray(weights[idx], dtype=np.float64)
        if resolved_type == "kfold":
            if idx.shape[0] < resolved_B:
                raise ValueError("stratum size must be at least as large as B for kfold")
            assignments = np.arange(idx.shape[0]) % resolved_B
            if shuffle:
                rng.shuffle(assignments)
            local = np.tile(local_weights[:, None], (1, resolved_B))
            for fold_id in range(resolved_B):
                local[assignments == fold_id, fold_id] = 0.0
        elif resolved_type == "bootstrap":
            total = float(np.sum(local_weights))
            if total <= 0.0:
                continue
            probs = local_weights / total
            local = np.zeros((idx.shape[0], resolved_B), dtype=np.float64)
            for b in range(resolved_B):
                draw = rng.choice(idx.shape[0], size=idx.shape[0], replace=True, p=probs)
                local[:, b] = np.bincount(draw, minlength=idx.shape[0]).astype(np.float64)
        else:
            total = float(np.sum(local_weights))
            if total <= 0.0:
                continue
            probs = local_weights / total
            train_size = max(1, int(np.floor(sample_fraction * idx.shape[0])))
            local = np.zeros((idx.shape[0], resolved_B), dtype=np.float64)
            for b in range(resolved_B):
                draw = rng.choice(idx.shape[0], size=train_size, replace=False, p=probs)
                local[draw, b] = local_weights[draw]
        out[idx, :] = local

    return out
