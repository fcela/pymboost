from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from mboost.api.glmboost import GLMBoostModel, cvrisk, glmboost, mstop
from mboost.baselearners.base import PreparedBaseLearner
from mboost.data import get_raw_column, to_formulaic_data
from mboost.core.cv import cv
from mboost.families.gaussian import Gaussian
from mboost.metrics import hatvalues


@dataclass(frozen=True)
class ConfIntResult:
    data: Any
    kind: str
    level: float
    term: str | None = None
    approximate: bool = True
    method: str = "normal"

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.data)


def _learner_feature_names(model: GLMBoostModel, learner_idx: int) -> tuple[str, ...]:
    learner = model.prepared_learners[learner_idx]
    if learner.feature_names is not None:
        return learner.feature_names
    return (learner.name,)


def _grid_for_feature(raw, *, grid_size: int):
    values = np.asarray(raw)
    if values.dtype.kind in {"O", "U", "S"} or not np.issubdtype(values.dtype, np.number):
        levels = np.unique(values[~np.equal(values, None)])
        return np.asarray(levels)
    return np.linspace(float(np.min(values)), float(np.max(values)), grid_size, dtype=np.float64)


def _repeat_value(value, n_rows: int):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.repeat(arr.reshape(1), n_rows)
    if arr.shape[0] == n_rows:
        return arr.copy()
    return np.repeat(arr[:1], n_rows)


def _training_data_dict(model: GLMBoostModel) -> dict[str, np.ndarray]:
    names = {model.response_name}
    for learner in model.prepared_learners:
        if learner.by_name is not None:
            names.add(learner.by_name)
        if learner.feature_names is not None:
            names.update(learner.feature_names)
        elif learner.formula_term is None and "," not in learner.name:
            names.add(learner.name)
    return {name: np.asarray(get_raw_column(model.data, name)) for name in names}


def _critical_value(level: float) -> float:
    alpha = 1.0 - float(level)
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _residual_variance(model: GLMBoostModel) -> float:
    _, trace_path = hatvalues(model)
    edf = float(trace_path[-1])
    denom = max(float(np.sum(model.sample_weights)) - edf, 1.0)
    return float(model.risk_[-1] / denom)


def _penalized_covariance(learner: PreparedBaseLearner, sigma2: float) -> np.ndarray:
    x = learner.design
    penalty = learner.penalty_matrix
    xtwx = np.einsum("ni,n,nj->ij", x, np.ones(x.shape[0], dtype=np.float64), x)
    if learner.penalized_gram is not None:
        lhs = learner.penalized_gram
        xtwx = lhs - penalty
    else:
        lhs = xtwx + penalty
    inv_lhs = np.linalg.inv(lhs)
    cov = sigma2 * (inv_lhs @ xtwx @ inv_lhs)
    return np.asarray(cov, dtype=np.float64)


def _learner_design_for_data(learner: PreparedBaseLearner, data) -> np.ndarray:
    return learner.transform(data)


def _resolve_confint_indices(
    model: GLMBoostModel,
    which: int | str | list[int] | list[str] | None,
) -> list[int]:
    if which is None:
        return list(range(len(model.prepared_learners)))
    values = which if isinstance(which, list) else [which]
    indices = []
    for value in values:
        if isinstance(value, int):
            indices.append(value)
        else:
            indices.append(model.term_labels.index(value))
    return indices


def _default_newdata_for_partial(model: GLMBoostModel, learner_idx: int, *, grid_size: int) -> tuple[object, np.ndarray]:
    learner = model.prepared_learners[learner_idx]
    feature_names = _learner_feature_names(model, learner_idx)
    if len(feature_names) != 1:
        raise NotImplementedError("bootstrap confint currently supports only one-dimensional learner effects")
    training = _training_data_dict(model)
    feature = feature_names[0]
    raw = np.asarray(get_raw_column(model.data, feature))
    x_values = _grid_for_feature(raw, grid_size=grid_size)
    plot_data = {name: value.copy() for name, value in training.items()}
    plot_data[feature] = x_values
    n_rows = x_values.shape[0]
    for key in list(plot_data):
        if key == feature:
            continue
        plot_data[key] = _repeat_value(plot_data[key], n_rows)
    return plot_data, x_values


def _bootstrap_weights(n_samples: int, *, B: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    weights = np.zeros((n_samples, B), dtype=np.float64)
    for b in range(B):
        draw = rng.choice(n_samples, size=n_samples, replace=True)
        weights[:, b] = np.bincount(draw, minlength=n_samples).astype(np.float64)
    return weights


def confint(
    model: GLMBoostModel,
    *,
    which: int | str | list[int] | list[str] | None = None,
    level: float = 0.95,
    grid_size: int = 100,
    method: str = "normal",
    B: int = 1000,
    B_mstop: int = 0,
    newdata=None,
    random_state: int = 0,
    bootstrap_weights: np.ndarray | None = None,
    inner_bootstrap_weights: np.ndarray | None = None,
    cvrisk_options: dict[str, object] | None = None,
):
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    if method not in {"normal", "bootstrap"}:
        raise ValueError("method must be either 'normal' or 'bootstrap'")
    if method == "normal" and not isinstance(model.family, Gaussian):
        raise NotImplementedError("normal-approximation confint is currently implemented only for Gaussian models")
    if B <= 0:
        raise ValueError("B must be positive")
    if B_mstop < 0:
        raise ValueError("B_mstop must be non-negative")
    if cvrisk_options is None:
        cvrisk_options = {}

    z_value = _critical_value(level)

    if method == "normal":
        sigma2 = _residual_variance(model)
        if which is None:
            hat_diag, _ = hatvalues(model)
            se = np.sqrt(np.maximum(sigma2 * hat_diag, 0.0))
            fitted = model.fitted_
            return ConfIntResult(
                data={
                    "observation": np.arange(fitted.shape[0], dtype=int),
                    "estimate": fitted,
                    "std_error": se,
                    "lower": fitted - z_value * se,
                    "upper": fitted + z_value * se,
                },
                kind="fitted",
                level=level,
                approximate=True,
                method=method,
            )

        rows: list[dict[str, object]] = []
        training = _training_data_dict(model)
        for idx in _resolve_confint_indices(model, which):
            learner = model.prepared_learners[idx]
            label = model.term_labels[idx]
            if learner.kind == "tree":
                raise NotImplementedError("confint is not implemented for tree learners")
            feature_names = _learner_feature_names(model, idx)
            if len(feature_names) != 1:
                raise NotImplementedError("confint currently supports only one-dimensional learner effects")
            coef = model.coefficients_[label]
            if isinstance(coef, list):
                raise NotImplementedError("confint is not implemented for list-based learner coefficients")
            cov = _penalized_covariance(learner, sigma2)
            feature = feature_names[0]
            raw = np.asarray(get_raw_column(model.data, feature))
            x_values = _grid_for_feature(raw, grid_size=grid_size)
            plot_data = {name: value.copy() for name, value in training.items()}
            plot_data[feature] = x_values
            n_rows = x_values.shape[0]
            for key in list(plot_data):
                if key == feature:
                    continue
                plot_data[key] = _repeat_value(plot_data[key], n_rows)
            design = _learner_design_for_data(learner, plot_data)
            estimate = np.asarray(np.einsum("ij,j->i", design, coef), dtype=np.float64)
            variance = np.einsum("ij,jk,ik->i", design, cov, design)
            se = np.sqrt(np.maximum(variance, 0.0))
            if x_values.dtype.kind in {"O", "U", "S"} or not np.issubdtype(x_values.dtype, np.number):
                x_out = [str(value) for value in x_values]
                kind = "categorical"
            else:
                x_out = np.asarray(x_values, dtype=np.float64)
                kind = "numeric"
            for x_value, fit_value, se_value in zip(x_out, estimate, se):
                rows.append(
                    {
                        "term": label,
                        "feature": feature,
                        "x": x_value,
                        "estimate": float(fit_value),
                        "std_error": float(se_value),
                        "lower": float(fit_value - z_value * se_value),
                        "upper": float(fit_value + z_value * se_value),
                        "kind": kind,
                    }
                )
        return ConfIntResult(
            data=rows,
            kind="partial",
            level=level,
            approximate=True,
            method=method,
        )

    alpha = 1.0 - level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    if bootstrap_weights is not None:
        boot_weights = np.asarray(bootstrap_weights, dtype=np.float64)
        if boot_weights.ndim != 2 or boot_weights.shape[0] != model.y.shape[0]:
            raise ValueError("bootstrap_weights must have shape (n_samples, B)")
        B = int(boot_weights.shape[1])
    else:
        boot_weights = _bootstrap_weights(model.y.shape[0], B=B, random_state=random_state)
    if inner_bootstrap_weights is not None:
        inner_boot = np.asarray(inner_bootstrap_weights, dtype=np.float64)
        if inner_boot.ndim != 3 or inner_boot.shape[0] != model.y.shape[0]:
            raise ValueError("inner_bootstrap_weights must have shape (n_samples, B_mstop, B)")
        if inner_boot.shape[2] != B:
            raise ValueError("inner_bootstrap_weights third dimension must match the number of outer bootstrap replicates")
        if B_mstop > 0 and inner_boot.shape[1] != B_mstop:
            raise ValueError("inner_bootstrap_weights second dimension must equal B_mstop")
    else:
        inner_boot = None

    if which is None:
        target_data = model.data if newdata is None else newdata
        estimate = model.predict(newdata=None if newdata is None else target_data)
        boot_pred = np.empty((B, estimate.shape[0]), dtype=np.float64)
        for b in range(B):
            refit = glmboost(
                model.formula,
                data=model.data,
                family=model.family,
                control=model.control,
                weights=boot_weights[:, b],
            )
            if B_mstop > 0:
                inner_folds = (
                    inner_boot[:, :, b]
                    if inner_boot is not None
                    else cv(refit.y.shape[0], type="bootstrap", B=B_mstop, random_state=random_state + b + 1)
                )
                inner_cv = cvrisk(
                    model.formula,
                    data=model.data,
                    family=model.family,
                    control=model.control,
                    weights=boot_weights[:, b],
                    folds=inner_folds,
                    **cvrisk_options,
                )
                refit = mstop(refit, inner_cv.best_mstop)
            boot_pred[b] = refit.predict(newdata=None if newdata is None else target_data)
        return ConfIntResult(
            data={
                "observation": np.arange(estimate.shape[0], dtype=int),
                "estimate": estimate,
                "lower": np.quantile(boot_pred, lower_q, axis=0),
                "upper": np.quantile(boot_pred, upper_q, axis=0),
                "std_error": np.std(boot_pred, axis=0, ddof=1),
            },
            kind="fitted",
            level=level,
            approximate=False,
            method=method,
        )

    rows: list[dict[str, object]] = []
    for idx in _resolve_confint_indices(model, which):
        learner = model.prepared_learners[idx]
        label = model.term_labels[idx]
        if learner.kind == "tree":
            raise NotImplementedError("bootstrap confint is not implemented for tree learners")
        coef = model.coefficients_[label]
        if isinstance(coef, list):
            raise NotImplementedError("bootstrap confint is not implemented for list-based learner coefficients")
        feature_names = _learner_feature_names(model, idx)
        if len(feature_names) != 1:
            raise NotImplementedError("bootstrap confint currently supports only one-dimensional learner effects")
        target_data, x_values = _default_newdata_for_partial(model, idx, grid_size=grid_size) if newdata is None else (newdata, None)
        estimate = np.asarray(np.einsum("ij,j->i", learner.transform(target_data), coef), dtype=np.float64)
        boot_pred = np.empty((B, estimate.shape[0]), dtype=np.float64)
        for b in range(B):
            refit = glmboost(
                model.formula,
                data=model.data,
                family=model.family,
                control=model.control,
                weights=boot_weights[:, b],
            )
            if B_mstop > 0:
                inner_folds = (
                    inner_boot[:, :, b]
                    if inner_boot is not None
                    else cv(refit.y.shape[0], type="bootstrap", B=B_mstop, random_state=random_state + b + 1)
                )
                inner_cv = cvrisk(
                    model.formula,
                    data=model.data,
                    family=model.family,
                    control=model.control,
                    weights=boot_weights[:, b],
                    folds=inner_folds,
                    **cvrisk_options,
                )
                refit = mstop(refit, inner_cv.best_mstop)
            refit_coef = refit.coefficients_[label]
            if isinstance(refit_coef, list):
                raise NotImplementedError("bootstrap confint is not implemented for list-based learner coefficients")
            refit_learner = refit.prepared_learners[idx]
            boot_pred[b] = np.asarray(np.einsum("ij,j->i", refit_learner.transform(target_data), refit_coef), dtype=np.float64)

        if x_values is None:
            raw = np.asarray(get_raw_column(target_data, feature_names[0]))
            x_values = raw
        if np.asarray(x_values).dtype.kind in {"O", "U", "S"} or not np.issubdtype(np.asarray(x_values).dtype, np.number):
            x_out = [str(value) for value in x_values]
            kind = "categorical"
        else:
            x_out = np.asarray(x_values, dtype=np.float64)
            kind = "numeric"
        lower = np.quantile(boot_pred, lower_q, axis=0)
        upper = np.quantile(boot_pred, upper_q, axis=0)
        se = np.std(boot_pred, axis=0, ddof=1)
        for x_value, fit_value, se_value, lo, hi in zip(x_out, estimate, se, lower, upper):
            rows.append(
                {
                    "term": label,
                    "feature": feature_names[0],
                    "x": x_value,
                    "estimate": float(fit_value),
                    "std_error": float(se_value),
                    "lower": float(lo),
                    "upper": float(hi),
                    "kind": kind,
                }
            )
    return ConfIntResult(
        data=rows,
        kind="partial",
        level=level,
        approximate=False,
        method=method,
    )
