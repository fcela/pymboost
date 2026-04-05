from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from sklearn.tree import DecisionTreeRegressor

from mboost.baselearners.base import PreparedBaseLearner, solve_constrained_quadratic
from mboost.core.engine_numba import componentwise_linear_fit_numba

from .control import BoostControl


@dataclass
class BoostingPath:
    offset: float
    selected: list[int]
    coefficients: list[object]
    risk: np.ndarray
    fitted: np.ndarray


def _fit_prepared_learner(
    learner: PreparedBaseLearner,
    u: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = learner.design
    penalty = learner.penalty_matrix
    if learner.kind == "tree":
        by_values = learner.tree_by_values
        fit_mask = weights > 0.0
        if by_values is not None:
            fit_mask = fit_mask & (by_values > 0.0)
        if not np.any(fit_mask):
            pred = np.zeros(x.shape[0], dtype=np.float64)
            err = float(np.dot(weights, (u - pred) ** 2))
            tree = DecisionTreeRegressor(
                max_depth=1 if learner.maxdepth is None else learner.maxdepth,
                min_samples_split=10 if learner.minsplit is None else learner.minsplit,
                min_samples_leaf=4 if learner.minbucket is None else learner.minbucket,
                random_state=0,
            )
            return tree, pred, err
        tree = DecisionTreeRegressor(
            max_depth=1 if learner.maxdepth is None else learner.maxdepth,
            min_samples_split=10 if learner.minsplit is None else learner.minsplit,
            min_samples_leaf=4 if learner.minbucket is None else learner.minbucket,
            random_state=0,
        )
        tree.fit(x[fit_mask], u[fit_mask], sample_weight=weights[fit_mask])
        pred = tree.predict(x)
        if by_values is not None:
            pred = pred * by_values
        err = float(np.dot(weights, (u - pred) ** 2))
        return tree, pred, err
    if x.shape[1] == 1:
        column = x[:, 0]
        denom = learner.scalar_denom
        if denom is None:
            denom = float(np.dot(weights, column * column) + penalty[0, 0])
        numer = float(np.dot(learner.weighted_design_t[0], u)) if learner.weighted_design_t is not None else float(np.dot(weights, column * u))
        beta = np.array([numer / denom if denom > 0.0 else 0.0], dtype=np.float64)
        pred = column * beta[0]
    else:
        lhs = learner.penalized_gram if learner.penalized_gram is not None else (np.einsum("ni,n,nj->ij", x, weights, x) + penalty)
        rhs = np.einsum("in,n->i", learner.weighted_design_t, u) if learner.weighted_design_t is not None else np.einsum("ni,n->i", x, weights * u)
        if learner.kind == "mono_spline":
            assert learner.constraint_matrix is not None
            assert learner.constraint_lb is not None
            assert learner.constraint_ub is not None
            beta = solve_constrained_quadratic(
                lhs,
                rhs,
                constraint_matrix=learner.constraint_matrix,
                constraint_lb=learner.constraint_lb,
                constraint_ub=learner.constraint_ub,
                method="quad.prog" if learner.solver_type is None else learner.solver_type,
                niter=learner.niter,
            )
        else:
            beta = cho_solve(learner.penalized_factor, rhs, check_finite=False) if learner.penalized_factor is not None else np.linalg.solve(lhs, rhs)
        pred = np.einsum("ij,j->i", x, beta)
    err = float(np.dot(weights, (u - pred) ** 2))
    return beta, pred, err


def apply_componentwise_path(
    *,
    learners: list[PreparedBaseLearner],
    offset: float,
    selected: list[int],
    coefficients: list[np.ndarray],
    nu: float,
    mstop: int | None = None,
) -> np.ndarray:
    n_steps = len(selected) if mstop is None else mstop
    fitted = np.full(learners[0].design.shape[0], offset, dtype=np.float64)
    for step in range(n_steps):
        idx = selected[step]
        if learners[idx].kind == "tree":
            fitted = fitted + nu * coefficients[step].predict(learners[idx].design)
        else:
            fitted = fitted + nu * np.einsum("ij,j->i", learners[idx].design, coefficients[step])
    return fitted


def evaluate_empirical_risk_path(
    *,
    learners: list[PreparedBaseLearner],
    offset: float,
    selected: list[int],
    coefficients: list[np.ndarray],
    nu: float,
    y: np.ndarray,
    weights: np.ndarray,
    family,
) -> np.ndarray:
    fitted = np.full(learners[0].design.shape[0], offset, dtype=np.float64)
    risk = np.empty(len(selected) + 1, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    family.calibrate(y, fitted, weights)
    risk[0] = family.risk(y, fitted, weights) / weight_sum
    for step, idx in enumerate(selected):
        if learners[idx].kind == "tree":
            fitted = fitted + nu * coefficients[step].predict(learners[idx].design)
        else:
            fitted = fitted + nu * np.einsum("ij,j->i", learners[idx].design, coefficients[step])
        family.calibrate(y, fitted, weights)
        risk[step + 1] = family.risk(y, fitted, weights) / weight_sum
    return risk


def fit_componentwise_model(
    *,
    learners: list[PreparedBaseLearner],
    y: np.ndarray,
    family,
    control: BoostControl,
    weights: np.ndarray,
) -> BoostingPath:
    n_samples = y.shape[0]
    offset = float(family.offset(y, weights))
    fitted = np.full(n_samples, offset, dtype=np.float64)
    risk = np.empty(control.mstop + 1, dtype=np.float64)
    family.calibrate(y, fitted, weights)
    risk[0] = family.risk(y, fitted, weights)

    scalar_indices = [
        idx
        for idx, learner in enumerate(learners)
        if learner.kind != "tree" and learner.kind != "mono_spline" and learner.design.shape[1] == 1
    ]
    scalar_matrix = None
    scalar_penalties = None
    if scalar_indices:
        scalar_matrix = np.ascontiguousarray(
            np.column_stack([learners[idx].design[:, 0] for idx in scalar_indices]),
            dtype=np.float64,
        )
        scalar_penalties = np.ascontiguousarray(
            np.array([learners[idx].penalty_matrix[0, 0] for idx in scalar_indices], dtype=np.float64)
        )
    block_indices = [idx for idx in range(len(learners)) if idx not in scalar_indices]

    selected: list[int] = []
    coefficients: list[np.ndarray] = []

    for step in range(control.mstop):
        family.calibrate(y, fitted, weights)
        u = family.negative_gradient(y, fitted)
        best_idx = -1
        best_coef = None
        best_pred = None
        best_err = np.inf

        if scalar_matrix is not None and scalar_penalties is not None:
            scalar_betas, scalar_sse = componentwise_linear_fit_numba(
                scalar_matrix,
                u,
                weights,
                scalar_penalties,
            )
            best_scalar = int(np.argmin(scalar_sse))
            best_err = float(scalar_sse[best_scalar])
            best_idx = scalar_indices[best_scalar]
            best_coef = np.array([scalar_betas[best_scalar]], dtype=np.float64)
            best_pred = scalar_matrix[:, best_scalar] * best_coef[0]

        for idx in block_indices:
            learner = learners[idx]
            coef, pred, err = _fit_prepared_learner(learner, u, weights)
            if err < best_err:
                best_idx = idx
                best_coef = coef
                best_pred = pred
                best_err = err
        assert best_coef is not None
        assert best_pred is not None
        fitted = fitted + control.nu * best_pred
        selected.append(best_idx)
        coefficients.append(best_coef)
        family.calibrate(y, fitted, weights)
        risk[step + 1] = family.risk(y, fitted, weights)

    return BoostingPath(
        offset=offset,
        selected=selected,
        coefficients=coefficients,
        risk=risk,
        fitted=fitted,
    )
