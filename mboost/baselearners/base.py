from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from formulaic import model_matrix
from scipy.linalg import cho_factor
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, brentq, minimize

from mboost.data import get_column, get_raw_column, to_formulaic_data


def _as_1d_array(data, key: str) -> np.ndarray:
    return get_column(data, key)


def _formulaic_safe_term(term: str) -> str:
    return term.replace("lambda=", "lambda_=")


def _formulaic_matrix_columns(matrix) -> list[str]:
    columns = getattr(matrix, "columns", None)
    if columns is None:
        raise ValueError("formulaic model matrix did not expose column names")
    return [str(column) for column in columns]


def _formulaic_matrix_to_numpy(matrix, *, column_names: list[str] | None = None) -> np.ndarray:
    columns = _formulaic_matrix_columns(matrix)
    selected = columns if column_names is None else column_names
    arrays = [np.asarray(matrix[column], dtype=np.float64).reshape(-1) for column in selected]
    return np.ascontiguousarray(np.column_stack(arrays), dtype=np.float64)


def _factor_levels(values: np.ndarray) -> np.ndarray:
    levels = np.unique(values[~np.equal(values, None)])
    return np.sort(levels)


def _factor_design(values: np.ndarray, *, intercept: bool, levels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values)
    if levels is None:
        levels = _factor_levels(arr)
    if levels.shape[0] == 0:
        raise ValueError("factor base learner must contain at least one observed level")
    dummy_levels = levels[1:]
    dummies = np.column_stack([(arr == level).astype(np.float64) for level in dummy_levels]) if dummy_levels.size else np.empty((arr.shape[0], 0), dtype=np.float64)
    if intercept:
        design = np.column_stack([np.ones(arr.shape[0], dtype=np.float64), dummies])
    else:
        design = dummies
    return np.ascontiguousarray(design, dtype=np.float64), levels


def _factor_full_design(values: np.ndarray, *, levels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values)
    if levels is None:
        levels = _factor_levels(arr)
    if levels.shape[0] == 0:
        raise ValueError("factor base learner must contain at least one observed level")
    design = np.column_stack([(arr == level).astype(np.float64) for level in levels])
    return np.ascontiguousarray(design, dtype=np.float64), levels


def _binary_by_vector(values, *, levels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        unique = np.unique(arr)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("tree by= currently requires values in {0, 1} for numeric modifiers")
        return np.asarray(arr, dtype=np.float64), None
    if levels is None:
        levels = _factor_levels(arr)
    if levels.shape[0] != 2:
        raise ValueError("tree by= factor currently requires exactly two levels")
    mapped = (arr == levels[1]).astype(np.float64)
    return np.ascontiguousarray(mapped, dtype=np.float64), levels


def _monotone_factor_constraint_matrix(
    *,
    n_levels: int,
    intercept: bool,
    constraint: str,
) -> np.ndarray:
    if not intercept:
        raise NotImplementedError("factor bmono currently requires intercept=True")
    means_from_beta = np.zeros((n_levels, n_levels), dtype=np.float64)
    means_from_beta[:, 0] = 1.0
    if n_levels > 1:
        means_from_beta[1:, 1:] = np.eye(n_levels - 1, dtype=np.float64)
    if constraint in {"increasing", "decreasing"}:
        constraint_matrix = np.diff(means_from_beta, n=1, axis=0)
    elif constraint in {"convex", "concave"}:
        constraint_matrix = np.diff(means_from_beta, n=2, axis=0)
    elif constraint in {"positive", "negative"}:
        constraint_matrix = np.eye(n_levels, dtype=np.float64)
        constraint_matrix[0, 0] = 0.0
    else:
        raise ValueError(f"unsupported monotone factor constraint: {constraint}")
    if constraint in {"decreasing", "concave", "negative"}:
        constraint_matrix = -constraint_matrix
    return np.ascontiguousarray(constraint_matrix, dtype=np.float64)


def _weighted_center_matrix(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.average(matrix, axis=0, weights=weights)
    return matrix - offsets, offsets


def _make_psd(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    min_eig = float(np.min(np.linalg.eigvalsh(out))) - np.sqrt(np.finfo(np.float64).eps)
    if min_eig < -1e-10:
        rho = 1.0 / (1.0 - min_eig)
        out = rho * out + (1.0 - rho) * np.eye(out.shape[0], dtype=np.float64)
        return _make_psd(out)
    return out


def _df2lambda(
    design: np.ndarray,
    *,
    df: float | None,
    lambda_value: float | None,
    penalty_matrix: np.ndarray,
    weights: np.ndarray,
    trace_s: bool = False,
    eps: float = 1e-9,
    lambda_max: float = 1e15,
) -> tuple[float, float]:
    if (df is None) == (lambda_value is None):
        raise ValueError("exactly one of df or lambda_value must be provided")

    rank_x = float(np.linalg.matrix_rank(design))
    if df is not None and df >= rank_x:
        return float(df), 0.0
    if lambda_value is not None and lambda_value == 0.0:
        return rank_x, 0.0

    xtx = np.einsum("ni,n,nj->ij", design, weights, design)
    dmat = penalty_matrix
    a = _make_psd(xtx + eps * dmat)
    chol_lower = np.linalg.cholesky(a)
    rm = np.linalg.solve(chol_lower.T, np.eye(a.shape[0], dtype=np.float64))
    d = np.linalg.svd(np.einsum("ki,kl,lj->ij", rm, dmat, rm), compute_uv=False)

    def df_fun(lam: float) -> float:
        vals = 1.0 / (1.0 + lam * d)
        if trace_s:
            return float(np.sum(vals))
        return float(2.0 * np.sum(vals) - np.sum(vals * vals))

    if lambda_value is not None:
        return df_fun(lambda_value), float(lambda_value)

    assert df is not None
    if df >= float(d.shape[0]):
        return float(df), 0.0

    def objective(lam: float) -> float:
        return df_fun(lam) - float(df)

    if objective(lambda_max) > 0.0:
        return float(df), float(lambda_max)

    lam = float(brentq(objective, 0.0, lambda_max, xtol=np.sqrt(np.finfo(np.float64).eps)))
    return float(df), lam


def _build_open_bspline_basis(x: np.ndarray, n_basis: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        raise ValueError("bbs requires a covariate with positive range")

    interior_count = max(n_basis - degree - 1, 0)
    if interior_count > 0:
        interior = np.linspace(xmin, xmax, interior_count + 2, dtype=np.float64)[1:-1]
    else:
        interior = np.empty(0, dtype=np.float64)
    knot_vector = np.concatenate(
        [
            np.full(degree + 1, xmin, dtype=np.float64),
            interior,
            np.full(degree + 1, xmax, dtype=np.float64),
        ]
    )
    basis = BSpline.design_matrix(x, knot_vector, degree, extrapolate=True).toarray()
    return np.ascontiguousarray(basis, dtype=np.float64), knot_vector


def _build_pspline_basis_from_knots(
    x: np.ndarray,
    knots: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        raise ValueError("bbs requires a covariate with positive range")
    spacing = (xmax - xmin) / (knots + 1)
    knot_vector = xmin + spacing * np.arange(-degree, knots + degree + 2, dtype=np.float64)
    basis = BSpline.design_matrix(x, knot_vector, degree, extrapolate=True).toarray()
    return np.ascontiguousarray(basis, dtype=np.float64), knot_vector


def _centered_penalized_spline_basis(
    design: np.ndarray,
    base_penalty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(base_penalty)
    keep = eigvals > 1e-10
    if not np.any(keep):
        raise ValueError("centered spline basis has no penalized subspace")
    scales = 1.0 / np.sqrt(eigvals[keep])
    transform = eigvecs[:, keep] * scales[None, :]
    centered_design = np.dot(np.asarray(design, dtype=np.float64), transform)
    penalty_matrix = np.eye(centered_design.shape[1], dtype=np.float64)
    return (
        np.ascontiguousarray(centered_design, dtype=np.float64),
        np.ascontiguousarray(penalty_matrix, dtype=np.float64),
        np.ascontiguousarray(transform, dtype=np.float64),
    )


@dataclass(frozen=True)
class PreparedBaseLearner:
    name: str
    kind: str
    design: np.ndarray
    penalty_matrix: np.ndarray
    center_offsets: np.ndarray
    feature_names: tuple[str, ...] | None = None
    intercept: bool = False
    factor_levels: np.ndarray | None = None
    target_level: object | None = None
    basis_transform: np.ndarray | None = None
    by_name: str | None = None
    degree: int | None = None
    knot_vector: np.ndarray | None = None
    formula_term: str | None = None
    formula_spec: object | None = None
    formula_column_names: tuple[str, ...] | None = None
    constraint_matrix: np.ndarray | None = None
    constraint_lb: np.ndarray | None = None
    constraint_ub: np.ndarray | None = None
    solver_type: str | None = None
    niter: int | None = None
    maxdepth: int | None = None
    minsplit: int | None = None
    minbucket: int | None = None
    tree_by_values: np.ndarray | None = None
    by_levels: np.ndarray | None = None
    weighted_design_t: np.ndarray | None = None
    penalized_gram: np.ndarray | None = None
    scalar_denom: float | None = None
    penalized_factor: tuple[np.ndarray, bool] | None = None

    def transform(self, data) -> np.ndarray:
        if self.kind == "linear":
            raw = get_raw_column(data, self.name)
            if self.factor_levels is not None:
                matrix, _ = _factor_design(raw, intercept=self.intercept, levels=self.factor_levels)
            else:
                x = np.asarray(raw, dtype=np.float64)
                matrix = np.column_stack([np.ones_like(x), x]) if self.intercept else x[:, None]
        elif self.kind == "random":
            raw = get_raw_column(data, self.name)
            if self.factor_levels is None:
                raise ValueError("random base learner requires factor levels")
            matrix, _ = _factor_full_design(raw, levels=self.factor_levels)
        elif self.kind == "factor_dummy":
            if self.target_level is None:
                raise ValueError("factor dummy learner requires a target level")
            raw = get_raw_column(data, self.name)
            matrix = (np.asarray(raw) == self.target_level).astype(np.float64)[:, None]
        elif self.kind in {"spline", "mono_spline"}:
            raw = get_raw_column(data, self.name)
            if self.factor_levels is not None:
                matrix, _ = _factor_design(raw, intercept=self.intercept, levels=self.factor_levels)
            else:
                x = _as_1d_array(data, self.name)
                assert self.knot_vector is not None
                assert self.degree is not None
                matrix = BSpline.design_matrix(
                    x,
                    self.knot_vector,
                    self.degree,
                    extrapolate=True,
                ).toarray()
                if self.basis_transform is not None:
                    matrix = np.dot(np.asarray(matrix, dtype=np.float64), self.basis_transform)
        elif self.kind == "tree":
            if self.feature_names is None:
                raise ValueError("tree learner requires feature names")
            columns = [get_column(data, feature_name) for feature_name in self.feature_names]
            matrix = np.column_stack(columns)
        elif self.kind == "formula_matrix":
            if self.formula_spec is None:
                raise ValueError("formula_matrix learner requires a model spec")
            matrix = _formulaic_matrix_to_numpy(
                self.formula_spec.get_model_matrix(to_formulaic_data(data)),
                column_names=list(self.formula_column_names) if self.formula_column_names is not None else None,
            )
        else:
            raise ValueError(f"unknown learner kind: {self.kind}")
        matrix = np.asarray(matrix, dtype=np.float64)
        if self.by_name is not None:
            by = get_column(data, self.by_name)
            matrix = matrix * by[:, None]
        if self.center_offsets.size:
            matrix = matrix - self.center_offsets
        return np.ascontiguousarray(matrix, dtype=np.float64)


@dataclass(frozen=True)
class BaseLearnerSpec:
    name: str
    kind: str
    penalty: float | None = 0.0
    center: bool = True
    intercept: bool = False
    feature_names: tuple[str, ...] | None = None
    by: str | None = None
    target_level: object | None = None
    df: int | None = None
    knots: int | None = None
    degree: int = 3
    differences: int = 2
    maxdepth: int = 1
    minsplit: int = 10
    minbucket: int = 4
    formula_term: str | None = None
    formula_column_names: tuple[str, ...] | None = None
    solver_type: str | None = None
    niter: int | None = None

    def prepare(self, data, weights: np.ndarray) -> PreparedBaseLearner:
        if self.kind == "linear":
            raw = get_raw_column(data, self.name)
            factor_levels = None
            df_center_linear = False
            if np.issubdtype(np.asarray(raw).dtype, np.number):
                x = np.asarray(raw, dtype=np.float64)
                design = np.column_stack([np.ones_like(x), x]) if self.intercept else x[:, None]
                df_center_linear = bool(self.intercept and self.df is not None and self.penalty is None)
            else:
                design, factor_levels = _factor_design(raw, intercept=self.intercept)
            if self.by is not None:
                by = get_column(data, self.by)
                design = design * by[:, None]
            center_offsets = np.zeros(design.shape[1], dtype=np.float64)
            if df_center_linear:
                center_offsets[1] = np.average(design[:, 1], weights=weights)
                design = design - center_offsets
            if self.center:
                design, center_offsets = _weighted_center_matrix(design, weights)
            penalty_base = np.eye(design.shape[1], dtype=np.float64)
            if self.intercept and design.shape[1] > 1 and not df_center_linear:
                penalty_base[0, 0] = 0.0
            _, lambda_value = _df2lambda(
                design,
                df=self.df if self.penalty is None else None,
                lambda_value=self.penalty if self.penalty is not None else None,
                penalty_matrix=penalty_base,
                weights=weights,
            )
            penalty_matrix = lambda_value * penalty_base
            penalized_gram = np.ascontiguousarray(np.einsum("ni,n,nj->ij", design, weights, design) + penalty_matrix, dtype=np.float64)
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=np.ascontiguousarray(penalty_matrix, dtype=np.float64),
                center_offsets=np.ascontiguousarray(center_offsets, dtype=np.float64),
                intercept=self.intercept,
                factor_levels=factor_levels,
                by_name=self.by,
                weighted_design_t=np.ascontiguousarray((design * weights[:, None]).T, dtype=np.float64),
                penalized_gram=penalized_gram,
                scalar_denom=None if design.shape[1] != 1 else float(np.dot(weights, design[:, 0] * design[:, 0]) + penalty_matrix[0, 0]),
                penalized_factor=None if design.shape[1] <= 1 else cho_factor(penalized_gram, lower=False, check_finite=False),
            )

        if self.kind == "formula_matrix":
            if self.formula_term is None:
                raise ValueError("formula_matrix learner requires a formula term")
            matrix = model_matrix(
                f"0 + {_formulaic_safe_term(self.formula_term)}",
                to_formulaic_data(data),
            )
            design = _formulaic_matrix_to_numpy(
                matrix,
                column_names=list(self.formula_column_names) if self.formula_column_names is not None else None,
            )
            center_offsets = np.zeros(design.shape[1], dtype=np.float64)
            if self.center:
                design, center_offsets = _weighted_center_matrix(design, weights)
            penalty_value = 0.0 if self.penalty is None else self.penalty
            penalty_matrix = penalty_value * np.eye(design.shape[1], dtype=np.float64)
            penalized_gram = np.ascontiguousarray(np.einsum("ni,n,nj->ij", design, weights, design) + penalty_matrix, dtype=np.float64)
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=np.ascontiguousarray(penalty_matrix, dtype=np.float64),
                center_offsets=np.ascontiguousarray(center_offsets, dtype=np.float64),
                formula_term=self.formula_term,
                formula_spec=matrix.model_spec,
                formula_column_names=self.formula_column_names,
                weighted_design_t=np.ascontiguousarray((design * weights[:, None]).T, dtype=np.float64),
                penalized_gram=penalized_gram,
                scalar_denom=None if design.shape[1] != 1 else float(np.dot(weights, design[:, 0] * design[:, 0]) + penalty_matrix[0, 0]),
                penalized_factor=None if design.shape[1] <= 1 else cho_factor(penalized_gram, lower=False, check_finite=False),
            )

        if self.kind == "tree":
            if self.feature_names is None or len(self.feature_names) == 0:
                raise ValueError("tree base learner requires at least one feature")
            columns = [get_column(data, feature_name) for feature_name in self.feature_names]
            design = np.column_stack(columns)
            tree_by_values = None
            by_levels = None
            if self.by is not None:
                tree_by_values, by_levels = _binary_by_vector(get_raw_column(data, self.by))
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=np.zeros((0, 0), dtype=np.float64),
                center_offsets=np.zeros(0, dtype=np.float64),
                feature_names=self.feature_names,
                by_name=self.by,
                maxdepth=self.maxdepth,
                minsplit=self.minsplit,
                minbucket=self.minbucket,
                tree_by_values=tree_by_values,
                by_levels=by_levels,
            )

        if self.kind == "random":
            raw = get_raw_column(data, self.name)
            factor_levels = _factor_levels(np.asarray(raw))
            design, factor_levels = _factor_full_design(raw, levels=factor_levels)
            if self.by is not None:
                by = get_column(data, self.by)
                design = design * by[:, None]
            penalty_base = np.eye(design.shape[1], dtype=np.float64)
            _, lambda_value = _df2lambda(
                design,
                df=self.df if self.penalty is None else None,
                lambda_value=self.penalty,
                penalty_matrix=penalty_base,
                weights=weights,
            )
            penalty_matrix = lambda_value * penalty_base
            center_offsets = np.zeros(design.shape[1], dtype=np.float64)
            penalized_gram = np.ascontiguousarray(np.einsum("ni,n,nj->ij", design, weights, design) + penalty_matrix, dtype=np.float64)
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=np.ascontiguousarray(penalty_matrix, dtype=np.float64),
                center_offsets=np.ascontiguousarray(center_offsets, dtype=np.float64),
                intercept=False,
                factor_levels=factor_levels,
                by_name=self.by,
                weighted_design_t=np.ascontiguousarray((design * weights[:, None]).T, dtype=np.float64),
                penalized_gram=penalized_gram,
                scalar_denom=None if design.shape[1] != 1 else float(np.dot(weights, design[:, 0] * design[:, 0]) + penalty_matrix[0, 0]),
                penalized_factor=None if design.shape[1] <= 1 else cho_factor(penalized_gram, lower=False, check_finite=False),
            )

        if self.kind == "factor_dummy":
            if self.target_level is None:
                raise ValueError("factor dummy learner requires a target level")
            penalty_value = 0.0 if self.penalty is None else self.penalty
            raw = get_raw_column(data, self.name)
            design = (np.asarray(raw) == self.target_level).astype(np.float64)[:, None]
            if self.by is not None:
                by = get_column(data, self.by)
                design = design * by[:, None]
            center_offsets = np.zeros(design.shape[1], dtype=np.float64)
            if self.center:
                design, center_offsets = _weighted_center_matrix(design, weights)
            penalty_matrix = penalty_value * np.eye(design.shape[1], dtype=np.float64)
            penalized_gram = np.ascontiguousarray(np.einsum("ni,n,nj->ij", design, weights, design) + penalty_matrix, dtype=np.float64)
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=penalty_matrix,
                center_offsets=np.ascontiguousarray(center_offsets, dtype=np.float64),
                intercept=False,
                target_level=self.target_level,
                by_name=self.by,
                weighted_design_t=np.ascontiguousarray((design * weights[:, None]).T, dtype=np.float64),
                penalized_gram=penalized_gram,
                scalar_denom=None if design.shape[1] != 1 else float(np.dot(weights, design[:, 0] * design[:, 0]) + penalty_matrix[0, 0]),
                penalized_factor=None if design.shape[1] <= 1 else cho_factor(penalized_gram, lower=False, check_finite=False),
            )

        if self.kind in {"spline", "mono_spline"}:
            raw = get_raw_column(data, self.name)
            factor_levels = None
            if np.issubdtype(np.asarray(raw).dtype, np.number):
                x = np.asarray(raw, dtype=np.float64)
                if self.knots is not None:
                    if self.knots <= 0:
                        raise ValueError("bbs knots must be positive")
                    design, knot_vector = _build_pspline_basis_from_knots(
                        x,
                        knots=self.knots,
                        degree=self.degree,
                    )
                else:
                    n_basis = self.df if self.df is not None else 6
                    if n_basis < self.degree + 1:
                        raise ValueError("bbs df must be at least degree + 1")
                    design, knot_vector = _build_open_bspline_basis(x, n_basis=n_basis, degree=self.degree)
            else:
                factor_levels = _factor_levels(np.asarray(raw))
                design, factor_levels = _factor_design(raw, intercept=self.intercept, levels=factor_levels)
                knot_vector = None
            if self.by is not None:
                by = get_column(data, self.by)
                design = design * by[:, None]
            if factor_levels is None:
                diff_matrix = np.diff(np.eye(design.shape[1]), n=self.differences, axis=0)
                base_penalty = np.einsum("ki,kj->ij", diff_matrix, diff_matrix)
            else:
                kdiff = np.diff(np.eye(design.shape[1] + 1, dtype=np.float64), n=1, axis=0)[:, 1:]
                base_penalty = np.einsum("ki,kj->ij", kdiff, kdiff)
            center_offsets = np.zeros(0, dtype=np.float64)
            basis_transform = None
            if factor_levels is None and self.center:
                design, base_penalty, basis_transform = _centered_penalized_spline_basis(design, base_penalty)
            else:
                center_offsets = np.zeros(design.shape[1], dtype=np.float64)
            _, lambda_value = _df2lambda(
                design,
                df=self.df if self.penalty is None else None,
                lambda_value=self.penalty,
                penalty_matrix=base_penalty,
                weights=weights,
            )
            penalty_matrix = lambda_value * base_penalty
            penalized_gram = np.ascontiguousarray(np.einsum("ni,n,nj->ij", design, weights, design) + penalty_matrix, dtype=np.float64)
            constraint_matrix = None
            constraint_lb = None
            constraint_ub = None
            if self.kind == "mono_spline":
                constraint_name = self.target_level
                if constraint_name not in {
                    "increasing",
                    "decreasing",
                    "convex",
                    "concave",
                    "positive",
                    "negative",
                }:
                    raise ValueError(f"unsupported monotonic constraint: {constraint_name}")
                if factor_levels is not None:
                    constraint_matrix = _monotone_factor_constraint_matrix(
                        n_levels=factor_levels.shape[0],
                        intercept=self.intercept,
                        constraint=constraint_name,
                    )
                elif constraint_name in {"increasing", "decreasing"}:
                    constraint_matrix = np.diff(np.eye(design.shape[1]), n=1, axis=0)
                elif constraint_name in {"convex", "concave"}:
                    constraint_matrix = np.diff(np.eye(design.shape[1]), n=2, axis=0)
                else:
                    constraint_matrix = np.eye(design.shape[1], dtype=np.float64)
                if factor_levels is None and constraint_name in {"decreasing", "concave", "negative"}:
                    constraint_matrix = -constraint_matrix
                constraint_lb = np.zeros(constraint_matrix.shape[0], dtype=np.float64)
                constraint_ub = np.full(constraint_matrix.shape[0], np.inf, dtype=np.float64)
            return PreparedBaseLearner(
                name=self.name,
                kind=self.kind,
                design=np.ascontiguousarray(design, dtype=np.float64),
                penalty_matrix=np.ascontiguousarray(penalty_matrix, dtype=np.float64),
                center_offsets=np.ascontiguousarray(center_offsets, dtype=np.float64),
                intercept=self.intercept if factor_levels is not None else False,
                factor_levels=factor_levels,
                basis_transform=basis_transform,
                by_name=self.by,
                degree=self.degree,
                knot_vector=None if knot_vector is None else np.ascontiguousarray(knot_vector, dtype=np.float64),
                constraint_matrix=None if constraint_matrix is None else np.ascontiguousarray(constraint_matrix, dtype=np.float64),
                constraint_lb=None if constraint_lb is None else np.ascontiguousarray(constraint_lb, dtype=np.float64),
                constraint_ub=None if constraint_ub is None else np.ascontiguousarray(constraint_ub, dtype=np.float64),
                solver_type=self.solver_type,
                niter=self.niter,
                weighted_design_t=np.ascontiguousarray((design * weights[:, None]).T, dtype=np.float64),
                penalized_gram=penalized_gram,
                scalar_denom=None if design.shape[1] != 1 else float(np.dot(weights, design[:, 0] * design[:, 0]) + penalty_matrix[0, 0]),
                penalized_factor=(
                    None
                    if design.shape[1] <= 1 or self.kind == "mono_spline"
                    else cho_factor(penalized_gram, lower=False, check_finite=False)
                ),
            )

        raise ValueError(f"unknown learner kind: {self.kind}")


def solve_constrained_quadratic(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    constraint_matrix: np.ndarray,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    method: str = "quad.prog",
    niter: int | None = None,
) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    constraint_matrix = np.asarray(constraint_matrix, dtype=np.float64)
    constraint_lb = np.asarray(constraint_lb, dtype=np.float64)
    constraint_ub = np.asarray(constraint_ub, dtype=np.float64)

    def _transform_first_order(size: int, sign: float) -> np.ndarray:
        return sign * np.tril(np.ones((size, size), dtype=np.float64))

    def _transform_second_order(size: int, sign: float) -> np.ndarray:
        transform = np.zeros((size, size), dtype=np.float64)
        transform[:, 0] = sign
        if size > 1:
            transform[:, 1] = sign * np.arange(size, dtype=np.float64)
        for j in range(2, size):
            seed = np.zeros(size - 2, dtype=np.float64)
            seed[j - 2] = 1.0
            transform[:, j] = sign * np.concatenate(
                [
                    np.zeros(2, dtype=np.float64),
                    np.cumsum(np.cumsum(seed)),
                ]
            )
        return transform

    def _difference_transform(matrix: np.ndarray) -> tuple[np.ndarray, int] | None:
        p = matrix.shape[1]
        if matrix.shape == (p, p):
            if np.allclose(matrix, np.eye(p), atol=1e-12, rtol=0.0):
                return np.eye(p, dtype=np.float64), 0
            if np.allclose(matrix, -np.eye(p), atol=1e-12, rtol=0.0):
                return -np.eye(p, dtype=np.float64), 0
        if matrix.shape == (p - 1, p):
            diff1 = np.diff(np.eye(p, dtype=np.float64), n=1, axis=0)
            if np.allclose(matrix, diff1, atol=1e-12, rtol=0.0):
                return _transform_first_order(p, 1.0), 1
            if np.allclose(matrix, -diff1, atol=1e-12, rtol=0.0):
                return _transform_first_order(p, -1.0), 1
        if matrix.shape == (p - 2, p):
            diff2 = np.diff(np.eye(p, dtype=np.float64), n=2, axis=0)
            if np.allclose(matrix, diff2, atol=1e-12, rtol=0.0):
                return _transform_second_order(p, 1.0), 2
            if np.allclose(matrix, -diff2, atol=1e-12, rtol=0.0):
                return _transform_second_order(p, -1.0), 2
        return None

    try:
        beta0 = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta0 = np.zeros(lhs.shape[0], dtype=np.float64)

    initial_violation = constraint_matrix @ beta0 - constraint_lb
    if float(np.min(initial_violation)) >= -1e-10:
        return np.ascontiguousarray(beta0, dtype=np.float64)

    specialized = _difference_transform(constraint_matrix)
    if specialized is not None and np.all(np.isfinite(constraint_lb)) and np.allclose(constraint_lb, 0.0, atol=1e-12) and np.all(np.isinf(constraint_ub)):
        transform, n_free = specialized
        transformed_lhs = np.ascontiguousarray(np.einsum("ki,kl,lj->ij", transform, lhs, transform), dtype=np.float64)
        transformed_rhs = np.ascontiguousarray(np.einsum("ki,k->i", transform, rhs), dtype=np.float64)
        theta0 = np.zeros(transform.shape[1], dtype=np.float64)
        if n_free == 0:
            theta0 = np.maximum(np.diag(transform) * beta0, 0.0)
        elif n_free == 1:
            signed_beta = transform[0, 0] * beta0
            theta0[0] = signed_beta[0]
            theta0[1:] = np.maximum(np.diff(signed_beta), 0.0)
        elif n_free == 2:
            signed_beta = np.sign(transform[0, 0]) * beta0
            theta0[0] = signed_beta[0]
            if theta0.shape[0] > 1:
                theta0[1] = signed_beta[1] - signed_beta[0]
            if theta0.shape[0] > 2:
                theta0[2:] = np.maximum(np.diff(signed_beta, n=2), 0.0)

        theta_objective = lambda theta: 0.5 * float(theta @ transformed_lhs @ theta) - float(transformed_rhs @ theta)
        theta_gradient = lambda theta: transformed_lhs @ theta - transformed_rhs
        bounds = [(None, None)] * n_free + [(0.0, None)] * (transform.shape[1] - n_free)
        result = minimize(
            theta_objective,
            theta0,
            method="L-BFGS-B",
            jac=theta_gradient,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success:
            return np.ascontiguousarray(transform @ result.x, dtype=np.float64)

    objective = lambda beta: 0.5 * float(beta @ lhs @ beta) - float(rhs @ beta)
    gradient = lambda beta: lhs @ beta - rhs
    hessian = lambda _beta: lhs
    ineq_constraint = {
        "type": "ineq",
        "fun": lambda beta: constraint_matrix @ beta - constraint_lb,
        "jac": lambda _beta: constraint_matrix,
    }

    if method == "quad.prog":
        result = minimize(
            objective,
            beta0,
            method="SLSQP",
            jac=gradient,
            constraints=[ineq_constraint],
            options={"maxiter": 200, "ftol": 1e-12},
        )
        violation = constraint_matrix @ result.x - constraint_lb if result.x is not None else np.array([-np.inf])
        if result.success and float(np.min(violation)) >= -1e-8:
            return np.ascontiguousarray(result.x, dtype=np.float64)
        constraint = LinearConstraint(constraint_matrix, constraint_lb, constraint_ub)
        result = minimize(
            objective,
            beta0 if result.x is None else result.x,
            method="trust-constr",
            jac=gradient,
            hess=hessian,
            constraints=[constraint],
            options={"verbose": 0, "maxiter": 300},
        )
        if result.success or getattr(result, "constr_violation", np.inf) <= 1e-8:
            return np.ascontiguousarray(result.x, dtype=np.float64)

    if method not in {"quad.prog", "iterative"}:
        raise ValueError(f"unsupported constrained solver method: {method}")

    # First-pass iterative constrained solve: use SLSQP as a lighter-weight alternative path.
    # The iteration budget scales from niter but keeps a reasonable floor for stability.
    maxiter = max(50, 10 * (10 if niter is None else int(niter)))
    result = minimize(
        objective,
        beta0,
        method="SLSQP",
        jac=gradient,
        constraints=[ineq_constraint],
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    if not result.success and method == "quad.prog":
        result = minimize(
            objective,
            beta0,
            method="SLSQP",
            jac=gradient,
            constraints=[ineq_constraint],
            options={"maxiter": 500, "ftol": 1e-12},
        )
    if not result.success:
        raise RuntimeError(f"constrained quadratic solve failed: {result.message}")
    return np.ascontiguousarray(result.x, dtype=np.float64)
