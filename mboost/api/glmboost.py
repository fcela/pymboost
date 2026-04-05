from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter

import numpy as np
from formulaic import Formula, model_matrix

from mboost.baselearners.base import BaseLearnerSpec, PreparedBaseLearner, _binary_by_vector
from mboost.baselearners.linear import bols, brandom, factor_dummy
from mboost.baselearners.spline import bbs, bmono
from mboost.baselearners.tree import btree
from mboost.core.control import BoostControl, boost_control
from mboost.core.cv import CVRiskResult, cv
from mboost.core.engine import BoostingPath, apply_componentwise_path, evaluate_empirical_risk_path, fit_componentwise_model
from mboost.families.base import Family
from mboost.families.binomial import Binomial
from mboost.families.expectile import Expectile
from mboost.families.gamma import GammaReg
from mboost.families.gaussian import Gaussian
from mboost.families.huber import Huber
from mboost.families.laplace import Laplace
from mboost.families.poisson import Poisson
from mboost.families.quantile import Quantile
from mboost.data import get_column, get_raw_column, to_formulaic_data


def _formulaic_safe_formula(formula: str) -> str:
    return formula.replace("lambda=", "lambda_=")


def _restore_formulaic_term(term: str) -> str:
    return term.replace("lambda_=", "lambda=")


def _split_formula(formula: str, data=None) -> tuple[str, list[str]]:
    response_part, rhs_part = formula.split("~", 1)
    response_name = response_part.strip()
    if rhs_part.strip() == ".":
        if data is None:
            raise ValueError("formula '.' expansion requires data")
        terms = [name for name in to_formulaic_data(data).columns if name != response_name]
        if not terms:
            raise ValueError("formula '.' expansion requires at least one predictor")
        return response_name, terms

    parsed = Formula(_formulaic_safe_formula(formula))
    response = str(parsed.lhs).strip()
    terms = [
        _restore_formulaic_term(str(term).strip())
        for term in parsed.rhs
        if str(term).strip() != "1"
    ]
    if not response or not terms:
        raise ValueError("formula must have a response and at least one term")
    return response, terms


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        current.append(char)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def _parse_call(term: str) -> tuple[str, list[str], dict[str, object]]:
    fn_name, arg_text = term.split("(", 1)
    arg_text = arg_text[:-1]
    parts = _split_args(arg_text)
    if not parts:
        raise ValueError(f"{fn_name} requires at least one argument")
    args: list[str] = []
    kwargs: dict[str, object] = {}
    for token in parts:
        if "=" not in token:
            args.append(token.strip())
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            parsed = value[1:-1]
        elif value in {"True", "False"}:
            parsed: object = value == "True"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError as exc:
                    if value.isidentifier():
                        parsed = value
                    else:
                        raise ValueError(f"unsupported argument value: {value}") from exc
        kwargs[key] = parsed
    return fn_name.strip(), args, kwargs


def _parse_term(term: str) -> BaseLearnerSpec:
    if term.endswith(")"):
        fn_name, args, kwargs = _parse_call(term)
        if fn_name == "bols":
            if len(args) != 1:
                raise ValueError("bols requires exactly one feature")
            if "lambda" in kwargs and "lambda_" not in kwargs:
                kwargs["lambda_"] = kwargs.pop("lambda")
            return bols(args[0], **kwargs)
        if fn_name == "bbs":
            if len(args) != 1:
                raise ValueError("bbs requires exactly one feature")
            if "lambda" in kwargs and "lambda_" not in kwargs:
                kwargs["lambda_"] = kwargs.pop("lambda")
            return bbs(args[0], **kwargs)
        if fn_name == "bmono":
            if len(args) != 1:
                raise ValueError("bmono requires exactly one feature")
            if "lambda" in kwargs and "lambda_" not in kwargs:
                kwargs["lambda_"] = kwargs.pop("lambda")
            if "boundary.constraints" in kwargs and "boundary_constraints" not in kwargs:
                kwargs["boundary_constraints"] = kwargs.pop("boundary.constraints")
            return bmono(args[0], **kwargs)
        if fn_name == "brandom":
            if len(args) != 1:
                raise ValueError("brandom requires exactly one feature")
            if "lambda" in kwargs and "lambda_" not in kwargs:
                kwargs["lambda_"] = kwargs.pop("lambda")
            return brandom(args[0], **kwargs)
        if fn_name == "btree":
            return btree(*args, **kwargs)
    if term.isidentifier():
        return bols(term, intercept=False, center=True)
    return BaseLearnerSpec(
        name=term,
        kind="formula_matrix",
        penalty=0.0,
        center=True,
        formula_term=term,
    )


def _factor_levels(values: np.ndarray) -> np.ndarray:
    levels = np.unique(values[~np.equal(values, None)])
    return np.sort(levels)


def _parse_term_for_family(term: str, family: Family, data) -> list[tuple[BaseLearnerSpec, str]]:
    if term.endswith(")"):
        maybe_name = term.split("(", 1)[0].strip()
        if maybe_name in {"bols", "bbs", "bmono", "brandom", "btree"}:
            return [(_parse_term(term), term)]
    if not term.isidentifier():
        spec = _parse_term(term)
        if spec.kind != "formula_matrix":
            return [(spec, term)]
        matrix = model_matrix(f"0 + {_formulaic_safe_formula(term)}", to_formulaic_data(data))
        columns = [str(column) for column in matrix.columns]
        if len(columns) <= 1:
            return [(spec, term)]
        return [
            (
                BaseLearnerSpec(
                    name=column,
                    kind="formula_matrix",
                    penalty=0.0,
                    center=True,
                    formula_term=term,
                    formula_column_names=(column,),
                ),
                column,
            )
            for column in columns
        ]
    raw = np.asarray(get_raw_column(data, term))
    if not np.issubdtype(raw.dtype, np.number):
        levels = _factor_levels(raw)
        return [
            (factor_dummy(term, target_level=level), f"{term}[{level}]")
            for level in levels[1:]
        ]
    if isinstance(family, (Quantile, Expectile, Huber)):
        return [(bols(term), term)]
    return [(_parse_term(term), term)]


def _as_array(data, key: str) -> np.ndarray:
    return get_column(data, key)


def _prepare_learners(
    *,
    data,
    learner_specs: list[BaseLearnerSpec],
    sample_weights: np.ndarray,
) -> list[PreparedBaseLearner]:
    return [spec.prepare(data, sample_weights) for spec in learner_specs]


@dataclass
class GLMBoostModel:
    formula: str
    family: Family
    control: BoostControl
    response_name: str
    data: object
    learner_specs: list[BaseLearnerSpec]
    term_labels: list[str]
    prepared_learners: list[PreparedBaseLearner]
    y: np.ndarray
    sample_weights: np.ndarray
    path: BoostingPath
    feature_names: list[str]

    @property
    def mstop(self) -> int:
        return self.control.mstop

    @property
    def selected(self) -> list[str]:
        return [self.feature_names[idx] for idx in self.path.selected]

    @property
    def fitted_(self) -> np.ndarray:
        return self.path.fitted

    @property
    def offset_(self) -> float:
        return self.path.offset

    @property
    def risk_(self) -> np.ndarray:
        return self.path.risk

    @property
    def coef_path_(self) -> list[np.ndarray]:
        out = []
        for coef in self.path.coefficients:
            out.append(coef.copy() if hasattr(coef, "copy") else coef)
        return out

    @property
    def coefficients_(self) -> dict[str, object]:
        coefs: dict[str, object] = {}
        for label, learner in zip(self.term_labels, self.prepared_learners):
            if learner.kind == "tree":
                coefs[label] = []
            else:
                coefs[label] = np.zeros(learner.design.shape[1], dtype=np.float64)
        for idx, coef in zip(self.path.selected, self.path.coefficients):
            label = self.term_labels[idx]
            if self.prepared_learners[idx].kind == "tree":
                coefs[label].append(coef)
            else:
                coefs[label] += self.control.nu * coef
        out: dict[str, object] = {}
        for key, value in coefs.items():
            if hasattr(value, "copy"):
                out[key] = value.copy()
            elif isinstance(value, list):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:
        return (
            f"GLMBoostModel(formula={self.formula!r}, family={self.family.__class__.__name__}, "
            f"mstop={self.mstop}, learners={len(self.learner_specs)}, "
            f"final_risk={self.risk_[-1]:.6f})"
        )

    def summary(self, *, max_terms: int = 8) -> str:
        counts = Counter(self.selected)
        top_terms = counts.most_common(max_terms)
        top_terms_text = ", ".join(f"{name} ({count})" for name, count in top_terms) if top_terms else "none"
        risk_drop = float(self.risk_[0] - self.risk_[-1])
        risk_ratio = risk_drop / float(self.risk_[0]) if self.risk_[0] != 0.0 else np.nan
        lines = [
            "Model-based boosting fit",
            f"Formula: {self.formula}",
            f"Family: {self.family.__class__.__name__}",
            f"Observations: {self.y.shape[0]}",
            f"Base-learners: {len(self.learner_specs)}",
            f"Boosting iterations: {self.mstop}",
            f"Step size (nu): {self.control.nu:g}",
            f"Offset: {self.offset_:.6g}",
            f"Empirical risk: {self.risk_[0]:.6g} -> {self.risk_[-1]:.6g}",
            f"Risk reduction: {risk_drop:.6g} ({risk_ratio:.1%})" if np.isfinite(risk_ratio) else f"Risk reduction: {risk_drop:.6g}",
            f"Selected learners: {len(counts)} unique",
            f"Top selected terms: {top_terms_text}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def predict(self, *, newdata=None, mstop: int | None = None, type: str = "link") -> np.ndarray:
        if type not in {"link", "response"}:
            raise ValueError("type must be either 'link' or 'response'")
        steps = self.control.mstop if mstop is None else mstop
        if steps < 0 or steps > self.control.mstop:
            raise ValueError("mstop must be between 0 and the fitted model's mstop")
        if newdata is None and steps == self.control.mstop:
            link_pred = self.path.fitted.copy()
            return link_pred if type == "link" else self.family.response(link_pred)

        if newdata is None:
            learners = self.prepared_learners
        else:
            learners = [
                PreparedBaseLearner(
                    name=learner.name,
                    kind=learner.kind,
                    design=learner.transform(newdata),
                    penalty_matrix=learner.penalty_matrix,
                    center_offsets=learner.center_offsets,
                    feature_names=learner.feature_names,
                    by_name=learner.by_name,
                    degree=learner.degree,
                    basis_transform=learner.basis_transform,
                    knot_vector=learner.knot_vector,
                    formula_term=learner.formula_term,
                    formula_spec=learner.formula_spec,
                formula_column_names=learner.formula_column_names,
                constraint_matrix=learner.constraint_matrix,
                constraint_lb=learner.constraint_lb,
                constraint_ub=learner.constraint_ub,
                solver_type=learner.solver_type,
                niter=learner.niter,
                maxdepth=learner.maxdepth,
                minsplit=learner.minsplit,
                minbucket=learner.minbucket,
                tree_by_values=(
                    None
                    if learner.kind != "tree" or learner.by_name is None
                    else _binary_by_vector(get_raw_column(newdata, learner.by_name), levels=learner.by_levels)[0]
                ),
                by_levels=learner.by_levels,
            )
            for learner in self.prepared_learners
        ]
        link_pred = apply_componentwise_path(
            learners=learners,
            offset=self.path.offset,
            selected=self.path.selected,
            coefficients=self.path.coefficients,
            nu=self.control.nu,
            mstop=steps,
        )
        return link_pred if type == "link" else self.family.response(link_pred)

    def with_mstop(self, mstop: int) -> "GLMBoostModel":
        fitted = self.predict(mstop=mstop)
        truncated_path = BoostingPath(
            offset=self.path.offset,
            selected=self.path.selected[:mstop],
            coefficients=[coef.copy() if hasattr(coef, "copy") else coef for coef in self.path.coefficients[:mstop]],
            risk=self.path.risk[: mstop + 1].copy(),
            fitted=fitted,
        )
        return GLMBoostModel(
            formula=self.formula,
            family=self.family,
            control=boost_control(mstop=mstop, nu=self.control.nu),
            response_name=self.response_name,
            data=self.data,
            learner_specs=self.learner_specs,
            term_labels=self.term_labels,
            prepared_learners=self.prepared_learners,
            y=self.y,
            sample_weights=self.sample_weights,
            path=truncated_path,
            feature_names=self.feature_names,
        )


def glmboost(
    formula: str,
    *,
    data,
    family=None,
    control: BoostControl | None = None,
    weights=None,
) -> GLMBoostModel:
    response_name, term_strings = _split_formula(formula, data=data)
    if family is None:
        family = Gaussian()
    if not isinstance(family, Family):
        raise TypeError("family must be an instance of mboost.families.base.Family")
    if control is None:
        control = boost_control()
    expanded_terms = [item for term in term_strings for item in _parse_term_for_family(term, family, data)]
    learner_specs = [spec for spec, _ in expanded_terms]
    term_labels = [label for _, label in expanded_terms]

    y = _as_array(data, response_name)
    if weights is None:
        sample_weights = np.ones_like(y, dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != y.shape:
            raise ValueError("weights must have the same shape as the response")

    y_array = np.ascontiguousarray(y, dtype=np.float64)
    weight_array = np.ascontiguousarray(sample_weights, dtype=np.float64)
    prepared_learners = _prepare_learners(
        data=data,
        learner_specs=learner_specs,
        sample_weights=weight_array,
    )
    path = fit_componentwise_model(
        learners=prepared_learners,
        y=y_array,
        family=family,
        control=control,
        weights=weight_array,
    )

    return GLMBoostModel(
        formula=formula,
        family=family,
        control=control,
        response_name=response_name,
        data=data,
        learner_specs=learner_specs,
        term_labels=term_labels,
        prepared_learners=prepared_learners,
        y=y_array,
        sample_weights=weight_array,
        path=path,
        feature_names=[label if spec.kind == "factor_dummy" else spec.name for spec, label in expanded_terms],
    )


def mstop(model: GLMBoostModel, value: int | None = None):
    if value is None:
        return model.mstop
    return model.with_mstop(value)


def cvrisk(
    formula: str,
    *,
    data,
    family=None,
    control: BoostControl | None = None,
    folds: int | np.ndarray | None = None,
    type: str | None = None,
    B: int | None = None,
    fraction: float = 0.5,
    shuffle: bool = False,
    random_state: int = 0,
    weights=None,
) -> CVRiskResult:
    if control is None:
        control = boost_control()
    response_name, term_strings = _split_formula(formula, data=data)
    if family is None:
        family = Gaussian()
    if not isinstance(family, Family):
        raise TypeError("family must be an instance of mboost.families.base.Family")
    expanded_terms = [item for term in term_strings for item in _parse_term_for_family(term, family, data)]
    learner_specs = [spec for spec, _ in expanded_terms]
    y = _as_array(data, response_name)
    if weights is None:
        sample_weights = np.ones_like(y, dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != y.shape:
            raise ValueError("weights must have the same shape as the response")

    def _generated_training_weights() -> np.ndarray:
        resolved_type = type
        if resolved_type is None:
            resolved_type = "bootstrap" if folds is None else "kfold"

        if resolved_type == "kfold":
            resolved_folds = 10 if folds is None else int(folds)
            fold_ids = cv(
                y.shape[0],
                folds=resolved_folds,
                type="kfold",
                shuffle=shuffle,
                random_state=random_state,
            )
            unique_folds = np.unique(fold_ids)
            train = np.ones((y.shape[0], unique_folds.shape[0]), dtype=np.float64)
            for idx, fold_id in enumerate(unique_folds):
                train[fold_ids == fold_id, idx] = 0.0
            return train

        rng = np.random.default_rng(random_state)
        n_samples = y.shape[0]
        probs = np.asarray(sample_weights, dtype=np.float64)
        prob_sum = float(np.sum(probs))
        if prob_sum <= 0.0:
            raise ValueError("weights must sum to a positive value")
        probs = probs / prob_sum
        choice_probs = None if np.allclose(probs, np.full_like(probs, 1.0 / probs.shape[0])) else probs

        resolved_B = B
        if resolved_B is None:
            resolved_B = 25 if folds is None else int(folds)
        if resolved_B <= 0:
            raise ValueError("B must be positive")

        if resolved_type == "bootstrap":
            train = np.zeros((n_samples, resolved_B), dtype=np.float64)
            for b in range(resolved_B):
                draw = rng.choice(n_samples, size=n_samples, replace=True, p=choice_probs)
                train[:, b] = np.bincount(draw, minlength=n_samples).astype(np.float64)
            return train

        if resolved_type == "subsampling":
            if not 0.0 < fraction < 1.0:
                raise ValueError("fraction must be strictly between 0 and 1 for subsampling")
            train_size = max(1, int(np.floor(fraction * n_samples)))
            train = np.zeros((n_samples, resolved_B), dtype=np.float64)
            for b in range(resolved_B):
                draw = rng.choice(n_samples, size=train_size, replace=False, p=choice_probs)
                train[draw, b] = 1.0
            return train

        raise ValueError("type must be one of {'bootstrap', 'kfold', 'subsampling'}")

    if folds is None or isinstance(folds, int):
        fold_structure = _generated_training_weights()
        fold_mode = "train_weights"
    else:
        fold_structure = np.asarray(folds)
        if fold_structure.ndim == 1 and fold_structure.shape != y.shape:
            raise ValueError("fold assignments must have the same shape as the response")
        if fold_structure.ndim == 2 and fold_structure.shape[0] != y.shape[0]:
            raise ValueError("fold matrix must have n_samples rows")
        if fold_structure.ndim not in {1, 2}:
            raise ValueError("folds must be an integer, a fold-id vector, or an out-of-bag mask matrix")
        if fold_structure.ndim == 2 and np.any(fold_structure > 1.0):
            fold_mode = "train_weights"
        else:
            fold_mode = "holdout_mask_or_fold_ids"

    y_array = np.ascontiguousarray(y, dtype=np.float64)
    weight_array = np.ascontiguousarray(sample_weights, dtype=np.float64)
    base_prepared_learners = _prepare_learners(
        data=data,
        learner_specs=learner_specs,
        sample_weights=weight_array,
    )
    if fold_mode == "train_weights":
        holdout_masks = [np.asarray(fold_structure[:, j] == 0.0, dtype=np.float64) for j in range(fold_structure.shape[1])]
        training_weight_columns = [np.asarray(fold_structure[:, j], dtype=np.float64) for j in range(fold_structure.shape[1])]
    elif np.asarray(fold_structure).ndim == 1:
        fold_ids = np.asarray(fold_structure)
        unique_folds = np.unique(fold_ids)
        holdout_masks = [(fold_ids == fold_id).astype(np.float64) for fold_id in unique_folds]
        training_weight_columns = None
    else:
        fold_ids = np.asarray(fold_structure, dtype=np.float64)
        holdout_masks = [fold_ids[:, j] for j in range(fold_ids.shape[1])]
        training_weight_columns = None

    fold_risk = np.empty((len(holdout_masks), control.mstop + 1), dtype=np.float64)

    for row_idx, holdout_mask in enumerate(holdout_masks):
        if training_weight_columns is None:
            train_weights = np.ascontiguousarray(weight_array * (1.0 - holdout_mask), dtype=np.float64)
            oob_weights = np.ascontiguousarray(weight_array * holdout_mask, dtype=np.float64)
        else:
            train_weights = np.ascontiguousarray(weight_array * training_weight_columns[row_idx], dtype=np.float64)
            oob_weights = np.ascontiguousarray(weight_array * holdout_mask, dtype=np.float64)
        prepared_learners = []
        for spec, base_prepared in zip(learner_specs, base_prepared_learners):
            if spec.kind in {"spline", "mono_spline", "tree"}:
                prepared_learners.append(spec.prepare(data, train_weights))
            else:
                prepared_learners.append(
                    replace(
                        base_prepared,
                        weighted_design_t=None,
                        penalized_gram=None,
                        scalar_denom=None,
                        penalized_factor=None,
                    )
                )
        path = fit_componentwise_model(
            learners=prepared_learners,
            y=y_array,
            family=family,
            control=control,
            weights=train_weights,
        )
        fold_risk[row_idx] = evaluate_empirical_risk_path(
            learners=prepared_learners,
            offset=path.offset,
            selected=path.selected,
            coefficients=path.coefficients,
            nu=control.nu,
            y=y_array,
            weights=oob_weights,
            family=family,
        )

    return CVRiskResult(
        risk=fold_risk.mean(axis=0),
        fold_risk=fold_risk,
        folds=np.asarray(fold_structure),
    )
