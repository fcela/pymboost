from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np

from mboost.api.glmboost import GLMBoostModel
from mboost.core.cv import CVRiskResult
from mboost.baselearners.base import _binary_by_vector
from mboost.data import get_raw_column
from mboost.inference import ConfIntResult
from mboost.metrics import AICResult


def _require_plot_deps():
    import altair as alt
    import pandas as pd

    return alt, pd


@dataclass(frozen=True)
class VarImpResult:
    data: Any
    type: str
    percent: bool

    def to_pandas(self):
        _, pd = _require_plot_deps()
        return pd.DataFrame(self.data)


def _variable_group_name(name: str) -> str:
    parts = [part.strip() for part in name.split(",")]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(sorted(parts))


def _learner_feature_names(model: GLMBoostModel, learner_idx: int) -> tuple[str, ...]:
    learner = model.prepared_learners[learner_idx]
    if learner.feature_names is not None:
        return learner.feature_names
    return (learner.name,)


def _format_label_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else repr(float(value))
    return str(value)


def _baselearner_label(model: GLMBoostModel, learner_idx: int) -> str:
    spec = model.learner_specs[learner_idx]
    if spec.kind == "linear":
        args = [spec.name]
        if spec.penalty not in {None, 0.0}:
            args.append(f"lambda = {_format_label_value(spec.penalty)}")
        if not spec.intercept:
            args.append("intercept = FALSE")
        if spec.center:
            args.append("center = TRUE")
        if spec.by is not None:
            args.append(f"by = {spec.by}")
        return f"bols({', '.join(args)})"
    if spec.kind == "spline":
        args = [spec.name]
        if spec.knots is not None:
            args.append(f"knots = {_format_label_value(spec.knots)}")
        args.append(f"degree = {_format_label_value(spec.degree)}")
        args.append(f"differences = {_format_label_value(spec.differences)}")
        if spec.penalty is not None:
            args.append(f"lambda = {_format_label_value(spec.penalty)}")
        else:
            args.append(f"df = {_format_label_value(spec.df)}")
        if spec.by is not None:
            args.append(f"by = {spec.by}")
        return f"bbs({', '.join(args)})"
    if spec.kind == "mono_spline":
        args = [
            spec.name,
            f"constraint = {_format_label_value(spec.target_level)}",
            f"knots = {_format_label_value(spec.knots)}",
            f"degree = {_format_label_value(spec.degree)}",
            f"differences = {_format_label_value(spec.differences)}",
        ]
        if spec.penalty is not None:
            args.append(f"lambda = {_format_label_value(spec.penalty)}")
        else:
            args.append(f"df = {_format_label_value(spec.df)}")
        if spec.by is not None:
            args.append(f"by = {spec.by}")
        return f"bmono({', '.join(args)})"
    if spec.kind == "random":
        args = [spec.name]
        if spec.df is not None:
            args.append(f"df = {_format_label_value(spec.df)}")
        if spec.penalty is not None:
            args.append(f"lambda = {_format_label_value(spec.penalty)}")
        return f"brandom({', '.join(args)})"
    if spec.kind == "tree":
        names = spec.feature_names if spec.feature_names is not None else (spec.name,)
        return f"btree({', '.join(names)})"
    return model.term_labels[learner_idx]


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


def _final_learner_contribution(model: GLMBoostModel, learner_idx: int, data) -> np.ndarray:
    learner = model.prepared_learners[learner_idx]
    design = learner.transform(data)
    label = model.term_labels[learner_idx]
    coef = model.coefficients_.get(label)
    if coef is None:
        raise KeyError(f"missing coefficient entry for learner {label!r}")
    if learner.kind == "tree":
        out = np.zeros(design.shape[0], dtype=np.float64)
        for tree in coef:
            out += model.control.nu * tree.predict(design)
        if learner.by_name is not None:
            by_values = _binary_by_vector(get_raw_column(data, learner.by_name), levels=learner.by_levels)[0]
            out = out * by_values
        return out
    if isinstance(coef, list):
        raise NotImplementedError("list-based learner coefficients are not supported for partial plots")
    return np.asarray(np.einsum("ij,j->i", design, coef), dtype=np.float64)


def varimp(model: GLMBoostModel, *, percent: bool = True, type: str = "variable") -> VarImpResult:
    if type not in {"variable", "blearner"}:
        raise ValueError("type must be 'variable' or 'blearner'")

    n_obs = model.y.shape[0]
    riskdiff = (-np.diff(model.risk_)) / n_obs
    baselearner_names = [_baselearner_label(model, idx) for idx in range(len(model.prepared_learners))]
    variable_names = list(model.feature_names)
    selected = model.path.selected

    explained_by_learner: dict[str, float] = {name: 0.0 for name in baselearner_names}
    selfreq_by_learner: dict[str, float] = {name: 0.0 for name in baselearner_names}
    for idx, name in enumerate(baselearner_names):
        explained_by_learner[name] = float(sum(riskdiff[step] for step, sel in enumerate(selected) if sel == idx))
        selfreq_by_learner[name] = float(np.mean(np.asarray(selected) == idx)) if selected else 0.0

    rows: list[dict[str, object]] = []
    if type == "blearner":
        values = explained_by_learner
        total = sum(abs(value) for value in values.values())
        for name, value in values.items():
            reduction = 100.0 * value / total if percent and total > 0.0 else value
            rows.append(
                {
                    "label": name,
                    "reduction": reduction,
                    "selfreq": selfreq_by_learner[name],
                    "baselearner": name,
                    "variable": _variable_group_name(name),
                }
            )
    else:
        grouped: dict[str, float] = {}
        grouped_freq: dict[str, float] = {}
        for name, value, variable_name in zip(baselearner_names, explained_by_learner.values(), variable_names):
            variable_name = _variable_group_name(variable_name)
            grouped[variable_name] = grouped.get(variable_name, 0.0) + value
            grouped_freq[variable_name] = grouped_freq.get(variable_name, 0.0) + selfreq_by_learner[name]
        total = sum(abs(value) for value in grouped.values())
        for name, value in grouped.items():
            reduction = 100.0 * value / total if percent and total > 0.0 else value
            rows.append(
                {
                    "label": name,
                    "reduction": reduction,
                    "selfreq": grouped_freq[name],
                    "baselearner": name,
                    "variable": name,
                }
            )

    rows.sort(key=lambda row: row["reduction"])
    return VarImpResult(data=rows, type=type, percent=percent)


def _contribution_data_for_learner(model: GLMBoostModel, *, learner_idx: int, grid_size: int | None = 100):
    _, pd = _require_plot_deps()
    learner = model.prepared_learners[learner_idx]
    label = model.term_labels[learner_idx]
    training = _training_data_dict(model)
    feature_names = _learner_feature_names(model, learner_idx)
    effective_grid_size = 100 if grid_size is None else grid_size

    if len(feature_names) == 1:
        feature = feature_names[0]
        raw = np.asarray(get_raw_column(model.data, feature))
        if grid_size is None and np.issubdtype(raw.dtype, np.number):
            x_values = np.sort(np.asarray(raw, dtype=np.float64))
        else:
            x_values = _grid_for_feature(raw, grid_size=effective_grid_size)
        plot_data = {name: value.copy() for name, value in training.items()}
        plot_data[feature] = x_values
        n_rows = x_values.shape[0]
        for key in list(plot_data):
            if key == feature:
                continue
            plot_data[key] = _repeat_value(plot_data[key], n_rows)
        if learner.by_name is not None and learner.by_name not in plot_data:
            plot_data[learner.by_name] = np.ones(n_rows, dtype=np.float64)
        effect = _final_learner_contribution(model, learner_idx, plot_data)
        if x_values.dtype.kind in {"O", "U", "S"} or not np.issubdtype(x_values.dtype, np.number):
            return pd.DataFrame(
                {
                    "term": np.repeat(label, x_values.shape[0]),
                    "feature": np.repeat(feature, x_values.shape[0]),
                    "x": [str(value) for value in x_values],
                    "effect": effect,
                    "kind": np.repeat("categorical", x_values.shape[0]),
                }
            )
        return pd.DataFrame(
            {
                "term": np.repeat(label, x_values.shape[0]),
                "feature": np.repeat(feature, x_values.shape[0]),
                "x": np.asarray(x_values, dtype=np.float64),
                "effect": effect,
                "kind": np.repeat("numeric", x_values.shape[0]),
            }
        )

    if len(feature_names) == 2:
        raw_x = get_raw_column(model.data, feature_names[0])
        raw_y = get_raw_column(model.data, feature_names[1])
        x_grid = _grid_for_feature(raw_x, grid_size=max(16, min(effective_grid_size // 2, 50)))
        y_grid = _grid_for_feature(raw_y, grid_size=max(16, min(effective_grid_size // 2, 50)))
        if (
            np.issubdtype(np.asarray(x_grid).dtype, np.number)
            and np.issubdtype(np.asarray(y_grid).dtype, np.number)
        ):
            xx, yy = np.meshgrid(np.asarray(x_grid, dtype=np.float64), np.asarray(y_grid, dtype=np.float64), indexing="xy")
            plot_data = {name: _repeat_value(value, xx.size) for name, value in training.items()}
            plot_data[feature_names[0]] = xx.reshape(-1)
            plot_data[feature_names[1]] = yy.reshape(-1)
            effect = _final_learner_contribution(model, learner_idx, plot_data)
            return pd.DataFrame(
                {
                    "term": np.repeat(label, xx.size),
                    "x": xx.reshape(-1),
                    "y": yy.reshape(-1),
                    "effect": effect,
                    "kind": np.repeat("surface", xx.size),
                    "feature": np.repeat(f"{feature_names[0]}, {feature_names[1]}", xx.size),
                    "feature_x": np.repeat(feature_names[0], xx.size),
                    "feature_y": np.repeat(feature_names[1], xx.size),
                }
            )

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        raw = get_raw_column(model.data, feature)
        grid = _grid_for_feature(raw, grid_size=effective_grid_size)
        for value in grid:
            plot_data = {name: np.asarray(values).copy() for name, values in training.items()}
            plot_data[feature] = _repeat_value(value, model.y.shape[0])
            effect = float(np.mean(_final_learner_contribution(model, learner_idx, plot_data)))
            rows.append(
                {
                    "term": label,
                    "feature": feature,
                    "x": str(value) if not np.issubdtype(np.asarray(grid).dtype, np.number) else float(value),
                    "effect": effect,
                    "kind": "sensitivity_categorical"
                    if (np.asarray(grid).dtype.kind in {"O", "U", "S"} or not np.issubdtype(np.asarray(grid).dtype, np.number))
                    else "sensitivity_numeric",
                }
            )
    return pd.DataFrame(rows)


def _resolve_plot_indices(
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


def partial_plot_data(
    model: GLMBoostModel,
    *,
    which: int | str | list[int] | list[str] | None = None,
    grid_size: int | None = 100,
):
    _, pd = _require_plot_deps()
    indices = _resolve_plot_indices(model, which)
    frames = [_contribution_data_for_learner(model, learner_idx=idx, grid_size=grid_size) for idx in indices]
    return pd.concat(frames, ignore_index=True)


@singledispatch
def plot(obj, **kwargs):
    raise NotImplementedError(f"plot is not implemented for {type(obj)!r}")


@plot.register
def _(obj: GLMBoostModel, *, which=None, grid_size: int | None = None, width: int = 220, height: int = 160):
    alt, pd = _require_plot_deps()
    indices = _resolve_plot_indices(obj, which)
    data = partial_plot_data(obj, which=which, grid_size=grid_size)
    kinds = set(data["kind"].unique())
    if kinds == {"numeric"}:
        rug_rows: list[dict[str, object]] = []
        for idx in indices:
            label = obj.term_labels[idx]
            feature = _learner_feature_names(obj, idx)[0]
            raw = np.asarray(get_raw_column(obj.data, feature), dtype=np.float64)
            rug_rows.extend({"term": label, "rug_x": value} for value in raw)
        rug_data = pd.DataFrame(rug_rows)
        lines = (
            alt.Chart(data)
            .mark_line()
            .properties(width=width, height=height)
            .encode(
                x=alt.X("x:Q", title=None),
                y=alt.Y("effect:Q", title="Partial effect"),
            )
        )
        points = (
            alt.Chart(data)
            .mark_point(filled=True, size=24)
            .properties(width=width, height=height)
            .encode(
                x=alt.X("x:Q", title=None),
                y=alt.Y("effect:Q", title="Partial effect"),
            )
        )
        rugs = (
            alt.Chart(rug_data.drop_duplicates(subset=["term", "rug_x"]))
            .mark_tick(opacity=0.25, orient="vertical")
            .properties(width=width, height=height)
            .encode(x=alt.X("rug_x:Q", title=None))
        )
        chart = (
            alt.layer(lines, points, rugs, data=data)
            .facet("term:N")
            .resolve_scale(x="independent", y="independent")
        )
    elif kinds == {"categorical"}:
        categorical_data = data.copy()
        categorical_data["x_label"] = categorical_data["x"].astype(str)
        categorical_data["x_pos"] = categorical_data.groupby("term").cumcount() + 1
        categorical_data["x_start"] = categorical_data["x_pos"] - 0.35
        categorical_data["x_end"] = categorical_data["x_pos"] + 0.35
        segments = (
            alt.Chart(categorical_data)
            .mark_rule(strokeWidth=2.5)
            .properties(width=width, height=height)
            .encode(
                x=alt.X(
                    "x_start:Q",
                    title=None,
                    axis=alt.Axis(
                        values=sorted(categorical_data["x_pos"].unique().tolist()),
                        labelExpr="datum.value",
                    ),
                ),
                x2="x_end:Q",
                y=alt.Y("effect:Q", title="Partial effect"),
            )
        )
        labels = (
            alt.Chart(categorical_data)
            .mark_text(dy=12, baseline="top")
            .properties(width=width, height=height)
            .encode(
                x=alt.X("x_pos:Q", title=None, axis=None),
                y=alt.value(0),
                text="x_label:N",
            )
        )
        chart = (
            alt.layer(segments, data=categorical_data)
            .facet("term:N")
            .resolve_scale(x="independent", y="independent")
        )
    elif kinds == {"surface"}:
        chart = (
            alt.Chart(data)
            .mark_rect()
            .properties(width=width, height=height)
            .encode(
                x=alt.X("x:Q", title="Feature 1"),
                y=alt.Y("y:Q", title="Feature 2"),
                color=alt.Color("effect:Q", title="Partial effect"),
                tooltip=["term:N", "feature_x:N", "feature_y:N", "x:Q", "y:Q", "effect:Q"],
            )
            .facet("term:N")
            .resolve_scale(x="independent", y="independent", color="independent")
        )
    elif kinds <= {"sensitivity_numeric", "sensitivity_categorical"}:
        base = alt.Chart(data)
        if "sensitivity_numeric" in kinds:
            chart = (
                base.mark_line(point=True)
                .properties(width=width, height=height)
                .encode(
                    x=alt.X("x:Q", title=None),
                    y=alt.Y("effect:Q", title="Average partial effect"),
                )
                .facet(row="term:N", column="feature:N")
                .resolve_scale(x="independent", y="independent")
            )
        else:
            chart = (
                base.mark_bar()
                .properties(width=width, height=height)
                .encode(
                    x=alt.X("x:N", title=None),
                    y=alt.Y("effect:Q", title="Average partial effect"),
                )
                .facet(row="term:N", column="feature:N")
                .resolve_scale(x="independent", y="independent")
            )
    else:
        raise NotImplementedError(f"mixed partial plot kinds are not implemented: {sorted(kinds)}")
    return chart


@plot.register
def _(obj: CVRiskResult, *, width: int = 420, height: int = 260):
    alt, pd = _require_plot_deps()
    iterations = np.arange(obj.risk.shape[0], dtype=int)
    mean_df = pd.DataFrame({"iteration": iterations, "risk": obj.risk, "kind": "mean"})
    fold_rows = []
    for fold_idx in range(obj.fold_risk.shape[0]):
        for iteration, risk in enumerate(obj.fold_risk[fold_idx]):
            fold_rows.append({"iteration": iteration, "risk": risk, "fold": str(fold_idx), "kind": "fold"})
    fold_df = pd.DataFrame(fold_rows)
    folds = (
        alt.Chart(fold_df)
        .mark_line(color="lightgray")
        .encode(x=alt.X("iteration:Q", title="Boosting iteration"), y=alt.Y("risk:Q", title="Risk"), detail="fold:N")
    )
    mean = alt.Chart(mean_df).mark_line(color="#1f77b4").encode(x="iteration:Q", y="risk:Q")
    rule = alt.Chart(pd.DataFrame({"iteration": [obj.best_mstop]})).mark_rule(strokeDash=[4, 4]).encode(x="iteration:Q")
    return alt.layer(folds, mean, rule).properties(width=width, height=height)


@plot.register
def _(obj: AICResult, *, width: int = 420, height: int = 260):
    alt, pd = _require_plot_deps()
    iterations = np.arange(1, obj.aic_path.shape[0] + 1, dtype=int)
    data = pd.DataFrame({"iteration": iterations, "aic": obj.aic_path})
    line = alt.Chart(data).mark_line(color="#1f77b4").encode(
        x=alt.X("iteration:Q", title="Boosting iteration"),
        y=alt.Y("aic:Q", title="Corrected AIC" if obj.corrected else "AIC"),
    )
    point = alt.Chart(pd.DataFrame({"iteration": [obj.mstop], "aic": [obj.value]})).mark_point(size=80, color="#d62728").encode(
        x="iteration:Q", y="aic:Q"
    )
    rule = alt.Chart(pd.DataFrame({"iteration": [obj.mstop]})).mark_rule(strokeDash=[4, 4]).encode(x="iteration:Q")
    return alt.layer(line, point, rule).properties(width=width, height=height)


@plot.register
def _(obj: VarImpResult, *, width: int = 420, height: int = 260):
    alt, pd = _require_plot_deps()
    data = pd.DataFrame(obj.data)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("reduction:Q", title="In-bag risk reduction (%)" if obj.percent else "In-bag risk reduction"),
            y=alt.Y("label:N", sort="-x", title="Variable" if obj.type == "variable" else "Baselearner"),
            tooltip=["label:N", "reduction:Q", "selfreq:Q"],
        )
        .properties(width=width, height=height)
    )


@plot.register
def _(obj: ConfIntResult, *, width: int = 420, height: int = 260):
    alt, pd = _require_plot_deps()
    data = pd.DataFrame(obj.data)
    if obj.kind == "fitted":
        band = alt.Chart(data).mark_area(opacity=0.25, color="#4c78a8").encode(
            x=alt.X("observation:Q", title="Observation"),
            y=alt.Y("lower:Q", title="Estimate"),
            y2="upper:Q",
        )
        line = alt.Chart(data).mark_line(color="#1f77b4").encode(x="observation:Q", y="estimate:Q")
        return alt.layer(band, line).properties(width=width, height=height)
    if obj.kind == "partial":
        kinds = set(data["kind"].unique())
        if kinds == {"numeric"}:
            band = (
                alt.Chart(data)
                .mark_area(opacity=0.22, color="#4c78a8")
                .properties(width=width, height=height)
                .encode(x=alt.X("x:Q", title=None), y=alt.Y("lower:Q", title="Partial effect"), y2="upper:Q")
            )
            line = alt.Chart(data).mark_line(color="#1f77b4").properties(width=width, height=height).encode(x="x:Q", y="estimate:Q")
            return alt.layer(band, line).facet("term:N")
        if kinds == {"categorical"}:
            interval = alt.Chart(data).mark_rule(strokeWidth=3).properties(width=width, height=height).encode(
                x=alt.X("x:N", title=None),
                y=alt.Y("lower:Q", title="Partial effect"),
                y2="upper:Q",
            )
            points = alt.Chart(data).mark_point(color="#1f77b4", filled=True, size=70).properties(width=width, height=height).encode(x="x:N", y="estimate:Q")
            return alt.layer(interval, points).facet("term:N")
    raise NotImplementedError(f"plot is not implemented for ConfIntResult kind {obj.kind!r}")
