"""Shared utilities for the pymboost Jupyter Book.

This module consolidates what used to be duplicated setup blocks across every
book chapter:

* locating the repository root and putting it on ``sys.path``
* a single Altair theme + color palette
* small rpy2 helpers for moving frames and vectors between R and Python
* a library of reusable Altair chart factories that overlay Python vs R
  outputs — the core visual evidence the book is built on

Book chapters should import from here rather than re-defining the same
helpers locally. Keep this module small and stable; page-specific logic
should live in the page.

Polars is the first-class DataFrame type throughout. Altair 6 consumes
polars DataFrames natively via the narwhals interchange layer, so callers
should pass ``pl.DataFrame`` instances. The rpy2 helpers that need to
cross the pandas bridge (``r_assign_dataframe``) accept a polars frame and
convert internally; pandas is treated as an interop detail, not a public
type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    import altair as alt
    import numpy as np
    import pandas as pd
    import polars as pl


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

PYTHON_COLOR = "#4c78a8"  # Vega Tableau10 blue
R_COLOR = "#e45756"  # Vega Tableau10 red
NEUTRAL_COLOR = "#bab0ac"  # Vega Tableau10 grey
ACCENT_COLOR = "#54a24b"  # Vega Tableau10 green, used for deltas and fold means
WARN_COLOR = "#f58518"  # Vega Tableau10 orange, used for selected mstop markers

PALETTE = {
    "Python": PYTHON_COLOR,
    "R mboost": R_COLOR,
    "neutral": NEUTRAL_COLOR,
    "accent": ACCENT_COLOR,
    "warn": WARN_COLOR,
}

_SOURCE_DOMAIN = ["Python", "R mboost"]
_SOURCE_RANGE = [PYTHON_COLOR, R_COLOR]


# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #


def project_root() -> Path:
    """Return the repository root (the directory containing ``mboost/``).

    The search walks up from ``Path.cwd()`` until a directory containing a
    ``mboost`` subdirectory is found. This matches the pattern already used
    across the chapters and lets notebooks be executed from either the repo
    root or from ``book/``.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "mboost").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the pymboost project root. "
        "Expected to find a directory containing a 'mboost' subdirectory."
    )


def r_cache_dir() -> Path:
    """Return the directory used for cached R-side reference outputs."""
    d = project_root() / "book" / "_static" / "r_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """Return the repository ``data/`` directory."""
    return project_root() / "data"


# --------------------------------------------------------------------------- #
# One-call configuration
# --------------------------------------------------------------------------- #


def configure() -> Path:
    """Configure the book execution environment.

    * adds the repository root to ``sys.path`` so ``import mboost`` works
    * disables Altair's max-rows guard (book cells sometimes plot many rows)
    * registers a consistent Altair theme for the whole book

    Returns the project root so callers can use it to locate data files.
    """
    import sys

    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        import altair as alt

        alt.data_transformers.disable_max_rows()
        _register_altair_theme(alt)
    except ImportError:  # pragma: no cover - altair is a hard dep of the book
        pass

    return root


def _register_altair_theme(alt: "Any") -> None:
    theme_name = "pymboost_book"

    def theme() -> dict[str, Any]:
        return {
            "config": {
                "view": {"continuousWidth": 520, "continuousHeight": 260, "stroke": "transparent"},
                "title": {"fontSize": 14, "font": "Helvetica, Arial, sans-serif", "anchor": "start"},
                "axis": {
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "grid": True,
                    "gridOpacity": 0.35,
                    "domain": False,
                    "tickColor": "#888888",
                    "labelColor": "#333333",
                    "titleColor": "#222222",
                },
                "legend": {
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "orient": "top",
                    "labelColor": "#333333",
                },
                "range": {
                    "category": [PYTHON_COLOR, R_COLOR, "#72b7b2", ACCENT_COLOR, "#eeca3b", "#b279a2"],
                },
            }
        }

    # enable() is idempotent under Altair; register() is harmless on repeat.
    try:
        alt.themes.register(theme_name, theme)
        alt.themes.enable(theme_name)
    except Exception:  # pragma: no cover - altair API flux
        pass


# --------------------------------------------------------------------------- #
# Frame coercion helpers
# --------------------------------------------------------------------------- #


def _to_polars(df: "Any") -> "pl.DataFrame":
    """Return a polars DataFrame regardless of input type.

    Accepts polars, pandas, dict, or anything ``pl.DataFrame`` can wrap.
    The book treats polars as the canonical frame type; legacy call sites
    that still hold pandas frames are converted on entry.
    """
    import polars as pl

    if isinstance(df, pl.DataFrame):
        return df
    try:
        import pandas as pd  # noqa: WPS433

        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:  # pragma: no cover - pandas is optional
        pass
    return pl.DataFrame(df)


def _to_pandas(df: "Any") -> "pd.DataFrame":
    """Return a pandas DataFrame; used only for pandas-only interop points (rpy2)."""
    import pandas as pd
    import polars as pl

    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return pd.DataFrame(df)


# --------------------------------------------------------------------------- #
# rpy2 helpers
# --------------------------------------------------------------------------- #


def r_assign_dataframe(name: str, frame: "Any") -> None:
    """Push a DataFrame into R's global environment under ``name``.

    Accepts a polars or pandas DataFrame. The value goes through pandas
    only because rpy2's converter pipeline is pandas-native.
    """
    from rpy2 import robjects as ro
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv[name] = _to_pandas(frame)


def r_numeric(expr: str) -> "np.ndarray":
    """Evaluate an R expression and return a numpy float64 array."""
    import numpy as np
    from rpy2 import robjects as ro
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + numpy2ri.converter):
        return np.asarray(ro.r(expr), dtype=np.float64)


def r_named_vector(expr: str, value_name: str = "value") -> "pl.DataFrame":
    """Evaluate an R expression returning a named numeric vector.

    Returns a two-column polars DataFrame with columns ``term`` and ``value_name``.
    """
    import numpy as np
    import polars as pl
    from rpy2 import robjects as ro

    vec = ro.r(expr)
    return pl.DataFrame(
        {
            "term": [str(t) for t in vec.names],
            value_name: np.asarray(vec, dtype=np.float64),
        }
    )


def wide_compare(
    left: "Any",
    right: "Any",
    value_name: str = "coefficient",
    key: str = "term",
) -> "pl.DataFrame":
    """Join two frames on ``key`` and compute Python − R difference.

    Returns a polars DataFrame with columns ``term``, ``{value_name}_python``,
    ``{value_name}_r``, ``{value_name}_diff``.
    """
    import polars as pl

    left_pl = _to_polars(left).rename({value_name: f"{value_name}_python"})
    right_pl = _to_polars(right).rename({value_name: f"{value_name}_r"})
    merged = left_pl.join(right_pl, on=key, how="inner")
    return merged.with_columns(
        (pl.col(f"{value_name}_python") - pl.col(f"{value_name}_r")).alias(f"{value_name}_diff")
    )


def r_load_library(name: str) -> None:
    """Load an R package by name (equivalent to ``library(name)``)."""
    from rpy2 import robjects as ro

    ro.r(f"suppressPackageStartupMessages(library({name}))")


# --------------------------------------------------------------------------- #
# Reusable Altair chart factories (Python vs R overlays)
# --------------------------------------------------------------------------- #


def _source_color(
    field: str = "source",
    domain: Sequence[str] = _SOURCE_DOMAIN,
    range_: Sequence[str] = _SOURCE_RANGE,
    title: str | None = None,
) -> "alt.Color":
    import altair as alt

    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=list(domain), range=list(range_)),
        title=title,
    )


def parity_scatter(
    df: "Any",
    *,
    python_col: str = "Python",
    r_col: str = "R mboost",
    title: str = "Python vs R parity",
    width: int = 360,
    height: int = 360,
) -> "alt.Chart":
    """Scatter of Python vs R fitted values with a y=x reference line.

    Accepts polars or pandas; internally it operates on a polars DataFrame
    that is then passed directly to Altair 6 (which consumes polars
    natively via narwhals).
    """
    import altair as alt
    import polars as pl

    data = _to_polars(df)
    lo = float(min(data[python_col].min(), data[r_col].min()))
    hi = float(max(data[python_col].max(), data[r_col].max()))
    line = pl.DataFrame({"v": [lo, hi]})

    scatter = (
        alt.Chart(data)
        .mark_circle(size=55, opacity=0.85, color=PYTHON_COLOR)
        .encode(
            x=alt.X(f"{r_col}:Q", title="R fitted value"),
            y=alt.Y(f"{python_col}:Q", title="Python fitted value"),
            tooltip=list(data.columns),
        )
    )
    reference = (
        alt.Chart(line)
        .mark_line(strokeDash=[4, 4], color=NEUTRAL_COLOR)
        .encode(x="v:Q", y="v:Q")
    )
    return (reference + scatter).properties(width=width, height=height, title=title)


def risk_path_chart(
    df: "Any",
    *,
    iteration_col: str = "iteration",
    risk_col: str = "risk",
    source_col: str = "source",
    title: str = "Boosting risk path",
    width: int = 520,
    height: int = 240,
) -> "alt.Chart":
    """Line chart of empirical risk vs iteration, grouped by source (Python / R)."""
    import altair as alt

    data = _to_polars(df)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(f"{iteration_col}:Q", title="Boosting iteration"),
            y=alt.Y(f"{risk_col}:Q", title="Empirical risk"),
            color=_source_color(field=source_col),
        )
        .properties(width=width, height=height, title=title)
    )


def coefficient_bar(
    df: "Any",
    *,
    term_col: str = "term",
    coef_col: str = "coefficient",
    source_col: str = "source",
    title: str = "Coefficients",
    width: int = 560,
    height: int = 220,
) -> "alt.Chart":
    """Grouped bar chart of coefficients by term, split by source."""
    import altair as alt

    data = _to_polars(df)
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X(f"{coef_col}:Q", title="Coefficient"),
            y=alt.Y(f"{term_col}:N", title=None, sort=None),
            color=_source_color(field=source_col),
            yOffset=f"{source_col}:N",
            tooltip=[term_col, source_col, coef_col],
        )
        .properties(width=width, height=height, title=title)
    )


def partial_effect_overlay(
    df: "Any",
    *,
    feature_col: str = "feature",
    x_col: str = "x",
    effect_col: str = "effect",
    source_col: str = "source",
    title: str = "Partial effects: Python vs R mboost",
    facet_columns: int | None = None,
    width: int = 220,
    height: int = 180,
) -> "alt.Chart":
    """Faceted overlay of per-feature partial-effect curves (Python vs R).

    The long-form DataFrame must have one row per (feature, x, source) triple.
    Python and R curves for each feature are drawn together with the shared
    book palette. Facets wrap automatically so the chart grows horizontally
    rather than shrinking individual panels.
    """
    import altair as alt

    data = _to_polars(df)
    base = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(f"{x_col}:Q", title=None),
            y=alt.Y(f"{effect_col}:Q", title="Partial effect"),
            color=_source_color(field=source_col),
        )
        .properties(width=width, height=height)
    )
    facet_kwargs: dict[str, Any] = {"title": title}
    if facet_columns is not None:
        facet_kwargs["columns"] = facet_columns
    return base.facet(facet=alt.Facet(f"{feature_col}:N", title=None), **facet_kwargs)


def coefficient_path_overlay(
    df: "Any",
    *,
    iteration_col: str = "iteration",
    value_col: str = "coefficient",
    term_col: str = "term",
    source_col: str = "source",
    title: str = "Coefficient paths: Python vs R mboost",
    width: int = 520,
    height: int = 280,
) -> "alt.Chart":
    """Line chart of coefficient paths versus boosting iteration.

    Each ``term`` gets one line per source. R curves are drawn with
    dashed strokes so the overlay is legible even when the two implementations
    agree to numerical precision (the lines would otherwise coincide).
    """
    import altair as alt

    data = _to_polars(df)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{iteration_col}:Q", title="Boosting iteration"),
            y=alt.Y(f"{value_col}:Q", title="Coefficient"),
            color=alt.Color(f"{term_col}:N", title="Term"),
            strokeDash=alt.StrokeDash(
                f"{source_col}:N",
                scale=alt.Scale(
                    domain=list(_SOURCE_DOMAIN),
                    range=[[1, 0], [5, 3]],
                ),
                title="Source",
            ),
            tooltip=[term_col, source_col, iteration_col, value_col],
        )
        .properties(width=width, height=height, title=title)
    )


def cvrisk_fold_overlay(
    folds_df: "Any",
    mean_df: "Any",
    *,
    iteration_col: str = "iteration",
    value_col: str = "risk",
    fold_col: str = "fold",
    source_col: str = "source",
    selected: Mapping[str, int] | None = None,
    title: str = "CV risk: Python vs R mboost",
    width: int = 560,
    height: int = 300,
) -> "alt.Chart":
    """Fold-fan chart for ``cvrisk``.

    ``folds_df`` has one row per (source, fold, iteration); ``mean_df`` has
    one row per (source, iteration) carrying the per-iteration mean risk.
    If ``selected`` is provided it is a mapping ``{"Python": mstop_py,
    "R mboost": mstop_r}`` and a rule is drawn at each selected iteration.
    """
    import altair as alt
    import polars as pl

    folds = _to_polars(folds_df)
    means = _to_polars(mean_df)

    fold_layer = (
        alt.Chart(folds)
        .mark_line(opacity=0.25, strokeWidth=1)
        .encode(
            x=alt.X(f"{iteration_col}:Q", title="Boosting iteration"),
            y=alt.Y(f"{value_col}:Q", title="Held-out risk"),
            color=_source_color(field=source_col),
            detail=f"{fold_col}:N",
        )
    )
    mean_layer = (
        alt.Chart(means)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X(f"{iteration_col}:Q"),
            y=alt.Y(f"{value_col}:Q"),
            color=_source_color(field=source_col),
        )
    )
    layers: list[alt.Chart] = [fold_layer, mean_layer]

    if selected:
        rule_rows = [
            {"source": key, "iteration": int(value)} for key, value in selected.items()
        ]
        rule_df = pl.DataFrame(rule_rows)
        rules = (
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
            .encode(
                x=alt.X("iteration:Q"),
                color=_source_color(field="source"),
            )
        )
        layers.append(rules)

    return alt.layer(*layers).properties(width=width, height=height, title=title)


def aic_curve_overlay(
    df: "Any",
    *,
    iteration_col: str = "iteration",
    value_col: str = "aic",
    source_col: str = "source",
    selected: Mapping[str, int] | None = None,
    title: str = "Corrected AIC: Python vs R mboost",
    width: int = 520,
    height: int = 260,
) -> "alt.Chart":
    """Line chart of corrected-AIC curves for Python vs R.

    Draws the curve for each source and, if ``selected`` is supplied, vertical
    rules at the AIC-selected iteration per source.
    """
    import altair as alt
    import polars as pl

    data = _to_polars(df)
    curve = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(f"{iteration_col}:Q", title="Boosting iteration"),
            y=alt.Y(f"{value_col}:Q", title="Corrected AIC"),
            color=_source_color(field=source_col),
        )
    )
    layers: list[alt.Chart] = [curve]
    if selected:
        rule_df = pl.DataFrame(
            [{"source": k, "iteration": int(v)} for k, v in selected.items()]
        )
        rules = (
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
            .encode(x=alt.X("iteration:Q"), color=_source_color(field="source"))
        )
        layers.append(rules)
    return alt.layer(*layers).properties(width=width, height=height, title=title)


def varimp_bar_overlay(
    df: "Any",
    *,
    term_col: str = "term",
    value_col: str = "reduction",
    source_col: str = "source",
    title: str = "Variable importance: Python vs R mboost",
    width: int = 520,
    height: int = 260,
) -> "alt.Chart":
    """Horizontal bar chart of per-term risk reduction, split by source."""
    import altair as alt

    data = _to_polars(df)
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X(f"{value_col}:Q", title="Risk reduction"),
            y=alt.Y(f"{term_col}:N", title=None, sort="-x"),
            color=_source_color(field=source_col),
            yOffset=f"{source_col}:N",
            tooltip=[term_col, source_col, value_col],
        )
        .properties(width=width, height=height, title=title)
    )


def prediction_grid_overlay(
    df: "Any",
    *,
    x_col: str = "x",
    value_col: str = "prediction",
    source_col: str = "source",
    title: str = "Prediction on grid: Python vs R mboost",
    facet_col: str | None = None,
    width: int = 520,
    height: int = 260,
) -> "alt.Chart":
    """Line overlay of Python and R predictions on a shared grid.

    Designed for tree learners where coefficient-level parity is not
    achievable but prediction-level shape can still be compared.
    """
    import altair as alt

    data = _to_polars(df)
    base = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5, interpolate="step-after")
        .encode(
            x=alt.X(f"{x_col}:Q", title=None),
            y=alt.Y(f"{value_col}:Q", title="Prediction"),
            color=_source_color(field=source_col),
        )
    )
    if facet_col is not None:
        return base.properties(width=width, height=height).facet(
            facet=alt.Facet(f"{facet_col}:N", title=None), title=title
        )
    return base.properties(width=width, height=height, title=title)


def diff_heatmap(
    df: "Any",
    *,
    row_col: str,
    col_col: str,
    value_col: str = "diff",
    title: str = "Python − R difference",
    width: int = 520,
    height: int = 260,
) -> "alt.Chart":
    """Signed-difference heatmap, typically Python minus R on a 2-D grid.

    The color scale is a diverging blue–white–red map centred at zero so
    parity-quality (diffs near zero) shows up as near-white.
    """
    import altair as alt

    data = _to_polars(df)
    max_abs = float(data[value_col].abs().max() or 1.0)
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(f"{col_col}:O", title=None),
            y=alt.Y(f"{row_col}:O", title=None),
            color=alt.Color(
                f"{value_col}:Q",
                title="Δ",
                scale=alt.Scale(
                    scheme="redblue",
                    domain=[-max_abs, 0, max_abs],
                    reverse=True,
                ),
            ),
            tooltip=list(data.columns),
        )
        .properties(width=width, height=height, title=title)
    )


def parity_badge(max_abs_diff: float, *, kind: str = "fitted") -> str:
    """Return a short Markdown-safe parity badge like '**max |Δ| = 4.2e-13**'.

    Used in chapter prose so the narrative line that introduces each parity
    panel can carry the numeric parity quality inline, without forcing the
    reader to compute it from the chart.
    """
    return f"**max |Δ {kind}| = {max_abs_diff:.2e}**"


# --------------------------------------------------------------------------- #
# Cached R outputs
# --------------------------------------------------------------------------- #


def load_cached_r_json(name: str) -> dict[str, Any]:
    """Load a cached R reference output by name from ``book/_static/r_cache``.

    These JSON files are produced by ``scripts/refresh_book_assets.py`` so that
    book cells can consume pre-computed R outputs without invoking rpy2 at
    build time. If the cache does not exist, chapters can fall back to a live
    rpy2 call via the helpers above.
    """
    path = r_cache_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached R reference output named {name!r}. "
            "Run scripts/refresh_book_assets.py to populate the cache."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_r_json(name: str, payload: dict[str, Any]) -> Path:
    """Write a dict of R reference outputs to the cache (used by the refresh script)."""
    path = r_cache_dir() / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


__all__ = [
    "ACCENT_COLOR",
    "NEUTRAL_COLOR",
    "PALETTE",
    "PYTHON_COLOR",
    "R_COLOR",
    "WARN_COLOR",
    "aic_curve_overlay",
    "coefficient_bar",
    "coefficient_path_overlay",
    "configure",
    "cvrisk_fold_overlay",
    "data_dir",
    "diff_heatmap",
    "load_cached_r_json",
    "parity_badge",
    "parity_scatter",
    "partial_effect_overlay",
    "prediction_grid_overlay",
    "project_root",
    "r_assign_dataframe",
    "r_cache_dir",
    "r_load_library",
    "r_named_vector",
    "r_numeric",
    "risk_path_chart",
    "save_cached_r_json",
    "varimp_bar_overlay",
    "wide_compare",
]
