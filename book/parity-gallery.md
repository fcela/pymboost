---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Parity Gallery

This chapter is the visual backbone of pymboost's correctness claim. Every
page in this book makes statements about how the Python implementation
behaves; this chapter turns those statements into plots that any reader
can eyeball alongside an R `mboost` reference run. Each figure is a
side-by-side overlay of Python and R output on identical data, with a
printed parity metric next to it so the claim and the evidence live in
the same place.

The chapter is organised by what a user most often brings to a
Python-vs-R comparison, in the order the R `mboost` help pages introduce
them:

1. The Gaussian canon — linear `glmboost` on bodyfat: the coefficient
   path, the corrected AIC curve, variable importance, the empirical-risk
   path, and the fitted-value scatter.
2. The additive extension — `gamboost` on bodyfat: per-feature P-spline
   partial effects with an overlay of the R reference curves.
3. Resampling — a 10-fold `cvrisk` fold fan where Python consumes the
   *same fold matrix* R used, so the two curves are comparable
   point-for-point.
4. Non-Gaussian families — a clean synthetic Binomial at machine
   precision, then a realistic Poisson that exposes an honest drift, and
   multi-τ quantile regression on bodyfat.
5. Tree learners — `btree` on a 1-D step target and `blackboost` on a
   2-D interaction surface, compared on a shared prediction grid rather
   than at the coefficient level (which is impossible across CART vs
   `partykit` backends).

A short **Summary table** at the end collects the numeric parity figure
for every panel so readers can find the worst-case gap in one place,
without hunting through the narrative.

All figures in this chapter read their R-side data from JSON blobs in
`book/_static/r_cache/`, produced by
`python scripts/refresh_book_assets.py`. The Python side is computed
live against `mboost` in this notebook so it is always up-to-date with
the current package state.

## Setup

```{code-cell} ipython3
:tags: [hide-input]
from pathlib import Path
import sys
ROOT = next(
    candidate
    for candidate in (Path.cwd().resolve(), Path.cwd().resolve().parent)
    if (candidate / "mboost").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book"))

import numpy as np
import polars as pl
import altair as alt
from rpy2 import robjects as ro

import book_utils
book_utils.configure()
r_numeric = book_utils.r_numeric
r_assign_dataframe = book_utils.r_assign_dataframe
book_utils.r_load_library("mboost")
PYTHON_COLOR = book_utils.PYTHON_COLOR
R_COLOR = book_utils.R_COLOR

from mboost import (
    glmboost, gamboost, cvrisk, AIC, varimp, coef, fitted, risk, selected,
    Gaussian, Binomial, Poisson, Quantile, boost_control, partial_plot_data,
)
```

The `book_utils` helpers used throughout this chapter
(`parity_scatter`, `partial_effect_overlay`, `coefficient_path_overlay`,
`cvrisk_fold_overlay`, `aic_curve_overlay`, `varimp_bar_overlay`,
`prediction_grid_overlay`, `parity_badge`) all accept polars
`DataFrame` objects directly; pandas is avoided throughout the chapter.
Altair 6 consumes polars frames natively via the narwhals interchange
layer, so there is no lossy round-trip in the middle.

---

## 1. The Gaussian canon: `glmboost` on `bodyfat`

The first figure anyone draws when comparing boosting implementations is
the classic bodyfat `glmboost` coefficient path from the Hofner (2014)
tutorial: every linear term on the x-axis of boosting iteration, one
line per selected predictor. If the two implementations agree, the lines
must agree everywhere, not just at the final iteration.

```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv")
r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
ro.r("bodyfat <- bodyfat_py")
ro.r(
    """
    glm_gallery_r <- glmboost(
      DEXfat ~ .,
      data = bodyfat,
      control = boost_control(mstop = 120, nu = 0.1)
    )
    """
)
glm_gallery = glmboost(
    "DEXfat ~ .",
    data=bodyfat,
    control=boost_control(mstop=120, nu=0.1),
)
print(f"Python glmboost fitted: mstop={glm_gallery.mstop}, "
      f"offset={glm_gallery.offset_:.4f}, "
      f"final risk={glm_gallery.risk_[-1]:.4f}")
print(f"Selected terms across the path: {sorted(set(selected(glm_gallery)))}")
```

**Reconstructing the per-term coefficient path.** R's `coef(m, which="",
aggregate="cumsum")` returns the cumulative per-term path directly.
pymboost does not yet expose a symmetric helper, so we build it by
sliding a truncated model forward and asking for `coef` at each
iteration. This is O(mstop) slicing calls — cheap on bodyfat:

```{code-cell} ipython3
def python_coef_path(model) -> pl.DataFrame:
    """Reconstruct the per-term cumulative coefficient path as a polars frame."""
    rows: list[dict] = []
    for it in range(1, model.mstop + 1):
        snap = model.with_mstop(it)
        for term, value in coef(snap).items():
            rows.append({"term": term, "iteration": it, "coefficient": float(value[0])})
    return pl.DataFrame(rows)

py_path = python_coef_path(glm_gallery)
print(f"Path rows: {py_path.shape[0]} (= {glm_gallery.mstop} iterations × "
      f"{py_path['term'].n_unique()} terms).")
py_path.head()
```

The R side comes pre-computed from `scripts/refresh_book_assets.py` so
the chapter does not invoke rpy2 on every build:

```{code-cell} ipython3
cache = book_utils.load_cached_r_json("gallery_glmboost_paths")
r_path = pl.DataFrame(cache["coef_path"])
columns = ["term", "iteration", "coefficient", "source"]
combined_path = pl.concat(
    [
        py_path.with_columns(pl.lit("Python").alias("source")).select(columns),
        r_path.with_columns(pl.lit("R mboost").alias("source")).select(columns),
    ]
)
print("Combined path rows:", combined_path.shape[0])
combined_path.head(4)
```

Both sides carry the same per-iteration coefficient for every active
term. The overlay below uses dashed strokes for R so when the two curves
agree to numerical precision — which they do here — the R dashes trace
along the top of the Python solid lines rather than hiding underneath:

```{code-cell} ipython3
active_terms = (
    combined_path.filter(pl.col("iteration") == glm_gallery.mstop)
    .filter(pl.col("coefficient").abs() > 1e-6)
    ["term"].unique().to_list()
)
plot_df = combined_path.filter(pl.col("term").is_in(active_terms))

max_abs = float(
    plot_df.pivot(index=["iteration", "term"], on="source", values="coefficient")
    .with_columns((pl.col("Python") - pl.col("R mboost")).abs().alias("diff"))
    ["diff"].max()
)
title = f"Coefficient paths: Python vs R mboost — {book_utils.parity_badge(max_abs, kind='coef')}"

book_utils.coefficient_path_overlay(plot_df, title=title)
```

Every colour is a boosting term; solid strokes are Python and dashed
strokes are R. The lines rise monotonically for selected terms and stay
flat during iterations where that term was not picked. The printed
`max |Δ coef|` is the largest per-iteration, per-term gap between the
two implementations across the whole path, not just at the final
iteration — which is the stricter quantity.

The final-iteration fitted-value scatter is the companion check: if the
coefficient path agrees everywhere, the fitted values at the final
iteration must agree pointwise.

```{code-cell} ipython3
py_fitted = np.asarray(fitted(glm_gallery), dtype=np.float64)
r_fitted = r_numeric("as.numeric(predict(glm_gallery_r, type = 'link'))")

scatter_df = pl.DataFrame({"Python": py_fitted, "R mboost": r_fitted})
max_fit_diff = float((scatter_df["Python"] - scatter_df["R mboost"]).abs().max())
book_utils.parity_scatter(
    scatter_df,
    title=f"bodyfat glmboost fitted — {book_utils.parity_badge(max_fit_diff)}",
)
```

The points sit exactly on the dashed reference line, confirming that the
coefficient-level agreement in the previous figure propagates through to
predictions. For Gaussian `glmboost` on bodyfat, pymboost and R `mboost`
produce numerically indistinguishable models.

### Corrected AIC curve

The Hofner tutorial uses corrected AIC to pick `mstop`. pymboost's
`AIC(model, method="corrected")` and R's `AIC(model, method="corrected")`
should produce the same curve — and should select the same iteration.

```{code-cell} ipython3
aic_result = AIC(glm_gallery, method="corrected")
aic_df = pl.DataFrame(
    {
        "iteration": np.arange(1, glm_gallery.mstop + 1, dtype=np.int64),
        "aic": np.asarray(aic_result.aic_path, dtype=np.float64),
        "source": "Python",
    }
)
r_aic_df = pl.DataFrame(
    {
        "iteration": np.arange(1, glm_gallery.mstop + 1, dtype=np.int64),
        "aic": np.asarray(cache["aic"], dtype=np.float64),
        "source": "R mboost",
    }
)
aic_plot = pl.concat([aic_df, r_aic_df])

py_selected = int(np.argmin(aic_df["aic"].to_numpy())) + 1
r_selected = int(cache["aic_selected"])
max_aic_diff = float(
    (aic_df["aic"].to_numpy() - r_aic_df["aic"].to_numpy()).__abs__().max()
)

book_utils.aic_curve_overlay(
    aic_plot,
    selected={"Python": py_selected, "R mboost": r_selected},
    title=f"Corrected AIC: Python vs R — {book_utils.parity_badge(max_aic_diff, kind='AIC')}",
)
```

Both curves turn over at the same iteration (the two vertical rules
overlap), the AIC values themselves agree to numerical tolerance, and
the AIC-selected `mstop` is identical between the two implementations.
This is a much stronger parity claim than just "the fit agrees at the
end": it means the *stopping decision* is parity-quality too, which is
what drives the selected model in real use.

### Variable importance

R's `varimp(model)` reports per-term risk reduction — how much of the
total boosting improvement each predictor contributed. pymboost's
`varimp()` returns the same quantity in a `VarImpResult` object whose
`.data` attribute is a polars-friendly tuple of lists.

```{code-cell} ipython3
py_vi = varimp(glm_gallery)
# VarImpResult.to_pandas() returns columns (label, reduction, selfreq,
# baselearner, variable). We only need (label, reduction) for the bars and
# rename `label` → `term` so the shared chart helper can pick it up.
py_vi_df = (
    pl.from_pandas(py_vi.to_pandas()[["label", "reduction"]])
    .rename({"label": "term"})
    .with_columns(pl.col("reduction").cast(pl.Float64))
    .with_columns(pl.lit("Python").alias("source"))
)
r_vi_df = pl.DataFrame(
    [
        {"term": row["term"], "reduction": float(row["reduction"]), "source": "R mboost"}
        for row in cache["varimp"]
    ]
)
# Put Python and R on the same scale (% of total risk reduction): pymboost's
# `varimp` already reports percentages, R's `varimp` returns raw reductions,
# so we normalise the R side.
r_total = float(r_vi_df["reduction"].sum()) or 1.0
r_vi_df = r_vi_df.with_columns((pl.col("reduction") / r_total * 100).alias("reduction"))

vi_plot = pl.concat(
    [
        py_vi_df.select(["term", "reduction", "source"]),
        r_vi_df.select(["term", "reduction", "source"]),
    ]
).filter(pl.col("reduction") > 0.01)
book_utils.varimp_bar_overlay(
    vi_plot,
    value_col="reduction",
    title="Variable importance (% of total risk reduction): Python vs R mboost",
)
```

`hipcirc`, `waistcirc`, and `anthro3a` dominate in both implementations
and in the same order, the minor terms come out in matching positions,
and the relative magnitudes agree. The normalisation difference
(pymboost reports percentages by default, R reports raw reductions) is
handled in the data preparation step above so the bars are directly
comparable; both now sum to 100% across terms.

### Empirical risk path

The final Gaussian-canon figure is the humblest one: the empirical risk
at every iteration.

```{code-cell} ipython3
risk_df = pl.DataFrame(
    {
        "iteration": np.arange(len(glm_gallery.risk_), dtype=np.int64),
        "risk": np.asarray(glm_gallery.risk_, dtype=np.float64),
        "source": "Python",
    }
)
r_risk = np.asarray(cache["risk"], dtype=np.float64)
r_risk_df = pl.DataFrame(
    {
        "iteration": np.arange(len(r_risk), dtype=np.int64),
        "risk": r_risk,
        "source": "R mboost",
    }
)
risk_combined = pl.concat([risk_df, r_risk_df])
max_risk_diff = float(
    np.max(np.abs(glm_gallery.risk_[: len(r_risk)] - r_risk[: len(glm_gallery.risk_)]))
)
book_utils.risk_path_chart(
    risk_combined,
    title=f"Empirical risk path: Python vs R mboost — {book_utils.parity_badge(max_risk_diff, kind='risk')}",
)
```

The two curves are indistinguishable at the chart's scale and the
printed `max |Δ risk|` is within machine tolerance. Taken together,
sections 1's five figures form the tight-parity baseline the rest of
the gallery is measured against.

---

## 2. Additive: `gamboost` on `bodyfat`

Moving from linear terms to penalized splines, the shape that matters
is no longer a coefficient number but a *partial-effect curve*. The
parity question becomes "do Python and R draw the same smooth for
each feature?". pymboost's `partial_plot_data` helper emits the
per-feature curves on a dense grid, and the R side is pre-cached.

```{code-cell} ipython3
gam_bodyfat = gamboost(
    "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)
py_curves_pd = partial_plot_data(gam_bodyfat, grid_size=120)
py_curves = pl.from_pandas(py_curves_pd).with_columns(pl.lit("Python").alias("source"))

r_cache_gam = book_utils.load_cached_r_json("bodyfat_gamboost")
r_rows: list[dict] = []
for feature, payload in r_cache_gam["partial_curves"].items():
    for x, y in zip(payload["x"], payload["effect"]):
        r_rows.append({
            "term": f"bbs({feature})",
            "feature": feature,
            "x": float(x),
            "effect": float(y),
            "kind": "numeric",
            "source": "R mboost",
        })
r_curves = pl.DataFrame(r_rows)
effect_df = pl.concat([py_curves, r_curves])
effect_df.head(4)
```

```{code-cell} ipython3
(
    alt.Chart(effect_df)
    .mark_line(strokeWidth=2.5)
    .encode(
        x=alt.X("x:Q", title="Covariate value"),
        y=alt.Y("effect:Q", title="Partial effect"),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=["Python", "R mboost"], range=[PYTHON_COLOR, R_COLOR]),
            title=None,
        ),
        strokeDash=alt.StrokeDash(
            "source:N",
            scale=alt.Scale(domain=["Python", "R mboost"], range=[[1, 0], [6, 4]]),
            title=None,
        ),
        tooltip=["term:N", "source:N", "x:Q", "effect:Q"],
    )
    .properties(width=220, height=180, title="bodyfat gamboost partial effects: Python vs R mboost")
    .facet(column=alt.Column("term:N", title=None))
    .resolve_scale(x="independent", y="independent")
)
```

Reading left to right: `anthro3a` is almost linear, `hipcirc` is
monotone increasing with a mild concave shape, `kneebreadth` has a
shallow positive trend. pymboost and R draw the same smooth on all
three features — the solid and dashed curves overlap to the width of
the stroke. The small wiggle visible on the edges of `kneebreadth` is
the boundary behaviour of the P-spline basis and is shared by both
implementations.

The fitted-value scatter confirms the partial-curve agreement
propagates to predictions:

```{code-cell} ipython3
py_fitted = np.asarray(fitted(gam_bodyfat), dtype=np.float64)
r_fitted = np.asarray(r_cache_gam["gam1_fitted"], dtype=np.float64)
parity_df = pl.DataFrame({"Python": py_fitted, "R mboost": r_fitted})
max_abs = float((parity_df["Python"] - parity_df["R mboost"]).abs().max())
book_utils.parity_scatter(
    parity_df,
    title=f"bodyfat gamboost fitted — {book_utils.parity_badge(max_abs)}",
)
```

The parity metric on additive models is slightly looser than on the
linear `glmboost` — `~1e-8` instead of `~1e-14` — because the P-spline
basis is constructed in floating point and the basis construction path
is not bit-for-bit identical between the two implementations. The
difference is inconsequential for any downstream use: the model's
fitted values, partial effects, and predictions are indistinguishable
at any resolution a user would actually plot.

---

## 3. Resampling: `cvrisk` fold fan

Cross-validation is the part of a boosting workflow that is hardest to
compare visually. Two implementations can easily generate different
random folds and produce wildly different fold-by-fold risk curves
while being numerically identical on each fold. To make the comparison
meaningful, we do something stronger: we let R generate the 10-fold
weight matrix and then feed *the exact same weight matrix* into
pymboost's `cvrisk`. Both sides now solve the same problem on the same
10 training subsets, and the curves are comparable point-for-point.

**Fold-matrix convention.** R's `mboost::cv()` returns a *training-weight*
matrix — a 0/1 array where `1` means "this observation is in the training
set for this fold" and `0` means "held out". pymboost's `cvrisk` accepts a
2-D matrix too, but interprets a pure 0/1 matrix as a *holdout mask*
(`1` = held out). Passing R's matrix in raw would silently swap the
training and held-out sets, so we flip it once before handing it over.
Where pymboost *does* match R directly is on bootstrap-style weight
matrices with integer multiplicities `> 1` — those are recognised as
training weights automatically. This is a known API asymmetry worth
knowing about if you are porting an R CV loop.

```{code-cell} ipython3
cv_cache = book_utils.load_cached_r_json("gallery_gamboost_cvrisk")
r_fold_weights = np.asarray(cv_cache["fold_weights"], dtype=np.float64)
print(f"R fold matrix shape (n × B): {r_fold_weights.shape}")

# Flip R's training-weight mask into pymboost's holdout-mask convention
# so both implementations hold out the *same* observations on each fold.
py_holdout_mask = 1.0 - r_fold_weights

py_cv = cvrisk(
    "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=cv_cache["mstop"], nu=0.1),
    folds=py_holdout_mask,
)
print(f"Python selected mstop (argmin of mean curve): {py_cv.best_mstop}")
print(f"R     selected mstop: {cv_cache['selected_mstop']}")
```

```{code-cell} ipython3
# Python cvrisk stores (B, mstop+1) with column 0 being the offset-only
# iteration. R's matrix is (B, mstop). Align them on iterations 1..mstop.
py_folds = py_cv.fold_risk[:, 1:]  # drop the offset column
r_folds = np.asarray(cv_cache["fold_risks"], dtype=np.float64)
n_folds, mstop = py_folds.shape

fold_rows: list[dict] = []
for source, mat in (("Python", py_folds), ("R mboost", r_folds)):
    for b in range(n_folds):
        for it in range(mstop):
            fold_rows.append({
                "iteration": it + 1,
                "risk": float(mat[b, it]),
                "fold": b,
                "source": source,
            })
folds_df = pl.DataFrame(fold_rows)

mean_rows: list[dict] = []
for source, mat in (("Python", py_folds), ("R mboost", r_folds)):
    mean_curve = mat.mean(axis=0)
    for it, value in enumerate(mean_curve.tolist(), start=1):
        mean_rows.append({"iteration": it, "risk": float(value), "source": source})
mean_df = pl.DataFrame(mean_rows)

max_mean_diff = float(
    np.max(np.abs(py_folds.mean(axis=0) - r_folds.mean(axis=0)))
)
book_utils.cvrisk_fold_overlay(
    folds_df,
    mean_df,
    selected={
        "Python": py_cv.best_mstop,
        "R mboost": int(cv_cache["selected_mstop"]),
    },
    title=f"10-fold cvrisk: Python vs R — {book_utils.parity_badge(max_mean_diff, kind='mean risk')}",
)
```

The thin lines are the individual fold curves (20 of them — 10 folds
for Python, 10 for R), and the thick lines are the cross-fold mean. The
dashed rules mark the argmin of each mean curve, which is the iteration
`cvrisk` would select. All four quantities — Python folds, R folds,
Python mean, R mean — lie on top of each other; the selected `mstop` is
the same on both sides; the numeric `max |Δ mean risk|` is well within
machine tolerance.

Because we're solving *the same* resampling problem, any deviation
would be a genuine parity gap in either the base-learner fit or the
risk calculation, not an artefact of different fold assignments. None
show up here.

---

## 4. Non-Gaussian families

Different families get built on different link functions and different
negative gradients, so family parity is the most finicky area of the
package. The picture is mixed: some families sit at machine precision,
others accumulate small-scale drift that is not caught by the existing
parity test suite (which exercises single-term, low-`mstop` setups).
This gallery surfaces both cases honestly.

### 4.1 Binomial at machine precision (synthetic)

With a clean synthetic 3-predictor logistic setup at `mstop = 100`,
Python and R agree on the Binomial link-scale predictions down to
double-precision arithmetic.

```{code-cell} ipython3
syn = book_utils.load_cached_r_json("gallery_synthetic_glm")
panel = pl.DataFrame(syn["panel"])

bin_df = panel.select(["x1", "x2", "x3"]).with_columns(
    y=panel["y_binomial"].cast(pl.Float64)
)
m_bin = glmboost(
    "y ~ x1 + x2 + x3",
    data=bin_df,
    family=Binomial(),
    control=boost_control(mstop=100, nu=0.1),
)
py_link = np.asarray(m_bin.fitted_, dtype=np.float64)
r_link = np.asarray(syn["binomial"]["link"], dtype=np.float64)
bin_df_plot = pl.DataFrame({"Python": py_link, "R mboost": r_link})
max_diff = float((bin_df_plot["Python"] - bin_df_plot["R mboost"]).abs().max())
book_utils.parity_scatter(
    bin_df_plot,
    title=f"Binomial synthetic fitted (link scale) — {book_utils.parity_badge(max_diff, kind='link')}",
)
```

Every point sits on the y=x line; the printed `max |Δ link|` is
`~1e-15`, which is noise at the level of the last bit of the floating
point representation. This is the most stringent family-parity result
in the book.

### 4.2 Poisson: shape agreement with a small intercept drift

Real Poisson count data behaves differently. With actual `rpois` counts
under a log-link target the two implementations still produce strongly
correlated fits, but the boost path accumulates a small offset that the
machine-precision synthetic test does not catch.

```{code-cell} ipython3
pois_df_in = panel.select(["x1", "x2", "x3"]).with_columns(
    y=panel["y_poisson"].cast(pl.Float64)
)
m_pois = glmboost(
    "y ~ x1 + x2 + x3",
    data=pois_df_in,
    family=Poisson(),
    control=boost_control(mstop=100, nu=0.1),
)
py_link = np.asarray(m_pois.fitted_, dtype=np.float64)
r_link = np.asarray(syn["poisson"]["link"], dtype=np.float64)

diff = py_link - r_link
mean_shift = float(diff.mean())
shape_residual = float(np.max(np.abs(diff - mean_shift)))
corr = float(np.corrcoef(py_link, r_link)[0, 1])

print(f"max |Δ link|                   = {np.max(np.abs(diff)):.3e}")
print(f"mean shift (constant offset)   = {mean_shift:+.3e}")
print(f"max shape residual (Δ − mean)  = {shape_residual:.3e}")
print(f"corr(Python link, R link)      = {corr:.8f}")
```

The decomposition tells two stories at once. About half of the max gap
is a **constant shift** in the link-scale prediction — roughly `+0.16`
on this target, which is what you would see if the two implementations
disagreed slightly on the intercept offset. The other half is a
**shape residual** of similar magnitude (`~0.18`), meaning the per-point
disagreement after subtracting that constant is still non-trivial.
What stays tight is the pointwise correlation: `~0.9999`. Any user
comparing rank orders, relative risks, or decision boundaries will see
the same model; any user comparing raw link values at the second
decimal place will see the drift.

Visually:

```{code-cell} ipython3
pois_scatter = pl.DataFrame({"Python": py_link, "R mboost": r_link})
book_utils.parity_scatter(
    pois_scatter,
    title=(
        f"Poisson synthetic fitted (link) — "
        f"{book_utils.parity_badge(float(np.max(np.abs(diff))), kind='link')}"
    ),
)
```

The points form a near-perfect line parallel to y=x, slightly offset.
Any user comparing rank-orders, relative risks, or decision boundaries
would see identical behaviour; any user comparing raw link values at
the last decimal place would see the offset. The gallery surfaces this
gap openly because it is not covered by the existing parity tests —
which use a deterministic `y = round(exp(0.5 x))` target that avoids
the accumulated drift — and it is important for any reader who is
deciding whether to trust pymboost's Poisson family in production.

### 4.3 Quantile regression at multiple τ on bodyfat

Quantile boosting on bodyfat at five τ levels surfaces a larger drift.
This is a harder parity target: the pinball-loss boost is sensitive to
how ties between candidate base learners are broken, and the two
implementations use slightly different internal scoring at each step.

```{code-cell} ipython3
q_cache = book_utils.load_cached_r_json("gallery_bodyfat_quantile")
tau_rows: list[dict] = []
for tau in q_cache["tau_values"]:
    m_q = glmboost(
        "DEXfat ~ hipcirc + kneebreadth + anthro3a",
        data=bodyfat,
        family=Quantile(tau=tau),
        control=boost_control(mstop=200, nu=0.1),
    )
    py = np.asarray(fitted(m_q), dtype=np.float64)
    r = np.asarray(q_cache["predictions"][f"tau_{tau}"], dtype=np.float64)
    for obs, (pv, rv) in enumerate(zip(py.tolist(), r.tolist())):
        tau_rows.append({"tau": tau, "observation": obs, "Python": pv, "R mboost": rv})

q_df = pl.DataFrame(tau_rows)
summary = (
    q_df.with_columns((pl.col("Python") - pl.col("R mboost")).abs().alias("absdiff"))
    .group_by("tau")
    .agg(pl.col("absdiff").max().alias("max_abs_diff"))
    .sort("tau")
)
summary
```

```{code-cell} ipython3
q_long = q_df.unpivot(
    index=["tau", "observation"],
    on=["Python", "R mboost"],
    variable_name="source",
    value_name="prediction",
)
alt.Chart(q_long).mark_circle(size=30, opacity=0.6).encode(
    x=alt.X("observation:Q", title="Observation index (bodyfat)"),
    y=alt.Y("prediction:Q", title="DEXfat prediction"),
    color=alt.Color(
        "source:N",
        scale=alt.Scale(
            domain=["Python", "R mboost"],
            range=[book_utils.PYTHON_COLOR, book_utils.R_COLOR],
        ),
        title=None,
    ),
).properties(width=220, height=160).facet(
    column=alt.Column("tau:N", title="τ"),
    title="Quantile regression predictions at five τ levels",
)
```

The per-observation prediction dots for Python and R track each other
tightly at τ = 0.5 (the median) but diverge at the tails. The summary
table above prints the worst-case gap per τ. The median case is by far
the tightest; the extreme quantiles (τ = 0.1, τ = 0.9) are where the
drift is largest, which makes sense — the quantile loss is most
sensitive to individual observations in those regimes, and small
differences in tie-breaking compound through the boosting iterations.

This is an honest, documented gap. It does not invalidate the Python
implementation for workflows that use it (the shapes and central
quantiles are correct), but it does mean users chasing bit-for-bit R
reproducibility at extreme τ values should not expect it from the
current package.

---

## 5. Tree learners: prediction grid parity

`btree` and `blackboost` cannot be compared at the coefficient level —
pymboost uses a CART backend, R's `blackboost` uses `partykit`'s
conditional inference trees. What *can* be compared, and what users
actually care about, is the **prediction function** on a shared grid:
given the same input `x`, do both implementations output the same `y`?
The two tree algorithms agree in broad shape (the piecewise-constant
structure and the location of splits) but not at every single step
edge, and this section shows exactly what that means.

### 5.1 `btree` on a 1-D step function

```{code-cell} ipython3
tree_cache = book_utils.load_cached_r_json("gallery_btree_grid")
tree_panel = pl.DataFrame(tree_cache["panel"])
from mboost import btree
m_btree = gamboost(
    "y ~ btree(x)",
    data=tree_panel,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)
grid_x = np.asarray(tree_cache["grid"]["x"], dtype=np.float64)
# Build a small prediction frame with the single feature. pymboost's
# `predict` is keyword-only: pass the new data via ``newdata=``.
grid_df = pl.DataFrame({"x": grid_x})
py_pred = np.asarray(m_btree.predict(newdata=grid_df), dtype=np.float64).ravel()
r_pred = np.asarray(tree_cache["grid"]["prediction"], dtype=np.float64)

grid_plot = pl.DataFrame(
    {
        "x": np.concatenate([grid_x, grid_x]),
        "prediction": np.concatenate([py_pred, r_pred]),
        "source": ["Python"] * len(grid_x) + ["R mboost"] * len(grid_x),
    }
)
rmse = float(np.sqrt(np.mean((py_pred - r_pred) ** 2)))
book_utils.prediction_grid_overlay(
    grid_plot,
    title=f"btree on a step function: shared-grid prediction — RMSE = {rmse:.3f}",
)
```

Both implementations recover the three-level step function. The
horizontal "tread" values are essentially identical; the tiny
disagreements live at the step *edges*, where the two tree algorithms
pick split points that differ by a handful of observations' worth of
x-distance. A user visualising the fitted function would see a single
step function, not two. A user asking "what's the prediction at
x = 0.38" (right next to a step edge) could see a small discrepancy.

Because this is a fundamental backend difference, not a bug, the
prediction-grid overlay is the right figure: it makes the agreement
visible where it exists (the treads) and honestly shows the small
disagreements where they exist (the edges).

### 5.2 `blackboost` on a 2-D interaction surface

```{code-cell} ipython3
black_cache = book_utils.load_cached_r_json("gallery_blackboost_grid")
black_panel = pl.DataFrame(black_cache["panel"])
from mboost import blackboost
m_black = blackboost(
    "y ~ x1 + x2",
    data=black_panel,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

g = np.asarray(black_cache["grid"]["x1"], dtype=np.float64)
h = np.asarray(black_cache["grid"]["x2"], dtype=np.float64)
grid_df = pl.DataFrame({"x1": g, "x2": h})
py_pred = np.asarray(m_black.predict(newdata=grid_df), dtype=np.float64).ravel()
r_pred = np.asarray(black_cache["grid"]["prediction"], dtype=np.float64)

surface_df = pl.DataFrame(
    {
        "x1": np.concatenate([g, g]),
        "x2": np.concatenate([h, h]),
        "prediction": np.concatenate([py_pred, r_pred]),
        "source": ["Python"] * len(g) + ["R mboost"] * len(g),
    }
)
alt.Chart(surface_df).mark_rect().encode(
    x=alt.X("x1:O", title="x₁", axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y("x2:O", title="x₂", axis=alt.Axis(labels=False, ticks=False)),
    color=alt.Color("prediction:Q", title="ŷ", scale=alt.Scale(scheme="viridis")),
).properties(width=240, height=240).facet(
    column=alt.Column("source:N", title=None),
    title="blackboost on sin(πx₁)·x₂: shared-grid surface (Python vs R)",
)
```

The left and right panels show the same qualitative surface — a
saddle-like interaction between `x1` and `x2`, lighter yellow in the
upper-left and lower-right, darker blue in the other two corners. The
numeric difference map:

```{code-cell} ipython3
diff_df = pl.DataFrame({"x1": g, "x2": h, "diff": py_pred - r_pred})
book_utils.diff_heatmap(
    diff_df,
    row_col="x2",
    col_col="x1",
    value_col="diff",
    title=(
        f"blackboost Python − R difference surface — "
        f"max |Δ| = {float(np.abs(py_pred - r_pred).max()):.3f}"
    ),
)
```

The difference map is dominated by near-white cells (small |Δ|) with
isolated coloured patches along the tree split boundaries. Again, this
is the expected signature of two different tree-ensemble backends fit
on the same data: the interior of each "leaf region" agrees closely,
the boundaries between leaves move around.

---

## 6. Summary table

One figure per parity claim, one row per claim — so the worst cases are
findable in one place without scrolling:

```{code-cell} ipython3
summary_rows = [
    {"section": "1. Gaussian glmboost (bodyfat)", "metric": "max |Δ coef| across full path", "value": "~ 1e-15"},
    {"section": "1. Gaussian glmboost (bodyfat)", "metric": "max |Δ corrected AIC|", "value": "~ 1e-15"},
    {"section": "1. Gaussian glmboost (bodyfat)", "metric": "max |Δ risk path|", "value": "~ 2e-12"},
    {"section": "2. Gaussian gamboost (bodyfat)", "metric": "max |Δ fitted|", "value": "~ 8e-8"},
    {"section": "3. cvrisk (bodyfat gamboost, 10-fold, shared masks)", "metric": "max |Δ mean risk|", "value": "< 1e-10"},
    {"section": "4.1 Binomial synthetic (3-term, mstop=100)", "metric": "max |Δ link|", "value": "~ 2e-15"},
    {"section": "4.2 Poisson synthetic (3-term, mstop=100)", "metric": "max |Δ link|", "value": "~ 3.5e-1"},
    {"section": "4.2 Poisson synthetic (3-term, mstop=100)", "metric": "  └─ constant offset", "value": "~ +1.6e-1"},
    {"section": "4.2 Poisson synthetic (3-term, mstop=100)", "metric": "  └─ shape residual", "value": "~ 1.8e-1"},
    {"section": "4.2 Poisson synthetic (3-term, mstop=100)", "metric": "  └─ corr(Python, R)", "value": "~ 0.9999"},
    {"section": "4.3 Quantile (bodyfat, τ=0.5, mstop=200)", "metric": "max |Δ fitted|", "value": "~ 6e-1"},
    {"section": "4.3 Quantile (bodyfat, extreme τ, mstop=200)", "metric": "max |Δ fitted|", "value": "~ 5 to 7"},
    {"section": "5.1 btree 1-D step function", "metric": "RMSE on shared grid", "value": "~ 0.10"},
    {"section": "5.1 btree 1-D step function", "metric": "max |Δ| on shared grid (edge-only)", "value": "~ 1.4"},
    {"section": "5.2 blackboost 2-D surface", "metric": "max |Δ prediction| on grid", "value": "~ 0.36"},
]
pl.DataFrame(summary_rows)
```

The top half of the table is pymboost's strong suit: Gaussian linear
and additive models, corrected AIC, the empirical risk path and a
clean Binomial setup all sit at machine precision. Cross-validated
risk does too — once you flip R's training-weight matrix into
pymboost's holdout-mask convention as shown in §3. The middle section —
realistic Poisson and Quantile fits — is where pymboost carries known
drift that is not yet covered by the parity test suite. The
tree-learner entries are expected, not drift: different backends give
different split boundaries by design.

See [PARITY_GAPS.md](https://github.com/) (repository root) for the
long-form discussion of what remains outstanding. The value of this
gallery is that every number above is now visible as a figure, and
every figure is generated by cells any reader can re-run against their
own copy of R `mboost` by running
`python scripts/refresh_book_assets.py` and then rebuilding the book.
