---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `blackboost` — bundled-tree boosting

`blackboost` is the *bundled-tree* face of mboost: instead of letting each
covariate get its own `btree` term that competes for selection, every
iteration fits a single joint tree on all covariates at once. It is what
makes mboost look, at a distance, like a generic tree-boosting engine — a
Friedman-style additive combination of shallow trees, with all the
interaction-capturing behavior that implies. Under the hood it is still the
same component-wise algorithm; the "component" is simply the single bundled
tree learner.

The companion R routine is
[`?blackboost`](https://www.rdocumentation.org/packages/mboost/topics/blackboost);
the pymboost implementation lives in
[`mboost/api/blackboost.py`](../../mboost/api/blackboost.py). It is a thin
wrapper that rewrites the formula into a single multi-feature `btree(...)`
term and delegates to `glmboost`.

## Setup

```{code-cell} ipython3
:tags: [hide-input]
import sys
from pathlib import Path

_book_dir = next(
    c for c in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (c / "book" / "book_utils.py").exists()
) / "book"
if str(_book_dir) not in sys.path:
    sys.path.insert(0, str(_book_dir))

import book_utils
ROOT = book_utils.configure()

import altair as alt
import numpy as np
import pandas as pd
import polars as pl

from mboost import (
    Gaussian,
    TreeControls,
    blackboost,
    boost_control,
    btree,
    coef,
    fitted,
    gamboost,
    partial_plot_data,
    risk,
    selected,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

```{code-block} python
blackboost(
    formula: str,
    *,
    data,
    family=Gaussian(),
    control=boost_control(...),
    tree_controls: TreeControls | None = None,  # maxdepth, minsplit, minbucket
)

TreeControls(
    minsplit: int = 10,
    minbucket: int = 4,
    maxdepth: int = 2,
)
```

The corresponding R signature is

```{code-block} r
blackboost(formula, data = list(),
           weights = NULL, na.action = na.pass,
           offset = NULL, family = Gaussian(),
           control = boost_control(),
           tree_controls = partykit::ctree_control(
               teststat = "quad", testtype = "Teststatistic",
               mincriterion = 0, maxdepth = 2))
```

Two differences are worth highlighting:

1. **Backend: CART, not conditional inference trees.** pymboost's
   `blackboost` routes every tree through scikit-learn's
   `DecisionTreeRegressor`, exactly as `btree` does. R's `blackboost` uses
   `partykit::ctree`. See the `btree` chapter for a deeper discussion of the
   split-rule implications. On well-separated signals the two produce
   additive fits of the same shape; on data with ties or heavy-tailed
   covariates they will disagree on individual split points.
2. **`tree_controls` is a small dataclass, not a full `ctree_control` call.**
   pymboost exposes only the three knobs that actually matter for
   bundled-tree boosting: `maxdepth`, `minsplit`, and `minbucket`. Everything
   else in R's `ctree_control` is `ctree`-specific and has no counterpart in
   the CART backend.

The default `maxdepth = 2` matches R — a deeper stump than `btree`'s default
of `1`, because a bundled tree must have at least enough depth to capture
joint structure across features.

## 2. Bundled vs additive: where the difference lives

The single sentence that explains `blackboost` is:

> `blackboost(y ~ x1 + x2 + x3)` is exactly
> `glmboost(y ~ btree(x1, x2, x3, maxdepth=2))`.

Mechanically, the formula rewriter in
[`blackboost.py`](../../mboost/api/blackboost.py) walks every term, collects
the plain feature names into one list, and builds a single multi-feature
`btree(...)` call with the `tree_controls` baked in. From that point on it
is an ordinary `glmboost` run — the component-wise selection loop has only
one component to select, so it picks it every iteration.

Contrast this with the additive tree pattern covered in the `btree`
chapter:

```{code-block} python
gamboost("y ~ btree(x1) + btree(x2)", ...)  # one tree per feature, competing
blackboost("y ~ x1 + x2", ...)               # one joint tree on both
```

The difference is not cosmetic. In the additive form, each boosting step
fits a univariate tree on one feature and the algorithm picks the winner.
Interaction effects between `x1` and `x2` are unreachable by construction,
because no tree ever sees both features at once. In the bundled form, every
tree sees every feature, so interactions are automatic — but the additive
decomposition by variable is lost.

A short numerical experiment on synthetic interaction data makes the gap
concrete:

```{code-cell} ipython3
rng = np.random.default_rng(5)
n = 400
x1 = rng.uniform(-1.0, 1.0, n)
x2 = rng.uniform(-1.0, 1.0, n)
# Data with a genuine x1:x2 interaction (the 0.3 * x1 * x2 term)
y = (
    np.sign(x1) * 0.8
    + np.where(x2 > 0.0, 0.4, -0.4)
    + 0.3 * x1 * x2
    + rng.normal(scale=0.2, size=n)
)
demo = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

bundled = blackboost(
    "y ~ x1 + x2",
    data=demo,
    family=Gaussian(),
    control=boost_control(mstop=200, nu=0.1),
)
additive = gamboost(
    "y ~ btree(x1) + btree(x2)",
    data=demo,
    family=Gaussian(),
    control=boost_control(mstop=200, nu=0.1),
)

pd.DataFrame(
    {
        "model": ["blackboost (bundled)", "gamboost(btree + btree)"],
        "final risk":  [float(risk(bundled)[-1]), float(risk(additive)[-1])],
        "train MSE":   [
            float(((fitted(bundled) - demo["y"].values) ** 2).mean()),
            float(((fitted(additive) - demo["y"].values) ** 2).mean()),
        ],
    }
)
```

On this specifically interaction-heavy DGP the bundled model cuts the risk
roughly in half. On data that is genuinely additive the two approaches
agree; on data with interactions the bundled form wins. This is the
textbook trade-off: interpretability by variable vs interaction capacity.

## 3. The joint surface

A single bundled `blackboost` fit produces one partial-effect surface per
term, and because the formula collapses to one term, that one surface *is*
the model's prediction (up to the offset):

```{code-cell} ipython3
surface = partial_plot_data(bundled, which=0, grid_size=32)
print(surface["kind"].iloc[0], "| rows:", len(surface))

alt.Chart(surface).mark_rect().encode(
    x=alt.X("x:Q", bin=alt.Bin(maxbins=32), title="x1"),
    y=alt.Y("y:Q", bin=alt.Bin(maxbins=32), title="x2"),
    color=alt.Color(
        "effect:Q",
        scale=alt.Scale(scheme="blueorange", domainMid=0),
        title="bundled effect",
    ),
).properties(
    width=360,
    height=300,
    title="blackboost joint surface after 200 iterations",
)
```

The surface is a checkerboard of rectangular cells — the signature of a
stacked-tree predictor — and the visible diagonal tilt is the `x1 * x2`
interaction the additive model cannot capture.

## 4. Bodyfat worked example

`blackboost` on the running bodyfat dataset is a one-liner that gives a
drop-in nonlinear regression. Because the rewriter bundles every feature
into a single `btree` term, the model has exactly one "selected" learner
and every iteration wins it:

```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv").to_pandas()

fit = blackboost(
    "DEXfat ~ hipcirc + kneebreadth + anthro3a",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
    tree_controls=TreeControls(maxdepth=2, minsplit=10, minbucket=4),
)

pd.DataFrame(
    {
        "quantity": [
            "mstop",
            "final risk",
            "#terms in formula (after rewrite)",
            "bundled term label",
            "distinct terms selected across iterations",
            "iterations that selected the bundled term",
        ],
        "value": [
            fit.mstop,
            float(risk(fit)[-1]),
            len(coef(fit)),
            list(coef(fit).keys())[0],
            len(set(selected(fit))),
            len(selected(fit)),
        ],
    }
)
```

Every one of the 100 iterations selects the single bundled `btree` term —
there is nothing else to pick. The one-term coefficient dict confirms the
rewrite: what started as three feature names became a single
`btree(hipcirc, kneebreadth, anthro3a, maxdepth=2, minsplit=10,
minbucket=4)` call under the hood.

## 5. `maxdepth` and the bundled-tree trade-off

Depth is the dominant tuning knob. With `maxdepth=1` every iteration fits a
one-split stump on all features — the algorithm can still choose which
feature to split on, but each step contributes a two-level constant. With
`maxdepth=2` or higher, each step can capture genuine pairwise or higher
interactions. Increasing depth raises the per-step variance, exactly as in
the `btree` chapter:

```{code-cell} ipython3
depth_rows = []
for depth in [1, 2, 3, 4]:
    f = blackboost(
        "DEXfat ~ hipcirc + kneebreadth + anthro3a",
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=100, nu=0.1),
        tree_controls=TreeControls(maxdepth=depth, minsplit=10, minbucket=4),
    )
    depth_rows.append(
        {
            "maxdepth": depth,
            "train risk": float(risk(f)[-1]),
            "train MSE":  float(((fitted(f) - bodyfat["DEXfat"].values) ** 2).mean()),
        }
    )

pd.DataFrame(depth_rows)
```

Train risk falls monotonically with depth on this small dataset — the
bundled trees have more and more room to memorize it. The *generalization*
trade-off is the reason both R and pymboost default to `maxdepth=2`: it is
enough depth to pick up dominant pairwise interactions, but shallow enough
that 100 boosting steps cannot overfit a 71-row dataset.

## 6. Prediction-grid parity against R `mboost`

Pointwise coefficient parity with R `blackboost` is not attainable —
pymboost's CART splitter and R's `partykit::ctree` splitter will
disagree on split points wherever the data has ambiguous optima (see
the `btree` chapter for the details of this backend gap). What *is*
comparable is the **prediction surface** on a shared 2-D grid: given
the same `(x1, x2)` input, do the two implementations output the same
`ŷ`? This is the quantity any downstream user sees, independent of
the internal split mechanics.

The R reference is pre-computed into ``gallery_blackboost_grid.json``
by ``scripts/refresh_book_assets.py``: a synthetic
``y = sin(π x1) · x2`` target fit with ``blackboost("y ~ x1 + x2",
mstop=100)`` in R, predicted on a 25×25 grid covering the
``[-1, 1]²`` domain.

```{code-cell} ipython3
black_cache = book_utils.load_cached_r_json("gallery_blackboost_grid")
black_panel = pl.DataFrame(black_cache["panel"])

m_black_gallery = blackboost(
    "y ~ x1 + x2",
    data=black_panel,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

g = np.asarray(black_cache["grid"]["x1"], dtype=np.float64)
h = np.asarray(black_cache["grid"]["x2"], dtype=np.float64)
py_pred = np.asarray(
    m_black_gallery.predict(newdata=pl.DataFrame({"x1": g, "x2": h})),
    dtype=np.float64,
).ravel()
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
    title="blackboost on sin(π x₁)·x₂: shared-grid surface (Python vs R)",
)
```

The two panels show the same qualitative structure — a saddle-like
interaction surface, bright yellow where both features pull the
response up and dark blue where they pull it down. The difference map
localises where the two implementations actually disagree:

```{code-cell} ipython3
diff_df = pl.DataFrame({"x1": g, "x2": h, "diff": py_pred - r_pred})
max_abs = float(np.abs(py_pred - r_pred).max())
book_utils.diff_heatmap(
    diff_df,
    row_col="x2",
    col_col="x1",
    value_col="diff",
    title=f"blackboost Python − R difference surface — max |Δ| = {max_abs:.3f}",
)
```

The difference map is dominated by near-white cells (interior of each
leaf region, where the two backends agree closely) with isolated
coloured cells along the split boundaries (where the CART and ctree
splitters pick slightly different break-points). This is the textbook
signature of comparing two tree-ensemble backends on the same data: the
*interior* of each "leaf region" agrees tightly; the *boundaries*
between leaves move by a few observations' worth of covariate
distance.

For deeper drift analysis, the {doc}`../parity-gallery` chapter shows
the same figure alongside every other family-level parity claim so
readers can judge `blackboost`'s shared-grid parity against the whole
package at a glance.

## 7. Known deviations from R `blackboost`

- **CART vs ctree backend.** See above and the `btree` chapter.
- **`tree_controls` is a small dataclass.** Only `maxdepth`, `minsplit`,
  and `minbucket` are exposed. R's `ctree_control` takes a dozen more
  knobs, most of which are `ctree`-specific.
- **Formula terms must be plain features or explicit `btree(...)` calls.**
  Any other wrapped learner (`bbs`, `bols`, `brandom`, …) inside a
  `blackboost` formula raises `NotImplementedError`. The whole point of the
  function is that every feature is bundled into one tree; mixing learner
  types defeats the rewrite.
- **At most one `by=` modifier.** If a user supplies an explicit
  `btree(..., by=z)` term, any second `by=` triggers `NotImplementedError`.
- **Hat-matrix diagnostics and corrected `AIC`** raise on any
  `blackboost` model, because the cumulative hat matrix is not defined for
  tree learners. Use `cvrisk` for stopping.

See [Status and Roadmap](../status-and-roadmap.md) for the live list.

## See also

- {doc}`btree` — the per-variable tree learner; `blackboost` is the
  bundled sibling where every feature lives in one tree term instead of
  competing across many.
- {doc}`../mathematical-foundations` — the component-wise selection step
  that `blackboost` degenerates to (one component, selected every
  iteration).
- {doc}`../cv-and-stopping` — the cross-validated stopping rule that
  replaces `AIC` for tree-based models.
- Bühlmann, P. and Hothorn, T. (2007). *Boosting Algorithms: Regularization,
  Prediction and Model Fitting.* Statistical Science, 22(4), 477–505.
