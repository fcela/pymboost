---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `btree` — shallow piecewise-constant learners

`btree` is the tree-shaped weak learner: a single shallow regression tree
fitted to the pseudo-residual at every boosting step. Unlike the penalized
linear learners in this part of the book, `btree` is *not* linear in its
design — it has no closed-form smoother matrix, no `df → λ` calibration, and
no continuous penalty. Its regularization comes entirely from the depth cap.

In pymboost `btree` is CART-backed via scikit-learn's
`DecisionTreeRegressor`. This is *not* the same as R `mboost`'s
[`partykit::ctree`](https://rdrr.io/cran/partykit/man/ctree.html) backend,
so parity against R is by design limited to the shape of the response rather
than split-by-split equivalence. Everything else — the component-wise
selection, the additive predictor accumulation, the family interface — works
identically across the two packages.

The companion R routine is
[`?btree`](https://www.rdocumentation.org/packages/mboost/topics/baselearners);
the pymboost implementation lives in
[`mboost/baselearners/tree.py`](../../mboost/baselearners/tree.py).

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

from mboost import (
    Gaussian,
    boost_control,
    btree,
    coef,
    fitted,
    glmboost,
    partial_plot_data,
    risk,
    selected,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

```{code-block} python
btree(
    *names: str,              # one or more feature column names
    by: str | None = None,    # binary modifier for grouped trees
    maxdepth: int = 1,        # max tree depth
    minsplit: int = 10,       # min observations in a split
    minbucket: int = 4,       # min observations in a leaf
)
```

R's signature is a thin wrapper around `partykit::ctree_control`:

```{code-block} r
btree(..., by = NULL,
      tree_controls = partykit::ctree_control(
          teststat = "quad", testtype = "Teststatistic",
          mincriterion = 0, minsplit = 10, minbucket = 4,
          maxdepth = 1, stump = FALSE))
```

Two conceptual differences matter:

1. **Backend: CART, not conditional inference trees.** pymboost's `btree`
   uses a classical CART split on weighted sum-of-squares reduction (via
   scikit-learn's `DecisionTreeRegressor`). R's `btree` uses a
   quadratic-statistic split from `partykit::ctree`. The two splitters agree
   on simple signals but will pick different split points on data with
   heavy-tailed covariates or tied ranks.
2. **Depth, split, and bucket knobs are exposed directly.** pymboost
   surfaces the three most-used controls as plain keyword arguments; R
   routes them through `ctree_control`.

The `by=` argument is supported in pymboost, limited to binary (0/1 or
two-level factor) modifiers. Continuous `by=` for trees is not yet
implemented (see [Status and Roadmap](../status-and-roadmap.md)).

## 2. A piecewise-constant step in a single picture

`btree` with `maxdepth = 1` is a stump: each boosting step splits the
covariate at a single threshold and fits a two-level constant. The cumulative
fit is therefore a staircase — the archetype of a piecewise-constant
regression. Plotting the fitted vs true effect on a smooth signal makes the
approximation crystal clear:

```{code-cell} ipython3
rng = np.random.default_rng(3)
x = np.linspace(0.0, 1.0, 400)
y_truth = np.sin(2.0 * np.pi * x) + 0.4 * x
y = y_truth + rng.normal(scale=0.18, size=x.size)
toy = pd.DataFrame({"x": x, "y": y})

records = []
for label, md, mstop in [
    ("maxdepth=1, mstop=50",  1, 50),
    ("maxdepth=1, mstop=400", 1, 400),
    ("maxdepth=3, mstop=200", 3, 200),
]:
    fit = glmboost(
        f"y ~ btree(x, maxdepth={md})",
        data=toy,
        family=Gaussian(),
        control=boost_control(mstop=mstop, nu=0.1),
    )
    curve = partial_plot_data(fit, which=0, grid_size=200)
    curve["specification"] = label
    records.append(curve[["x", "effect", "specification"]])
truth = pd.DataFrame({"x": x, "effect": y_truth - y_truth.mean(), "specification": "truth"})
curves = pd.concat([truth, *records], ignore_index=True)

alt.Chart(curves).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
    x=alt.X("x:Q"),
    y=alt.Y("effect:Q", title="fitted contribution (centered)"),
    color=alt.Color(
        "specification:N",
        scale=alt.Scale(
            domain=["truth", "maxdepth=1, mstop=50", "maxdepth=1, mstop=400", "maxdepth=3, mstop=200"],
            range=[NEUTRAL_COLOR, "#eeca3b", PYTHON_COLOR, R_COLOR],
        ),
        title=None,
    ),
).properties(
    width=560,
    height=260,
    title="btree fits on a smooth sine: staircase quality improves with iterations and depth",
)
```

Three observations from this plot:

1. At `mstop = 50, maxdepth = 1` the staircase has too few treads: the fit
   is a crude piecewise constant with four or five visible steps.
2. At `mstop = 400, maxdepth = 1` the staircase is so dense it is
   indistinguishable from the smooth truth at the scale of the figure.
3. At `mstop = 200, maxdepth = 3` each boosting step contributes a deeper
   tree, so the fit reaches similar accuracy in fewer iterations — but each
   step has more variance, which is the reason the default is a stump.

The trade-off is the tree-boosting folklore from Friedman (2001): shallow
trees with many iterations beat deep trees with few iterations for stable
additive fits.

## 3. Component-wise selection with multiple trees

In the `mboost` / `pymboost` way of running trees, each covariate gets its
own `btree` term and the component-wise selection step chooses the winning
one at each iteration. This gives an additive decomposition by variable,
*not* a single multivariate tree. It is the core pattern that distinguishes
`mboost`'s tree use from `gbm` or `xgboost`.

```{code-cell} ipython3
rng = np.random.default_rng(13)
x1 = rng.uniform(-1.0, 1.0, 300)
x2 = rng.uniform(-1.0, 1.0, 300)
y = (
    np.where(x1 > 0.0, 1.0, -1.0)
    + np.where(x2 > 0.2, 0.5, -0.3)
    + rng.normal(scale=0.15, size=x1.size)
)
demo = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

model = glmboost(
    "y ~ btree(x1) + btree(x2)",
    data=demo,
    family=Gaussian(),
    control=boost_control(mstop=300, nu=0.1),
)

selection_counts = (
    pd.Series(selected(model))
    .value_counts()
    .rename_axis("term")
    .reset_index(name="iterations won")
    .sort_values("term")
)
selection_counts
```

The data was generated so that `x1` drives a larger jump than `x2`, so
`x1` wins essentially every one of the first ten iterations; once that jump
has been absorbed, `x2` takes over and wins the majority of later
iterations. This is exactly the behaviour R's `gamboost(y ~ btree(x1) +
btree(x2))` produces on the same data — the two packages disagree on
*which* split point a single tree finds, but they agree on the additive
decomposition and the selection pattern.

## 4. Multi-feature `btree(x1, x2)` — one tree per step

`btree(x1, x2)` with `maxdepth > 1` fits a single joint tree on both
covariates per boosting step — a full interaction learner. The behaviour
approaches `blackboost`'s (see `blackboost.md`) when every feature is
bundled into one `btree` term.

```{code-cell} ipython3
joint = glmboost(
    "y ~ btree(x1, x2, maxdepth=2)",
    data=demo,
    family=Gaussian(),
    control=boost_control(mstop=200, nu=0.1),
)
surface = partial_plot_data(joint, which=0, grid_size=32)
print(surface["kind"].iloc[0], "| rows:", len(surface))

alt.Chart(surface).mark_rect().encode(
    x=alt.X("x:Q", bin=alt.Bin(maxbins=32), title="x1"),
    y=alt.Y("y:Q", bin=alt.Bin(maxbins=32), title="x2"),
    color=alt.Color("effect:Q", scale=alt.Scale(scheme="blueorange", domainMid=0), title="joint effect"),
).properties(
    width=360,
    height=300,
    title="btree(x1, x2, maxdepth=2) surface after 200 iterations",
)
```

The surface is a checkerboard of constant rectangles — the defining feature
of tree-based learners — but stacked across 200 boosting iterations it
smoothly approximates the stepped truth in both dimensions simultaneously.

## 5. Prediction-grid parity against R `mboost`

At the coefficient level, pymboost's CART-backed `btree` and R `mboost`'s
`partykit::ctree` backend *cannot* agree: they are two different split-rule
algorithms. What can and should be compared is the **prediction function**
on a shared input grid — given the same x, do both implementations output
the same y? This is exactly what a downstream user sees, independent of
how the internal splits are picked.

The R side is pre-computed by ``scripts/refresh_book_assets.py`` into
``gallery_btree_grid.json``: a three-region step target (``y = -1`` for
``x < -0.3``, ``y = +0.5`` for ``-0.3 ≤ x < 0.3``, ``y = -0.5`` for
``x ≥ 0.3``), fit with ``gamboost("y ~ btree(x)", mstop=100)`` in R, and
predicted on a dense grid.

```{code-cell} ipython3
import polars as pl

tree_cache = book_utils.load_cached_r_json("gallery_btree_grid")
tree_panel = pl.DataFrame(tree_cache["panel"])

m_btree_gallery = glmboost(
    "y ~ btree(x)",
    data=tree_panel,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

grid_x = np.asarray(tree_cache["grid"]["x"], dtype=np.float64)
py_pred = np.asarray(
    m_btree_gallery.predict(newdata=pl.DataFrame({"x": grid_x})),
    dtype=np.float64,
).ravel()
r_pred = np.asarray(tree_cache["grid"]["prediction"], dtype=np.float64)

grid_plot = pl.DataFrame(
    {
        "x": np.concatenate([grid_x, grid_x]),
        "prediction": np.concatenate([py_pred, r_pred]),
        "source": ["Python"] * len(grid_x) + ["R mboost"] * len(grid_x),
    }
)
rmse = float(np.sqrt(np.mean((py_pred - r_pred) ** 2)))
max_abs = float(np.max(np.abs(py_pred - r_pred)))
book_utils.prediction_grid_overlay(
    grid_plot,
    title=(
        f"btree on a 3-level step target: shared-grid prediction — "
        f"RMSE = {rmse:.3f}, max |Δ| = {max_abs:.3f}"
    ),
)
```

Reading the figure: the two step functions overlap along every "tread" —
the flat regions of the fit — and diverge only at the step *edges*,
where the CART and ctree splitters pick break-points that differ by a
handful of observations' worth of x-distance. The RMSE printed in the
title is dominated by those edge disagreements; over the interior of
each flat region the two predictions agree to a part in 100.

This is the right figure for trees: it makes the large-scale agreement
visible where it exists (the treads) and the small-scale disagreement
honest where it exists (the edges). No coefficient-level parity claim
is possible across the two backends, but a user's mental model of
"this is what the function looks like" is preserved between
implementations.

## 6. Known deviations from R `btree`

- **CART vs ctree backend.** pymboost uses scikit-learn's
  `DecisionTreeRegressor` for its splits, not `partykit::ctree`. For any
  data where the split point is ambiguous (ties, near-uniform covariates),
  the two backends will make different choices. On well-separated signals
  the additive fits converge to the same shape within a few dozen
  iterations.
- **`by=` is binary-only.** Continuous `by=` modifiers for trees are not
  yet implemented.
- **No continuous regularization knob.** There is no `df ↔ λ` calibration
  for trees, because the smoother matrix does not exist for the CART
  split. Regularization is governed entirely by `maxdepth`, `minsplit`,
  `minbucket`, and `mstop`.
- **Hat-matrix diagnostics (`hatvalues`, corrected `AIC`)** raise
  `NotImplementedError` on any model that uses `btree` terms, because the
  cumulative hat matrix $B_m$ is not defined for non-linear-in-design
  learners. For tree-based stopping, use `cvrisk` instead — see the
  `cv-and-stopping` chapter.
- **Weighted splits.** Both pymboost and R `mboost` support sample
  weights in the split criterion, but the numerical handling of
  zero-weight observations is slightly different between the CART and
  ctree backends. Parity tests treat this as an intentional gap.

See [Status and Roadmap](../status-and-roadmap.md) for the current state.

## See also

- {doc}`blackboost` — the bundled-tree learner; one `btree`-like term per
  boosting step covering every feature jointly.
- {doc}`../mathematical-foundations` — the component-wise selection step
  that `btree` plugs into unchanged from its linear-learner siblings.
- {doc}`../cv-and-stopping` — the cross-validated stopping rule that
  replaces `AIC` for tree-based models.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient
  Boosting Machine.* Annals of Statistics, 29(5), 1189–1232.
