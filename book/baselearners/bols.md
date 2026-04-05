---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `bols` — ordinary and ridge-penalized linear learners

`bols` is the base-learner for linear effects. It is the smallest non-trivial
learner in the library: a single covariate (optionally with an intercept column,
optionally ridge-penalized, optionally crossed with a `by=` modifier) fitted by
weighted least squares at every boosting step. Every other linear-in-design
learner — `bbs`, `bmono`, `brandom`, factor dummies — reuses the same
closed-form update that `bols` introduces, so the mathematics in this chapter is
the foundation for the rest of the modeling language.

The companion R routine is
[`?bols`](https://www.rdocumentation.org/packages/mboost/topics/baselearners);
the pymboost implementation lives in
[`mboost/baselearners/linear.py`](../../mboost/baselearners/linear.py) and the
per-iteration update in
[`mboost/core/engine.py`](../../mboost/core/engine.py).

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
    bols,
    boost_control,
    coef,
    fitted,
    glmboost,
    risk,
    selected,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

`bols` accepts a single column name plus up to five keyword options:

```{code-block} python
bols(
    name: str,
    *,
    df: float | None = None,    # target effective degrees of freedom; mapped to lambda_
    lambda_: float = 0.0,       # ridge penalty; mutually exclusive with df
    intercept: bool = True,     # include an intercept column in this learner's design
    center: bool = False,       # center columns to weighted mean zero
    by: str | None = None,      # multiplicative modifier (varying-coefficient)
)
```

The canonical R call is

```{code-block} r
bols(x, by = NULL, intercept = TRUE, df = NULL, lambda = 0,
     contrasts.arg = "contr.treatment")
```

Two differences matter in day-to-day use:

1. **Keyword spelling.** The Python side uses `lambda_` because `lambda` is a
   reserved word; inside a formula string, `lambda=...` is rewritten to
   `lambda_=...` at parse time. So `glmboost("y ~ bols(x, lambda=0.5)", ...)`
   works exactly like R.
2. **Contrasts.** `bols` on a categorical column produces an R-style
   treatment contrast design in both packages. pymboost does not yet expose
   `contrasts.arg`; the only available contrast style is treatment.

`df` and `lambda_` are mutually exclusive: specifying both raises
`ValueError`. This matches R, where specifying both is silently ignored in
favour of `df`.

## 2. The closed-form ridge update

At iteration $m$ the boosting engine hands `bols` the current pseudo-residual
$u^{[m]}$ and a weight vector $w$. The learner forms its design matrix
$X \in \mathbb{R}^{n \times p}$, the penalty matrix $P = \lambda I_p$ (with the
intercept column unpenalized when present), and solves the weighted ridge
problem

$$
\hat\beta^{[m]} \;=\; \arg\min_{\beta}
\Bigl\{ (u^{[m]} - X\beta)^\top W (u^{[m]} - X\beta) + \beta^\top P \beta \Bigr\}
\;=\; (X^\top W X + P)^{-1} X^\top W u^{[m]},
$$

where $W = \mathrm{diag}(w)$. The fitted contribution is $\hat h^{[m]} = X
\hat\beta^{[m]}$, and the learner's **smoother matrix** is

$$
S \;=\; X (X^\top W X + P)^{-1} X^\top W.
$$

This is the same $S$ introduced in the mathematical foundations chapter — for
`bols` it is literally a weighted hat matrix with ridge regularization. When
$\lambda = 0$ and there is no intercept column, $S$ reduces to the ordinary OLS
projection onto a single column.

For the special case of a one-column centered design (`intercept=False,
center=True`, $p = 1$), the update collapses to a scalar:

$$
\hat\beta^{[m]} \;=\;
\frac{\sum_i w_i \tilde x_i u_i^{[m]}}{\sum_i w_i \tilde x_i^2 + \lambda},
\qquad
\tilde x_i = x_i - \bar x.
$$

This is the fastest path inside the engine and is what runs when every term in
a `glmboost` formula is a single centered covariate.

## 3. `df` ↔ `lambda` calibration

Users rarely want to reason about $\lambda$ directly; what matters is how much
complexity a single `bols` step is allowed to inject. pymboost therefore
translates a requested `df` target into the $\lambda$ that makes the trace of
$S$ equal to that df. The translation is the standard Demmler–Reinsch
calibration:

$$
\operatorname{tr}(S) \;=\; \sum_{k=1}^{p} \frac{1}{1 + \lambda \, d_k},
$$

where $d_k$ are the eigenvalues of $R^{-\top} P R^{-1}$ with $R$ the Cholesky
root of $X^\top W X + \varepsilon P$. A univariate root find on $\lambda$
recovers the matching penalty. The implementation is `_df2lambda` in
`mboost/baselearners/base.py` and it is shared with every linear-in-design
learner.

```{code-cell} ipython3
rng = np.random.default_rng(0)
x = np.linspace(-1.5, 1.5, 200)
y = 0.8 * x + rng.normal(scale=0.25, size=x.size)
toy = pd.DataFrame({"x": x, "y": y})


def slope_of(model) -> float:
    """Return the last-iteration slope of the single bols term in ``model``."""
    (vec,) = coef(model).values()
    return float(vec[-1])


sweep_rows = []
for df_target in [1.0, 2.0, 3.0]:
    model = glmboost(
        f"y ~ bols(x, df={df_target})",
        data=toy,
        family=Gaussian(),
        control=boost_control(mstop=1, nu=1.0),
    )
    sweep_rows.append({"knob": f"df = {df_target:.1f}", "slope on x": slope_of(model)})
for lam in [0.0, 0.5, 5.0, 50.0]:
    model = glmboost(
        f"y ~ bols(x, lambda={lam})",
        data=toy,
        family=Gaussian(),
        control=boost_control(mstop=1, nu=1.0),
    )
    sweep_rows.append({"knob": f"lambda = {lam}", "slope on x": slope_of(model)})

pd.DataFrame(sweep_rows)
```

A one-step, `nu=1.0` fit recovers the unshrunken least-squares coefficient when
`df=2` (the learner has an intercept + slope, so `df=2` means "no penalty").
Raising `lambda` shrinks the same coefficient toward zero, and the shrinkage is
monotone in `lambda` — the fundamental reason `df` is a strictly more
interpretable knob than `lambda` for human users.

:::{note}
A single boosting step with `nu=1.0` is *not* a full linear fit: once
additional iterations run, the coefficients accumulate along the boosting path.
The tiny example above exists to show the `df↔lambda` mapping at step 1 in
isolation; real models almost always use `nu=0.1` and `mstop` in the dozens or
hundreds.
:::

## 4. Identifiability and centering

`bols` exposes two independent switches that interact with the global
intercept, and in practice users only ever hit two combinations:

- **`intercept=True, center=False` (the default).** The learner design is the
  2-column matrix $[\mathbf{1}, x]$. Each boosting step fits both a per-step
  intercept *and* a slope, and the penalty on the intercept column is
  zeroed out inside `BaseLearnerSpec.prepare` so the intercept is free.
- **`intercept=False, center=True` (single-column centered).** The learner
  design is the 1-column matrix $[x - \bar x]$. This is the form the
  `glmboost` formula parser reaches for when a user writes a bare term
  `y ~ x` — see `mboost/api/glmboost.py::_parse_term`. Every learner is one
  centered column, which is what keeps component-wise boosting from
  splintering the global mean across many terms.

Combining `intercept=True, center=True` would zero out the intercept column
after weighted centering and leave the Gram matrix rank-deficient, so pymboost
rejects it at fit time. The two valid combinations produce the same fitted
values on a single-covariate example but store the intercept in different
places:

```{code-cell} ipython3
rng = np.random.default_rng(1)
x = np.linspace(0.0, 2.0, 120)
y = 1.0 + 1.5 * x + rng.normal(scale=0.2, size=x.size)
demo = pd.DataFrame({"x": x, "y": y})

def inspect(term: str, label: str) -> dict:
    model = glmboost(
        f"y ~ {term}",
        data=demo,
        family=Gaussian(),
        control=boost_control(mstop=100, nu=0.1),
    )
    (vec,) = coef(model).values()
    return {
        "specification": label,
        "design columns": int(vec.shape[0]),
        "stored offset": float(model.offset_),
        "learner intercept": float(vec[0]) if vec.shape[0] == 2 else 0.0,
        "slope on x": float(vec[-1]),
        "final fitted mean": float(model.fitted_.mean()),
    }

pd.DataFrame(
    [
        inspect("bols(x)",                                     "bols(x) default"),
        inspect("bols(x, intercept=False, center=True)",       "single-column centered"),
    ]
)
```

Both specifications produce the same fitted values (the last column is
essentially the sample mean in both rows), but the default form stashes the
intercept inside the learner's own coefficient while the centered form stashes
it in `model.offset_`. Users who want to compare pymboost coefficients to
R's `coef(model, off2int = TRUE)` therefore need to know which form was used
and fold the offset back in when appropriate; the `glmboost` chapter walks
through the fold-back on the bodyfat example.

## 5. Design-matrix figure

The shapes of the two valid `bols` designs are easiest to see directly — a
2-column intercept-plus-slope design on the left, a 1-column centered design
on the right:

```{code-cell} ipython3
:tags: [hide-input]
x_demo = np.array([0.0, 0.25, 0.75, 1.25, 2.0])
rows = []
for label, X in [
    ("intercept=True, center=False",  np.column_stack([np.ones_like(x_demo), x_demo])),
    ("intercept=False, center=True",  (x_demo - x_demo.mean())[:, None]),
]:
    for row_idx in range(X.shape[0]):
        for col_idx in range(X.shape[1]):
            rows.append({
                "specification": label,
                "row": row_idx,
                "column": col_idx,
                "value": float(X[row_idx, col_idx]),
            })

design_df = pd.DataFrame(rows)

alt.Chart(design_df).mark_rect().encode(
    x=alt.X("column:O", title="design column"),
    y=alt.Y("row:O", title="observation", sort="descending"),
    color=alt.Color(
        "value:Q",
        scale=alt.Scale(scheme="blueorange", domainMid=0),
        title="entry",
    ),
).properties(width=140, height=160).facet(
    column=alt.Column("specification:N", title=None),
).properties(title="Two bols design matrices on the same five-point x")
```

The left panel has a constant column (all ones) alongside the raw covariate;
the right panel has a single column with mean subtracted. The single-column
form is the one the boosting engine streams down its fastest path, because it
reduces the ridge update to a scalar.

## 6. Bodyfat worked example with R parity

The running example from the Hofner et al. (2014) tutorial is a linear
bodyfat model with three predictors. Every term here is an explicit `bols`
term; the point is to see the coefficient path plot and the final parity
against R side-by-side.

```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv")

glm_bodyfat = glmboost(
    "DEXfat ~ bols(hipcirc) + bols(kneebreadth) + bols(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

fit_summary = pd.DataFrame(
    {
        "quantity": ["mstop used", "final risk", "offset (stored intercept)"],
        "value":    [glm_bodyfat.mstop, float(risk(glm_bodyfat)[-1]), float(glm_bodyfat.offset_)],
    }
)
fit_summary
```

Because every term uses `bols`'s default (`intercept=True, center=False`),
each learner stores a 2-vector `[intercept, slope]`, and R's `coef(mboost_fit)`
returns exactly the same per-learner structure. The parity check against the
cached R output is a long-form join on `(term, component)`:

```{code-cell} ipython3
r_cache = book_utils.load_cached_r_json("baselearners_bols")
py_coef = coef(glm_bodyfat)

py_rows = []
for term_label, vec in py_coef.items():
    py_rows.append({"term": term_label, "component": "(Intercept)", "Python": float(vec[0])})
    py_rows.append({"term": term_label, "component": term_label.removeprefix("bols(").removesuffix(")"), "Python": float(vec[1])})
py_long = pd.DataFrame(py_rows)

r_long = pd.DataFrame(r_cache["coefficients"]).rename(columns={"value": "R mboost"})
combined = py_long.merge(r_long, on=["term", "component"])
combined["abs_diff"] = (combined["Python"] - combined["R mboost"]).abs()
combined
```

The absolute differences should be at the level of numerical noise
($< 10^{-6}$). The matching parity scatter of fitted values tells the same
story in picture form:

```{code-cell} ipython3
parity_df = pd.DataFrame(
    {
        "Python":   fitted(glm_bodyfat),
        "R mboost": np.asarray(r_cache["fitted"], dtype=np.float64),
    }
)
book_utils.parity_scatter(
    parity_df,
    title=f"bols bodyfat parity — max |Δ| = {float((parity_df['Python'] - parity_df['R mboost']).abs().max()):.3e}",
)
```

Both cells read from `book/_static/r_cache/baselearners_bols.json`, which is
produced by `python scripts/refresh_book_assets.py --only baselearners_bols`.
The refresh script runs R once locally and writes the JSON; the book build
itself does not call rpy2. If the cache is stale (e.g. after bumping the R
`mboost` version), rerun the refresh script.

## 7. Parameter sweep: df and the boosting path

The effect of `df` on a single `bols` term is easiest to see by holding every
other choice fixed and sweeping. The plot below shows the empirical risk path
for three different per-step complexity budgets on the bodyfat model.

```{code-cell} ipython3
sweep_paths = []
for df_target in [1.0, 2.0, 4.0]:
    model = glmboost(
        (
            f"DEXfat ~ bols(hipcirc, df={df_target}) "
            f"+ bols(kneebreadth, df={df_target}) "
            f"+ bols(anthro3a, df={df_target})"
        ),
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=200, nu=0.1),
    )
    risks = risk(model)
    sweep_paths.append(
        pd.DataFrame(
            {
                "iteration": np.arange(risks.size),
                "risk": risks,
                "df per step": f"df = {df_target:.0f}",
            }
        )
    )
sweep_df = pd.concat(sweep_paths, ignore_index=True)

alt.Chart(sweep_df).mark_line(strokeWidth=2.5).encode(
    x=alt.X("iteration:Q", title="Boosting iteration"),
    y=alt.Y("risk:Q", title="Empirical risk"),
    color=alt.Color(
        "df per step:N",
        scale=alt.Scale(range=[PYTHON_COLOR, R_COLOR, "#72b7b2"]),
        title=None,
    ),
).properties(
    width=520,
    height=240,
    title="Empirical risk paths on bodyfat as df per bols step varies",
)
```

The smaller the per-step df, the slower the descent and the later the risk
plateaus. This is the *same* regularization story that the mathematical
foundations chapter tells for effective degrees of freedom: shrinkage per step
compounds into slower movement through function space, which is exactly what
early stopping trades off against bias.

## 8. `by=` and varying-coefficient effects

`bols(x, by=z)` produces the varying-coefficient learner $z \cdot f(x)$ with
$f$ linear in $x$. This is how the book reproduces R's `bols(x, by = z)`
pattern from the `mboost_tutorial.Rnw` vignette. Mechanically, the design
column is the elementwise product $x \cdot z$, which means `bols(x, by=z)`
competes as its own learner against `bols(x)` — it does not *replace* it. The
most common pattern is therefore a pair:

```{code-block} python
glmboost(
    "y ~ bols(x) + bols(x, by=z)",
    data=...,
    family=Gaussian(),
)
```

The first term picks up the main effect of `x`; the second picks up the
interaction with `z`. If `z` is a 0/1 indicator, the second term is a
group-specific slope adjustment. If `z` is a continuous modifier, the
interaction is a full varying-coefficient term, and the resulting model is
still fully additive in the interaction structure.

A larger worked example of `by=` is in the `gamboost` chapter; this chapter
keeps the coverage to the surface area, because the mathematics of `by=` is
identical for `bbs` and `bmono` and gets its full treatment in `bbs.md`.

## 9. Known deviations from R `bols`

- **No `contrasts.arg`.** R's `bols` accepts a `contrasts.arg` argument that
  lets the user swap between treatment, sum, Helmert, etc. pymboost currently
  only supports treatment contrasts. For the handful of parity cases where
  this matters, the workaround is to pre-encode the factor in a pandas or
  polars frame and pass the encoded columns.
- **`index=` argument is not exposed.** R's `bols` supports an `index`
  argument that is used internally by `bmrf` and by very high-cardinality
  grouping factors. pymboost does not need it because the equivalent paths
  are reached through `brandom`.
- **Numerical rounding in the df→λ solver.** The Brent-style univariate root
  find in `_df2lambda` converges to roughly `sqrt(eps)` on the $\lambda$
  scale. In practice this produces trace differences of order $10^{-7}$
  between R and Python for identical `df` targets. The R-backed parity tests
  in `tests/test_against_r/test_bols.py` verify that coefficient-level
  differences stay below $10^{-6}$ on the bodyfat slice.

See [Status and Roadmap](../status-and-roadmap.md) for the live list of open
gaps.

## See also

- {doc}`../mathematical-foundations` — where the smoother matrix $S$ and the
  `df↔λ` calibration are derived from first principles.
- {doc}`bbs` — P-splines inherit `bols`'s ridge-plus-centering mechanics and
  add the difference penalty on top.
- {doc}`brandom` — grouped linear effects; mathematically a very wide `bols`
  factor with a ridge penalty.
- {doc}`../glmboost` — the full bodyfat workflow with interactive model
  inspection and coefficient fold-back.
