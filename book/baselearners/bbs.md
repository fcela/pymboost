---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `bbs` — penalized B-spline (P-spline) learners

`bbs` is the workhorse smooth learner in both `mboost` and `pymboost`: a
univariate B-spline basis with a quadratic difference penalty, calibrated so
that each boosting step injects a controlled amount of effective degrees of
freedom. It is the Eilers–Marx (1996) P-spline dropped straight into the
component-wise boosting framework, and it is why `gamboost(y ~ x + z)` is a
smooth additive model out of the box.

The companion R routine is
[`?bbs`](https://www.rdocumentation.org/packages/mboost/topics/baselearners);
the pymboost implementation lives in
[`mboost/baselearners/spline.py`](../../mboost/baselearners/spline.py) and the
basis construction in
[`mboost/baselearners/base.py::_build_pspline_basis_from_knots`](../../mboost/baselearners/base.py).

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
from scipy.interpolate import BSpline

from mboost import (
    Gaussian,
    bbs,
    boost_control,
    coef,
    fitted,
    gamboost,
    glmboost,
    partial_plot_data,
    risk,
    selected,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

```{code-block} python
bbs(
    name: str,
    *,
    df: int = 4,                # target effective degrees of freedom per step
    knots: int | None = 20,     # number of interior knots
    lambda_: float | None = None,  # direct smoothing penalty; overrides df
    degree: int = 3,            # B-spline degree (cubic by default)
    differences: int = 2,       # order of the difference penalty
    center: bool = False,       # map to centered (unpenalized) basis
    by: str | None = None,      # varying-coefficient modifier
)
```

R's `bbs` has the same semantics and essentially the same defaults:

```{code-block} r
bbs(x, by = NULL, knots = 20, boundary.knots = NULL, degree = 3,
    differences = 2, df = 4, lambda = NULL, center = FALSE, cyclic = FALSE)
```

Three differences are worth calling out:

1. **`lambda_` vs `lambda`.** The same rewrite trick as `bols` applies: inside
   a formula string, `lambda=` is rewritten to `lambda_=`, so
   `gamboost("y ~ bbs(x, lambda=0.5)", ...)` works unchanged from R.
2. **`boundary.knots` is not exposed.** pymboost always places boundary knots
   at the observed min/max of `x` with the standard `degree + 1` multiplicity.
   Users cannot override this yet. For Gaussian parity cases on the bodyfat
   slice this matches R exactly; for datasets with deliberately shifted
   boundaries the result may differ.
3. **`cyclic = TRUE` is not supported.** All `pymboost` splines are open.

Everything else — the default 20 interior knots, the default cubic B-splines,
the default second-order difference penalty, the default `df = 4` target — is
identical between the two packages.

## 2. The P-spline basis and penalty

For a covariate $x \in [a, b]$, `bbs` constructs an order-$K$ B-spline basis
with $n_{\text{knots}}$ equally spaced interior knots. With $n_{\text{knots}} =
20$ and $\text{degree} = 3$ this gives a design matrix $X \in \mathbb{R}^{n
\times 24}$ whose columns are the 24 non-zero B-spline functions on $[a, b]$
evaluated at the training covariate. Every basis function is compactly
supported on four consecutive knot intervals, so the design is sparse and
numerically well-behaved even for large $n$.

The penalty is the second-order squared difference of the spline coefficients:

$$
P(\beta) \;=\; \lambda \, \lVert D_2 \beta \rVert_2^2
\;=\; \lambda \, \sum_{k=3}^{p} (\beta_k - 2\beta_{k-1} + \beta_{k-2})^2,
$$

where $D_2$ is the $(p - 2) \times p$ second-difference matrix. $D_2 \beta =
0$ holds iff the spline is a straight line in $x$, so the null space of the
penalty is the space of linear functions. With $\lambda \to \infty$, `bbs`
collapses to a two-parameter linear learner; with $\lambda \to 0$, it
approaches the unpenalized B-spline projection of the pseudo-residual onto the
basis. The P-spline knob slides continuously between those two extremes.

The full smoother matrix

$$
S \;=\; X \bigl( X^\top W X + \lambda D_2^\top D_2 \bigr)^{-1} X^\top W
$$

is identical in form to the `bols` smoother — the only change is the penalty
kernel.

## 3. Eilers–Marx construction, visualized

A picture of the basis is worth several paragraphs:

```{code-cell} ipython3
:tags: [hide-input]
x_grid = np.linspace(0.0, 1.0, 400)
degree = 3
n_knots_interior = 20
spacing = 1.0 / (n_knots_interior + 1)
knot_vector = spacing * np.arange(-degree, n_knots_interior + degree + 2)
basis = BSpline.design_matrix(x_grid, knot_vector, degree, extrapolate=True).toarray()

selected_cols = [2, 6, 11, 16, 21]
plot_rows = []
for col in selected_cols:
    for i, xv in enumerate(x_grid):
        plot_rows.append({
            "basis": f"B_{col}",
            "x": float(xv),
            "value": float(basis[i, col]),
        })

alt.Chart(pd.DataFrame(plot_rows)).mark_line(strokeWidth=2.2).encode(
    x=alt.X("x:Q"),
    y=alt.Y("value:Q", title="B-spline basis function"),
    color=alt.Color("basis:N", title=None),
).properties(
    width=520,
    height=220,
    title="Five out of 24 default bbs basis functions on [0, 1]",
)
```

Each basis function is a non-negative bump supported on four adjacent knot
intervals. The full basis tiles $[a, b]$ so that every $x$ is covered by
exactly `degree + 1` functions summing to one; this is what gives the fit its
local-support property. Changing `knots` changes the spacing but not the
shape; changing `degree` changes the smoothness of each bump.

## 4. `df = 4`: why the boosting default is a very weak spline

`bbs` inherits the Demmler–Reinsch `df ↔ λ` calibration from `bols`: the
penalty is chosen so that $\text{tr}(S) = \text{df}$. The default `df = 4`
means every boosting step buys *at most four effective parameters* of spline
complexity — roughly, the complexity of a quadratic fit. The 24-column basis
is there to give the fit flexibility to bend, but only over many iterations.

The mathematical foundations chapter walks through this budget argument in
detail; the key takeaway for `bbs` is that a small per-step budget is the
whole point. Large `df` per step lets a single boosting iteration inject a
wildly flexible curve, which defeats the stage-wise bias–variance logic of
boosting.

```{code-cell} ipython3
rng = np.random.default_rng(0)
x = np.linspace(0.0, 1.0, 250)
y_truth = np.sin(2.0 * np.pi * x) + 0.3 * x
y = y_truth + rng.normal(scale=0.12, size=x.size)
toy = pd.DataFrame({"x": x, "y": y})

records = []
for df_step in [1, 4, 8, 20]:
    model = gamboost(
        f"y ~ bbs(x, df={df_step})",
        data=toy,
        family=Gaussian(),
        control=boost_control(mstop=200, nu=0.1),
    )
    pd_df = partial_plot_data(model, which=0, grid_size=120)
    pd_df["df per step"] = f"df = {df_step}"
    records.append(pd_df)

curves = pd.concat(records, ignore_index=True)
truth = pd.DataFrame({"x": x, "effect": y_truth - y_truth.mean(), "df per step": "truth"})

alt.Chart(
    pd.concat([truth, curves[["x", "effect", "df per step"]]], ignore_index=True)
).mark_line(strokeWidth=2.5).encode(
    x=alt.X("x:Q"),
    y=alt.Y("effect:Q", title="Fitted partial effect (centered)"),
    color=alt.Color(
        "df per step:N",
        scale=alt.Scale(
            domain=["truth", "df = 1", "df = 4", "df = 8", "df = 20"],
            range=[NEUTRAL_COLOR, "#72b7b2", PYTHON_COLOR, "#eeca3b", R_COLOR],
        ),
        title=None,
    ),
).properties(
    width=520,
    height=260,
    title="Effect of the per-step df budget on the final bbs fit (mstop=200)",
)
```

At `df = 1` the learner is so aggressively penalized it cannot bend; at
`df = 4` the fit recovers the sinusoid cleanly; at `df = 8` and `df = 20` the
same mstop produces a visibly wigglier curve with overfitting visible on the
tail. The default is a safe place to start; moving it should be justified.

## 5. Bodyfat worked example with R parity

The `bbs` equivalent of the `bols` bodyfat worked example is a purely additive
spline model — the same three predictors, each smoothed. This is close to what
Hofner et al. (2014) run in their tutorial section on `gamboost`.

```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv")

gam_bodyfat = gamboost(
    "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

pd.DataFrame(
    {
        "quantity": ["mstop used", "final risk", "offset", "#terms selected at least once"],
        "value": [
            gam_bodyfat.mstop,
            float(risk(gam_bodyfat)[-1]),
            float(gam_bodyfat.offset_),
            int(len(set(selected(gam_bodyfat)))),
        ],
    }
)
```

The partial-effect curves (centered, on a dense grid) are what users actually
look at. Plotted side-by-side with the cached R output they show the
smoothness is identical:

```{code-cell} ipython3
py_curves = partial_plot_data(gam_bodyfat, grid_size=120)
py_curves["source"] = "Python"

r_cache = book_utils.load_cached_r_json("baselearners_bbs")
r_rows = []
for feature, payload in r_cache["partial_curves"].items():
    r_rows.extend(
        {"term": f"bbs({feature})", "feature": feature, "x": float(xi), "effect": float(ei),
         "kind": "numeric", "source": "R mboost"}
        for xi, ei in zip(payload["x"], payload["effect"])
    )
r_curves = pd.DataFrame(r_rows)

combined = pd.concat([py_curves, r_curves], ignore_index=True)

alt.Chart(combined).mark_line(strokeWidth=2.5).encode(
    x=alt.X("x:Q", title=None),
    y=alt.Y("effect:Q", title="Partial effect on DEXfat"),
    color=alt.Color(
        "source:N",
        scale=alt.Scale(domain=["Python", "R mboost"], range=[PYTHON_COLOR, R_COLOR]),
        title=None,
    ),
).properties(width=220, height=180).facet(
    column=alt.Column("feature:N", title=None),
).properties(title="bbs bodyfat partial effects: Python vs R mboost")
```

The fitted-value parity completes the picture:

```{code-cell} ipython3
parity_df = pd.DataFrame(
    {
        "Python":   fitted(gam_bodyfat),
        "R mboost": np.asarray(r_cache["fitted"], dtype=np.float64),
    }
)
book_utils.parity_scatter(
    parity_df,
    title=f"bbs bodyfat parity — max |Δ| = {float((parity_df['Python'] - parity_df['R mboost']).abs().max()):.3e}",
)
```

Both cells read `book/_static/r_cache/baselearners_bbs.json`, produced by
`python scripts/refresh_book_assets.py --only baselearners_bbs`.

## 6. `center=True` and the unpenalized nullspace

`bbs(x, center=True)` applies the Fahrmeir–Kneib–Lang (2004) basis
decomposition: it splits the penalized P-spline basis into its penalty
nullspace (the straight line) and its orthogonal complement (everything the
penalty actually bends). The nullspace part is *removed* from the `bbs`
learner and must be handled by a separate `bols` term; the orthogonal
complement is rewritten as an unpenalized design.

Two reasons to do this matter in practice:

- **Clean separation of linear and nonlinear effects.** In the boosting
  competition, `bols(x) + bbs(x, center=True)` lets the linear and nonlinear
  parts of $f(x)$ win iterations independently. Without centering, `bbs(x)`
  can spend early iterations absorbing the linear component and obscure the
  interpretation of later iterations.
- **Numerical stability at small sample sizes.** The centered basis is
  orthogonal to the linear null space and has a condition number that does
  not blow up as the penalty shrinks.

```{code-cell} ipython3
rng = np.random.default_rng(2)
x = np.linspace(0.0, 2.0, 150)
y = 0.5 + 1.4 * x + 0.6 * np.sin(3.0 * x) + rng.normal(scale=0.15, size=x.size)
toy2 = pd.DataFrame({"x": x, "y": y})

plain = gamboost(
    "y ~ bbs(x)",
    data=toy2,
    family=Gaussian(),
    control=boost_control(mstop=400, nu=0.1),
)
centered = gamboost(
    "y ~ bols(x) + bbs(x, center=True)",
    data=toy2,
    family=Gaussian(),
    control=boost_control(mstop=400, nu=0.1),
)

comparison = pd.DataFrame(
    {
        "specification": ["bbs(x)", "bols(x) + bbs(x, center=True)"],
        "final risk":    [float(risk(plain)[-1]), float(risk(centered)[-1])],
        "terms ever selected": [
            sorted({lbl for lbl in plain.term_labels}),
            sorted({lbl for lbl in centered.term_labels}),
        ],
    }
)
comparison
```

Both fits arrive at essentially the same final risk, but the centered form
keeps the linear component separately attributable — a point Hofner et al.
(2014, §3.4) make at length, and one the parity wall chapter returns to.

## 7. `by=` and varying-coefficient smooths

`bbs(x, by=z)` is the smooth analogue of `bols(x, by=z)`: the P-spline basis
evaluated at `x` is multiplied row-wise by `z`, so the learner fits the
varying-coefficient term $z \cdot f(x)$ with $f$ a P-spline. The canonical
pattern, mirroring the mboost vignette, is a main-effect smooth plus an
interaction smooth:

```{code-block} python
gamboost(
    "y ~ bbs(x) + bbs(x, by=z)",
    data=...,
    family=Gaussian(),
)
```

If `z` is a binary indicator, the second term is a smooth
group-difference. If `z` is continuous, `bbs(x, by=z)` is a genuine
tensor-flavor interaction: the smooth shape in `x` scales linearly with `z`.
True tensor-product smooths (`bbs(x) %X% bbs(z)` in R) are *not* part of the
current pymboost surface; see the known-deviations section below.

## 8. Known deviations from R `bbs`

- **No `cyclic = TRUE`.** Cyclic splines (for angles, time of day, etc.) are
  not implemented. Attempting to pass the keyword raises a parser error.
- **No `boundary.knots` override.** Boundary knots are pinned to
  `(min(x), max(x))`. R's override is intended for out-of-sample prediction
  on an extended range and is not yet exposed here.
- **No tensor-product smooths.** `bbs` is strictly univariate. R's
  `bspatial(x, y)` and `bbs(x) %X% bbs(z)` have no direct pymboost
  equivalent; see [Status and Roadmap](../status-and-roadmap.md).
- **Rounding in the `df → λ` solver.** Identical to `bols`: differences of
  order $10^{-7}$ in the trace, which surface as coefficient differences of
  order $10^{-6}$ to $10^{-12}$ in the tested bodyfat slice.
- **Centered basis default.** `center` defaults to `False` in both R and
  Python. Users migrating from GAM literature that defaults to centered bases
  should toggle this explicitly.

## See also

- {doc}`../mathematical-foundations` — the derivation of the Demmler–Reinsch
  `df ↔ λ` calibration and the smoother-matrix representation this learner
  inherits.
- {doc}`bols` — linear effects; `bbs` with `center=True` stacks on top of a
  `bols` main-effect term.
- {doc}`bmono` — the same P-spline basis plus constrained shape restrictions.
- {doc}`../gamboost` — the full bodyfat smooth workflow with interactive
  partial-effect inspection.
