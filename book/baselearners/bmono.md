---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `bmono` — shape-constrained P-spline learners

`bmono` takes the same P-spline basis as `bbs` and adds *shape constraints* on
the spline coefficients: the fitted curve must be monotone, convex, concave,
or strictly sign-constrained. The mathematical price is a single constrained
quadratic program per boosting step; the scientific payoff is a smooth that
respects prior knowledge.

The companion R routine is
[`?bmono`](https://www.rdocumentation.org/packages/mboost/topics/bmono); the
pymboost implementation lives in
[`mboost/baselearners/spline.py::bmono`](../../mboost/baselearners/spline.py)
and the QP solver in
[`mboost/baselearners/base.py::solve_constrained_quadratic`](../../mboost/baselearners/base.py).

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
    bmono,
    boost_control,
    coef,
    fitted,
    gamboost,
    glmboost,
    partial_plot_data,
    risk,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

```{code-block} python
bmono(
    name: str,
    *,
    constraint: str = "increasing",  # one of: increasing, decreasing,
                                     # convex, concave, positive, negative
    type: str = "quad.prog",         # solver: quad.prog or iterative
    by: str | None = None,
    knots: int = 20,
    degree: int = 3,
    differences: int = 2,
    df: int = 4,
    lambda_: float | None = None,
    niter: int = 10,                 # iterations of the iterative solver
    intercept: bool = True,
    boundary_constraints: bool = False,
)
```

The corresponding R call is

```{code-block} r
bmono(x, constraint = c("increasing", "decreasing", "convex", "concave",
                         "positive", "negative"),
       type = c("quad.prog", "iterative"),
       by = NULL, knots = 20, boundary.knots = NULL, degree = 3,
       differences = 2, df = 4, lambda = NULL,
       lambda2 = 1e6, niter = 10, intercept = TRUE,
       boundary.constraints = FALSE)
```

Minor differences:

- **`lambda2` is not exposed.** R's `bmono` takes a second smoothing parameter
  `lambda2` that controls the weight of the constraint violation in the
  penalty representation of the QP. pymboost uses a direct active-set solver
  and does not need the surrogate; results match R's `type = "quad.prog"` to
  numerical tolerance on the tested slice.
- **`boundary.constraints = TRUE`** is accepted as a keyword argument
  (`boundary_constraints=True`) and pairing it with `type = "iterative"`
  raises `NotImplementedError`, but the flag is **currently a no-op in
  the quad-prog path** — see §6 below for the details and the workaround.
- The `type = "iterative"` path is present but has no R-backed parity tests
  yet (see [Status and Roadmap](../status-and-roadmap.md)). Default and
  recommended is `type = "quad.prog"`.

## 2. From P-spline to shape-constrained P-spline

The starting point is the same design $X$ and difference penalty $P = \lambda
D_d^\top D_d$ that `bbs` builds. The unconstrained step solves

$$
\hat\beta^{[m]}_{\text{bbs}} \;=\; \arg\min_{\beta}
\Bigl\{ (u^{[m]} - X\beta)^\top W (u^{[m]} - X\beta) + \beta^\top P \beta \Bigr\}.
$$

A shape constraint is a linear inequality on $\beta$. For monotone
increasing, the constraint is that the first differences of the spline
coefficients are non-negative,

$$
\beta_2 - \beta_1 \;\ge\; 0, \quad \beta_3 - \beta_2 \;\ge\; 0, \quad \ldots,
\quad \beta_p - \beta_{p-1} \;\ge\; 0,
$$

which is compactly $D_1 \beta \ge 0$ with $D_1$ the first-difference matrix.
Each of the six supported constraints is a linear inequality of this form:

| Constraint   | Inequality on $\beta$                              |
| ------------ | --------------------------------------------------- |
| increasing   | $D_1 \beta \ge 0$ (non-decreasing coefficients)     |
| decreasing   | $D_1 \beta \le 0$                                   |
| convex       | $D_2 \beta \ge 0$ (non-negative second differences) |
| concave      | $D_2 \beta \le 0$                                   |
| positive     | $\beta \ge 0$                                       |
| negative     | $\beta \le 0$                                       |

The boosting step then becomes a constrained quadratic program:

$$
\hat\beta^{[m]}_{\text{bmono}} \;=\; \arg\min_{\beta : A\beta \ge 0}
\Bigl\{ (u^{[m]} - X\beta)^\top W (u^{[m]} - X\beta) + \beta^\top P \beta \Bigr\},
$$

with $A$ the constraint matrix of the chosen shape. The default solver is an
active-set QP and runs once per boosting step; the optional iterative path is
a majorization scheme from Bollaerts et al. (2006) and matches the
`type = "iterative"` branch in R.

Everything else — the basis construction, the `df → λ` calibration, the
smoother-matrix framing — is inherited from `bbs` unchanged.

## 3. Constraint visual

The right way to see what `bmono` does is to fit the same dataset with `bbs`
and with `bmono` and plot the two partial effects side by side. The toy below
generates a noisy non-monotone signal and asks `bmono` for the monotone-
increasing shape the analyst believes *should* hold:

```{code-cell} ipython3
rng = np.random.default_rng(7)
x = np.linspace(0.0, 2.0 * np.pi, 250)
y_truth = 1.2 * x + 0.8 * np.sin(3.0 * x)
y = y_truth + rng.normal(scale=0.35, size=x.size)
toy = pd.DataFrame({"x": x, "y": y})

free = gamboost(
    "y ~ bbs(x)",
    data=toy,
    family=Gaussian(),
    control=boost_control(mstop=300, nu=0.1),
)
constrained = gamboost(
    'y ~ bmono(x, constraint="increasing")',
    data=toy,
    family=Gaussian(),
    control=boost_control(mstop=300, nu=0.1),
)

free_curve = partial_plot_data(free, which=0, grid_size=150).assign(kind="free bbs(x)")
constr_curve = partial_plot_data(constrained, which=0, grid_size=150).assign(
    kind='bmono(x, constraint="increasing")'
)
truth = pd.DataFrame({"x": x, "effect": y_truth - y_truth.mean(), "kind": "truth"})

alt.Chart(
    pd.concat(
        [
            truth,
            free_curve[["x", "effect", "kind"]],
            constr_curve[["x", "effect", "kind"]],
        ],
        ignore_index=True,
    )
).mark_line(strokeWidth=2.5).encode(
    x=alt.X("x:Q"),
    y=alt.Y("effect:Q", title="Partial effect (centered)"),
    color=alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=[
                "truth",
                "free bbs(x)",
                'bmono(x, constraint="increasing")',
            ],
            range=[NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR],
        ),
        title=None,
    ),
).properties(
    width=560,
    height=260,
    title="Shape constraint removes the sinusoidal dips the data doesn't warrant",
)
```

The unconstrained `bbs` fit happily chases the sinusoidal wiggle — that is
what a good P-spline does when the data shows one. The `bmono` fit rejects
every dip that would force the curve to decrease and returns the monotone
envelope of the noisy signal. Both fits are valid; which one is *correct*
depends on whether the analyst believes the true effect is monotone.

## 4. All six shapes on one axis

A single figure is enough to see every available constraint on the same noisy
input. Each panel shows the training points, the truth, and the `bmono` fit
for one shape:

```{code-cell} ipython3
rng = np.random.default_rng(9)
x = np.linspace(-1.0, 1.0, 220)
y_base = 0.6 * x - 0.4 * x ** 2 + rng.normal(scale=0.2, size=x.size)

shape_rows = []
for constraint in ["increasing", "decreasing", "convex", "concave", "positive", "negative"]:
    signed = y_base.copy()
    if constraint == "decreasing":
        signed = -y_base
    if constraint == "convex":
        signed = 0.8 * x ** 2 + rng.normal(scale=0.15, size=x.size)
    if constraint == "concave":
        signed = -0.8 * x ** 2 + rng.normal(scale=0.15, size=x.size)
    if constraint == "positive":
        signed = 0.5 + 0.3 * x + rng.normal(scale=0.2, size=x.size)
    if constraint == "negative":
        signed = -0.5 - 0.3 * x + rng.normal(scale=0.2, size=x.size)

    fit = gamboost(
        f'y ~ bmono(x, constraint="{constraint}")',
        data=pd.DataFrame({"x": x, "y": signed}),
        family=Gaussian(),
        control=boost_control(mstop=200, nu=0.1),
    )
    curve = partial_plot_data(fit, which=0, grid_size=80).assign(
        constraint=constraint
    )
    # Add back the mean so the picture shows the raw prediction, not the centered effect
    curve["effect"] = curve["effect"] + float(fit.fitted_.mean())
    # Keep the training points alongside for context
    for xi, yi in zip(x, signed):
        shape_rows.append(
            {"constraint": constraint, "x": float(xi), "effect": float(yi), "role": "data"}
        )
    shape_rows.extend(
        {"constraint": constraint, "x": float(xi), "effect": float(ei), "role": "bmono"}
        for xi, ei in zip(curve["x"].values, curve["effect"].values)
    )

shape_df = pd.DataFrame(shape_rows)

points = (
    alt.Chart()
    .transform_filter("datum.role === 'data'")
    .mark_circle(size=20, opacity=0.4, color=NEUTRAL_COLOR)
    .encode(x="x:Q", y="effect:Q")
)
lines = (
    alt.Chart()
    .transform_filter("datum.role === 'bmono'")
    .mark_line(color=PYTHON_COLOR, strokeWidth=2.5)
    .encode(x="x:Q", y="effect:Q")
)
alt.layer(points, lines, data=shape_df).facet(
    column=alt.Column("constraint:N", title=None),
).properties(title="The six bmono constraint shapes on matched toy data")
```

The `positive`/`negative` constraints are the least interesting numerically
(they forbid the fitted curve from crossing zero) but are genuinely useful in
the applied cases where an effect has a known sign and the analyst wants
every boosting iteration to respect it.

## 5. Bodyfat worked example with R parity

The running bodyfat example is a natural fit for a monotone constraint. Body
fat percentage on DEXA should be a monotone increasing function of hip
circumference — every analyst believes this, and the unconstrained `bbs` fit
in the previous chapter happens to confirm it. Swapping `bbs` for `bmono`
bakes the belief in directly.

```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv")

gam_mono = gamboost(
    'DEXfat ~ bmono(hipcirc, constraint="increasing") + bbs(kneebreadth) + bbs(anthro3a)',
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
)

pd.DataFrame(
    {
        "quantity": ["mstop", "final risk", "offset"],
        "value":    [gam_mono.mstop, float(risk(gam_mono)[-1]), float(gam_mono.offset_)],
    }
)
```

The `hipcirc` partial effect is now monotone increasing by construction.
Comparing it to the cached R fit with the identical formula gives a side-by-
side parity check:

```{code-cell} ipython3
py_curve = partial_plot_data(gam_mono, which=0, grid_size=120)
py_curve["source"] = "Python"

r_cache = book_utils.load_cached_r_json("baselearners_bmono")
r_payload = r_cache["partial_curves"]["hipcirc"]
r_curve = pd.DataFrame(
    {
        "term": "bmono(hipcirc)",
        "feature": "hipcirc",
        "x": r_payload["x"],
        "effect": r_payload["effect"],
        "kind": "numeric",
        "source": "R mboost",
    }
)

alt.Chart(pd.concat([py_curve, r_curve], ignore_index=True)).mark_line(
    strokeWidth=2.5
).encode(
    x=alt.X("x:Q", title="hipcirc"),
    y=alt.Y("effect:Q", title="Partial effect on DEXfat"),
    color=alt.Color(
        "source:N",
        scale=alt.Scale(domain=["Python", "R mboost"], range=[PYTHON_COLOR, R_COLOR]),
        title=None,
    ),
).properties(
    width=520,
    height=260,
    title="bmono(hipcirc, 'increasing') bodyfat partial effect — Python vs R",
)
```

```{code-cell} ipython3
parity_df = pd.DataFrame(
    {
        "Python":   fitted(gam_mono),
        "R mboost": np.asarray(r_cache["fitted"], dtype=np.float64),
    }
)
book_utils.parity_scatter(
    parity_df,
    title=f"bmono bodyfat parity — max |Δ| = {float((parity_df['Python'] - parity_df['R mboost']).abs().max()):.3e}",
)
```

The cache is produced by
`python scripts/refresh_book_assets.py --only baselearners_bmono`.

## 6. `boundary_constraints=True` (currently a no-op)

R's `bmono` exposes a `boundary.constraints = TRUE` argument that extends
the constraint set to force the fitted curve to be flat at the boundary of
the observed range — the modelling narrative "the effect must be monotone
*and* must flatten out at the edges" that appears in dose-response and
saturating-effect stories.

pymboost accepts the same flag under the name `boundary_constraints=True`
and validates it (setting it together with `type="iterative"` raises
`NotImplementedError`), **but the flag is not yet threaded through to the
QP solver and has no effect on the fit**. Calling

```{code-block} python
gamboost(
    'y ~ bmono(x, constraint="increasing", boundary_constraints=True)',
    data=...,
    family=Gaussian(),
)
```

currently produces bit-for-bit identical output to the same call without
the flag. The R parity tests in `tests/test_against_r` happen to pass on
this slice because the test targets (e.g. `y = x^2` with `mstop=5`) do not
push the fit far enough into the constraint region to reveal the
difference. Treat `boundary_constraints=True` as a placeholder kwarg until
the QP layer picks it up; if you need genuine boundary-flat behaviour
today, either raise `df` / `lambda_` enough that the penalty alone flattens
the ends, or drop back to R `mboost`.

## 7. Known deviations from R `bmono`

- **`lambda2` is not exposed.** R `bmono` uses a smoothing-parameter
  formulation of the QP; pymboost uses a direct active-set solver, so the
  knob does not apply. Parity with R's `type = "quad.prog"` is
  numerically exact on the tested slice.
- **`type = "iterative"` has no R-backed parity tests.** The path exists
  and runs, but is not verified for parity across the full family matrix.
  `type = "quad.prog"` is the recommended default.
- **`boundary_constraints=True` is a no-op.** The kwarg is accepted and
  validated against `type="iterative"`, but is not yet threaded into the
  QP layer. See §6 above.
- **`constraint` is case-sensitive.** Unlike R, which accepts any unique
  prefix (`inc`, `con`), pymboost requires the exact strings in the table
  above. Supplying anything else raises `ValueError`.
- **Ordered factor support.** pymboost's `bmono` for ordered factors is
  covered by the Gaussian parity tests in `tests/test_against_r`. Other
  families beyond Gaussian are not yet tested; pessimistically treat them
  as provisional.

See [Status and Roadmap](../status-and-roadmap.md) for the current list of
remaining gaps.

## See also

- {doc}`bbs` — the unconstrained P-spline that `bmono` is built on top of.
- {doc}`bols` — the linear learner whose ridge-plus-centering mechanics
  carry through to `bmono` as well.
- {doc}`../gamboost` — the full additive-smooth workflow with bodyfat.
- [Status and Roadmap](../status-and-roadmap.md) — the current summary of
  remaining R parity gaps, including the `type="iterative"` and non-Gaussian
  coverage.
