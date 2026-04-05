---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# `brandom` — grouped (random-intercept style) learners

`brandom` is the learner for *grouped effects*: one fitted parameter per level
of a factor, ridge-penalized into the origin. Under Gaussian loss the learner
is mathematically identical to a random-intercept model with the ridge penalty
playing the role of the variance-ratio shrinkage. Inside boosting it is the
workhorse for cross-sectional data with many small groups, for longitudinal
panels where the group intercepts are nuisance, and for every
disambiguation-by-ID case in applied work.

The companion R routine is
[`?brandom`](https://www.rdocumentation.org/packages/mboost/topics/baselearners);
the pymboost implementation lives in
[`mboost/baselearners/linear.py::brandom`](../../mboost/baselearners/linear.py)
and reuses `BaseLearnerSpec.prepare`'s `kind="random"` branch in
[`base.py`](../../mboost/baselearners/base.py).

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
    bols,
    boost_control,
    brandom,
    coef,
    fitted,
    glmboost,
    risk,
    selected,
)
from book_utils import NEUTRAL_COLOR, PYTHON_COLOR, R_COLOR
```

## 1. Signature and R parallel

```{code-block} python
brandom(
    name: str,
    *,
    df: float = 4,             # target effective degrees of freedom per step
    lambda_: float | None = None,  # ridge penalty; overrides df
    by: str | None = None,     # varying-random-coefficient modifier
)
```

The R signature is nearly identical:

```{code-block} r
brandom(x, by = NULL, index = NULL, df = 4, lambda = NULL,
        contrasts.arg = "contr.dummy")
```

Three behavioural points to keep in mind:

1. **`df = 4` is the default in both packages.** On a factor with many levels
   this is a strong shrinkage: the learner can pick up only 4 effective
   parameters per boosting step, even when the underlying factor has dozens
   of levels.
2. **`contrasts.arg` is fixed to the full dummy design.** Every level gets
   its own column; there is no reference level. This is what differentiates
   `brandom` from `bols` on a factor (which uses treatment contrasts).
3. **`index` is not exposed.** pymboost does not need an index argument
   because the engine handles level resolution directly from the input
   column.

## 2. From random intercept to ridge learner

At iteration $m$, `brandom` builds the $n \times K$ indicator matrix $Z$
whose rows are one-hot encodings of the factor level, so $Z_{ik} = 1$ iff
observation $i$ is in group $k$. The penalty matrix is the identity:
$P = \lambda I_K$. The boosting step solves

$$
\hat u^{[m]} \;=\; \arg\min_{u}
\Bigl\{ (r^{[m]} - Z u)^\top W (r^{[m]} - Z u) + \lambda \lVert u \rVert_2^2 \Bigr\}
\;=\; (Z^\top W Z + \lambda I)^{-1} Z^\top W r^{[m]},
$$

where $r^{[m]}$ is the current pseudo-residual and $u \in \mathbb{R}^K$ is the
per-group coefficient vector. For balanced groups with $n_k$ observations
each and uniform weights, this collapses to the elementwise formula

$$
\hat u_k^{[m]} \;=\; \frac{n_k}{n_k + \lambda} \, \bar r_k^{[m]},
$$

i.e. the *shrunken group mean* of the pseudo-residual. The scalar
$n_k / (n_k + \lambda)$ is exactly the posterior shrinkage factor of a
Gaussian random-intercept model with within-group variance $\sigma^2$ and
between-group variance $\sigma^2_g = \sigma^2 / \lambda$. That equivalence is
standard (Ruppert, Wand and Carroll 2003, §4.9) and is what justifies calling
`brandom` "the random-intercept learner".

The smoother matrix $S = Z (Z^\top W Z + \lambda I)^{-1} Z^\top W$ has trace
$\operatorname{tr}(S) = \sum_k n_k / (n_k + \lambda)$, which is what the
`df → λ` calibration solves for. At $\lambda = 0$ the learner reproduces a
plain per-group mean (trace $K$); at $\lambda \to \infty$ it collapses every
group to zero (trace $0$).

## 3. Shrinkage curve

A picture of the shrinkage factor as a function of `df` and the number of
levels $K$ makes the regularization intuitive:

```{code-cell} ipython3
:tags: [hide-input]
K_values = [5, 15, 50]
df_grid = np.linspace(0.1, 6.0, 80)

rows = []
for K in K_values:
    # balanced groups with n_k = 20 rows each
    n_per = 20
    n_total = K * n_per
    # For uniform weights and balanced groups, tr(S) = K * n_per / (n_per + lambda)
    # Solve for lambda given df target:
    for df_target in df_grid:
        if df_target >= K:
            lam = 0.0
        else:
            lam = n_per * (K - df_target) / df_target
        shrinkage = n_per / (n_per + lam)
        rows.append({
            "K (levels)": K,
            "df target": float(df_target),
            "per-group shrinkage": float(shrinkage),
        })

alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2.5).encode(
    x=alt.X("df target:Q", title="df per boosting step"),
    y=alt.Y(
        "per-group shrinkage:Q",
        title="shrinkage factor n_k / (n_k + λ)",
        scale=alt.Scale(domain=[0.0, 1.0]),
    ),
    color=alt.Color("K (levels):N", title="# factor levels"),
).properties(
    width=520,
    height=240,
    title="brandom shrinkage: balanced groups of 20, df ↔ per-level shrinkage",
)
```

At `df = 4` (the default) the learner applies very heavy shrinkage on a
50-level factor and essentially none on a 5-level factor. This is the
quantitative version of the R documentation's advice that `df` should usually
be raised on factors with very few levels.

## 4. Fixed plus random worked example

The canonical pattern is a fixed-effect slope (`bols`) crossed with a
random-intercept learner (`brandom`). The toy below generates data from
exactly that DGP and recovers both components from a single Python call:

```{code-cell} ipython3
rng = np.random.default_rng(11)
n_groups = 15
n_per = 30
# Zero-padded labels so the factor level order matches the integer order —
# brandom stores one coefficient per level in sorted-label order, and we
# want to compare against true_group_effects without re-sorting.
group_labels = np.repeat([f"{i:02d}" for i in range(n_groups)], n_per)
true_group_effects = rng.normal(scale=0.6, size=n_groups)
z = rng.normal(size=group_labels.size)
y = (
    0.4 * z
    + true_group_effects[np.repeat(np.arange(n_groups), n_per)]
    + rng.normal(scale=0.25, size=group_labels.size)
)

panel = pd.DataFrame({"group": group_labels, "z": z, "y": y})

fit = glmboost(
    "y ~ bols(z) + brandom(group)",
    data=panel,
    family=Gaussian(),
    control=boost_control(mstop=300, nu=0.1),
)

(vec_z,) = (v for k, v in coef(fit).items() if k.startswith("bols"))
(vec_group,) = (v for k, v in coef(fit).items() if k.startswith("brandom"))

pd.DataFrame(
    {
        "quantity": [
            "mstop",
            "final risk",
            "slope on z (true = 0.40)",
            "#group effects",
            "sd(true group effects)",
            "sd(fitted group effects)",
            "corr(fit, true)",
        ],
        "value": [
            fit.mstop,
            float(risk(fit)[-1]),
            float(vec_z[-1]),
            int(vec_group.shape[0]),
            float(true_group_effects.std()),
            float(vec_group.std()),
            float(np.corrcoef(vec_group, true_group_effects)[0, 1]),
        ],
    }
)
```

The fitted slope recovers the true 0.4 closely, and the fitted group effects
are shrunk toward zero relative to the true per-group intercepts — exactly
the behaviour a random-intercept model produces. Plotting fitted vs true
effects makes the shrinkage visible:

```{code-cell} ipython3
effect_df = pd.DataFrame(
    {
        "group": [f"g{i:02d}" for i in range(n_groups)],
        "true effect":    true_group_effects,
        "brandom effect": vec_group,
    }
)
long_effects = effect_df.melt(id_vars="group", var_name="source", value_name="effect")

alt.Chart(long_effects).mark_point(filled=True, size=80).encode(
    x=alt.X("group:N", title="group"),
    y=alt.Y("effect:Q", title="group effect"),
    color=alt.Color(
        "source:N",
        scale=alt.Scale(domain=["true effect", "brandom effect"], range=[NEUTRAL_COLOR, PYTHON_COLOR]),
        title=None,
    ),
).properties(
    width=560,
    height=240,
    title="Shrunken brandom group effects vs true effects on a 15-group panel",
)
```

The pattern is the classical one: the fitted effects follow the true ones in
direction, with magnitude shrunk toward zero by a factor that depends on the
penalty. On this 15-group panel the fitted standard deviation is a few
percent below the true standard deviation — a mild shrinkage, because `df = 4`
of a 15-level factor is a modest per-step budget.

## 5. R parity

Because the fixed-plus-random pattern is the same on both sides, the R parity
comparison is a direct fitted-value scatter and a per-group coefficient merge.
The cache target seeds R with identical data generated by the script:

```{code-cell} ipython3
r_cache = book_utils.load_cached_r_json("baselearners_brandom")

# Python side uses the cached data to guarantee identical inputs
panel = pd.DataFrame(r_cache["panel"])
fit = glmboost(
    "y ~ bols(z) + brandom(group)",
    data=panel,
    family=Gaussian(),
    control=boost_control(mstop=300, nu=0.1),
)

py_group = next(v for k, v in coef(fit).items() if k.startswith("brandom"))
r_group = np.asarray(r_cache["group_effects"], dtype=np.float64)

merged = pd.DataFrame(
    {
        "group": [f"g{i:02d}" for i in range(py_group.shape[0])],
        "Python":   py_group,
        "R mboost": r_group,
    }
)
merged["abs_diff"] = (merged["Python"] - merged["R mboost"]).abs()
merged
```

```{code-cell} ipython3
parity_df = pd.DataFrame(
    {
        "Python":   fitted(fit),
        "R mboost": np.asarray(r_cache["fitted"], dtype=np.float64),
    }
)
book_utils.parity_scatter(
    parity_df,
    title=f"brandom parity — max |Δ| = {float((parity_df['Python'] - parity_df['R mboost']).abs().max()):.3e}",
)
```

The cache is produced by
`python scripts/refresh_book_assets.py --only baselearners_brandom`.

## 6. `by=` and crossed random effects

`brandom(group, by=z)` produces a *random slopes* learner: instead of one
intercept per group, the learner fits one slope per group, multiplied by
`z`. Mechanically the indicator design $Z$ is scaled row-wise by `z`. This is
the same pattern as `bols(x, by=z)` but now the "fixed part" lives inside the
factor. A common pairing in longitudinal panels is the full quartet

```{code-block} python
glmboost(
    "y ~ bols(z) + brandom(group) + brandom(group, by=z)",
    data=...,
    family=Gaussian(),
)
```

which fits a fixed slope on `z`, a random intercept per group, and a random
slope per group — the canonical mixed-effects formulation in a single
component-wise boosting call.

## 7. Known deviations from R `brandom`

- **Contrast fixed to full dummy.** pymboost always builds a full indicator
  design on a factor. R's `contrasts.arg` lets you substitute `contr.sum` or
  other contrasts, which changes the constant offset but leaves the fitted
  values equivariant. Parity on coefficient-level comparisons therefore
  requires matching contrast choices on both sides.
- **`index` is not exposed.** See the bols chapter: the equivalent R idiom
  is reached by passing the factor directly.
- **Character versus factor inputs.** pymboost's `brandom` accepts
  character columns or pandas `Categorical` columns interchangeably; R
  requires an explicit factor. Mixed inputs are coerced to the union of
  levels in alphabetical order, which may differ from R if the R factor was
  constructed with a non-default level ordering. If coefficient-level parity
  matters, pass explicit ordered levels on both sides.

See [Status and Roadmap](../status-and-roadmap.md) for the live list of
remaining gaps.

## See also

- {doc}`../mathematical-foundations` — smoother matrices and the `df ↔ λ`
  calibration that drive `brandom`'s shrinkage.
- {doc}`bols` — linear learners; `brandom` is mathematically a *wide* `bols`
  on a factor with a ridge penalty and no treatment contrast.
- {doc}`../glmboost` — the full workflow, of which this chapter is the
  grouped-effects special case.
- Ruppert, D., Wand, M. P., and Carroll, R. J. (2003). *Semiparametric
  Regression.* Cambridge University Press. (§4.9 on the ridge/random-effect
  equivalence.)
