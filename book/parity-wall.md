---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Parity & Verification

This chapter provides evidence of the implementation fidelity of `pymboost` relative to R's `mboost`.

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure, data_dir, r_assign_dataframe, r_numeric, parity_scatter, risk_path_chart
import polars as pl
import pandas as pd
import numpy as np
import altair as alt

configure()
bodyfat = pl.read_csv(data_dir() / "bodyfat.csv")

# Initialize R reference
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
mboost_r = importr("mboost")
r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
ro.r("bodyfat <- bodyfat_py")

from mboost import Gaussian, bbs, gamboost, risk, fitted, coef
```

## Overlaying Partial Effects

The shapes of the learned partial effects should be identical between the two implementations.

```{code-cell} ipython3
# Python fit
gam1 = gamboost(
    "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
)

# R fit
ro.r('gam1_r <- gamboost(DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a), data = bodyfat)')

def learner_contribution(model, learner_index: int) -> np.ndarray:
    learner = model.prepared_learners[learner_index]
    label = model.term_labels[learner_index]
    beta = coef(model)[label]
    design = learner.transform(model.data)
    return np.asarray(np.einsum("ij,j->i", design, beta), dtype=np.float64)

gam1_terms = ["hipcirc", "kneebreadth", "anthro3a"]
rows = []
for i, feature in enumerate(gam1_terms):
    x_vals = bodyfat[feature].to_numpy()
    order = np.argsort(x_vals)
    rows.append(pd.DataFrame({"x": x_vals[order], "effect": learner_contribution(gam1, i)[order], "term": feature, "source": "Python"}))
    rows.append(pd.DataFrame({"x": x_vals[order], "effect": r_numeric(f"drop(fitted(gam1_r, which = {i+1}))")[order], "term": feature, "source": "R mboost"}))

(
    alt.Chart(pd.concat(rows)).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", title=None),
        y=alt.Y("effect:Q", title="Effect"),
        color=alt.Color("source:N", scale=alt.Scale(range=["#4c78a8", "#e45756"])),
    ).properties(width=210, height=180).facet(column="term:N")
)
```

## Risk Path Parity

The loss reduction over time (empirical risk path) is the signature of the boosting algorithm.

```{code-cell} ipython3
risk_compare = pd.DataFrame({
    "iteration": np.arange(len(risk(gam1))),
    "risk": risk(gam1),
    "source": "Python"
})
risk_compare_r = pd.DataFrame({
    "iteration": np.arange(len(risk(gam1))),
    "risk": r_numeric("risk(gam1_r)"),
    "source": "R mboost"
})

risk_path_chart(pd.concat([risk_compare, risk_compare_r]), title="Empirical risk path: Python vs R")
```

## Prediction Parity Scatterplot

We can also plot the fitted values directly against each other.

```{code-cell} ipython3
fitted_compare = pd.DataFrame({
    "Python": fitted(gam1),
    "R mboost": r_numeric("fitted(gam1_r)"),
})

parity_scatter(fitted_compare, title="Direct Prediction Parity: Python vs R")
```

The points cluster tightly around the dashed identity line. This indicates very
close agreement in fitted values, but not strict visual coincidence at the
pixel level.

## Quantitative Summary

For the `bodyfat` Gaussian model, the difference between the implementations is effectively zero.

```{code-cell} ipython3
pl.DataFrame({
    "metric": ["max_fitted_diff", "mean_fitted_diff", "max_risk_diff"],
    "value": [
        float(np.max(np.abs(gam1.fitted_ - r_numeric("fitted(gam1_r)")))),
        float(np.mean(np.abs(gam1.fitted_ - r_numeric("fitted(gam1_r)")))),
        float(np.max(np.abs(risk(gam1) - r_numeric("risk(gam1_r)")))),
    ]
})
```
