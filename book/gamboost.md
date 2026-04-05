---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Generalized Additive Models

This chapter covers `gamboost`, which extends the linear model to handle non-linear relationships using smooth P-splines and other non-parametric base-learners.

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure, data_dir, r_assign_dataframe
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
th_data_r = importr("TH.data")
ro.r('data(bodyfat, package = "TH.data")')
r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
ro.r("bodyfat <- bodyfat_py")

from mboost import Gaussian, gamboost, plot, confint
```

## Smoothing with `bbs()`

Instead of fixed linear terms, we can use penalized B-splines (`bbs`) to learn the shape of the relationship between the outcome and predictors.

```{code-cell} ipython3
gam1 = gamboost(
    "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
    data=bodyfat,
    family=Gaussian(),
)
print(gam1.summary())
```

### Visualizing Partial Effects

The standard `plot(gam1)` output shows the **partial contribution** of each term to the overall prediction.

```{code-cell} ipython3
plot(gam1)
```

Each panel represents $\hat f_j(x_j)$ in the additive predictor:
$$ \hat f(x) = \text{offset} + \sum_j \hat f_j(x_j) $$

## Confidence Intervals for Smooth Effects

We can estimate the uncertainty around these learned smooth shapes using `confint()`. Note that these are currently approximate pointwise intervals for Gaussian models.

```{code-cell} ipython3
gam1_effect_ci = confint(gam1, which=[0, 1, 2], level=0.95)
plot(gam1_effect_ci)
```

The ribbons represent the 95% confidence region for each partial effect. This helps identify where the model is confident in the non-linear trend and where it is more uncertain.

## Additive Prediction vs. Observations

We can also visualize the total prediction uncertainty against the observed data.

```{code-cell} ipython3
fitted_ci = confint(gam1, level=0.95).to_pandas()
fitted_ci["observed"] = bodyfat["DEXfat"].to_numpy()

(
    alt.Chart(fitted_ci)
    .mark_circle(size=45, opacity=0.45, color="#6c757d")
    .encode(
        x=alt.X("observation:Q", title="Observation index"),
        y=alt.Y("observed:Q", title="DEXfat / fitted value"),
    )
    +
    alt.Chart(fitted_ci)
    .mark_area(opacity=0.22, color="#4c78a8")
    .encode(
        x="observation:Q",
        y="lower:Q",
        y2="upper:Q",
    )
    +
    alt.Chart(fitted_ci)
    .mark_line(color="#1f77b4", strokeWidth=2.5)
    .encode(
        x="observation:Q",
        y="estimate:Q",
    )
).properties(width=560, height=240, title="Fitted additive predictor with 95% intervals")
```
