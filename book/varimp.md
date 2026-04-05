---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Variable Importance

Variable importance in boosting can be measured in two ways: **selection frequency** (how many times a learner was chosen) and **risk reduction** (how much the loss was reduced by each selection).

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure, data_dir
import polars as pl
import pandas as pd
import numpy as np
import altair as alt

configure()
bodyfat = pl.read_csv(data_dir() / "bodyfat.csv")

from mboost import (
    Gaussian, gamboost, varimp
)
```

## Calculating Importance

The `varimp()` function calculates the total risk reduction attributed to each base-learner.

```{code-cell} ipython3
gam2 = gamboost(
    "DEXfat ~ .",
    data=bodyfat,
    family=Gaussian(),
)

vi = varimp(gam2)
vi_df = vi.to_pandas()
vi_df.head()
```

## Visualizing Importance

A bar chart provides a clear summary of which predictors are most influential in the model. By default, `varimp` shows the percentage reduction in risk.

```{code-cell} ipython3
alt.Chart(vi_df).mark_bar().encode(
    x=alt.X("reduction:Q", title="Relative Risk Reduction (%)"),
    y=alt.Y("variable:N", sort="-x", title=None),
    color=alt.value("#4c78a8"),
).properties(width=520, height=280, title="Variable Importance (Risk Reduction)")
```

## Selection Frequency

Alternatively, we can look at how often each variable was selected by the algorithm, regardless of the magnitude of risk reduction.

```{code-cell} ipython3
alt.Chart(vi_df).mark_bar().encode(
    x=alt.X("selfreq:Q", title="Selection Frequency", axis=alt.Axis(format='%')),
    y=alt.Y("variable:N", sort="-x", title=None),
    color=alt.value("#f58518"),
).properties(width=520, height=280, title="Variable Selection Frequency")
```
