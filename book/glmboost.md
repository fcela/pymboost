---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Boosted Linear Models

This chapter focuses on `glmboost`, the interface for fitting boosted Generalized Linear Models (GLMs).

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure, data_dir, r_assign_dataframe, r_named_vector, wide_compare, coefficient_bar
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

from mboost import Gaussian, glmboost, coef, risk
```

## Linear Model with `glmboost`

We reproduce the linear formula from Garcia et al.:
$$ \texttt{DEXfat} \sim \texttt{hipcirc} + \texttt{kneebreadth} + \texttt{anthro3a} $$

```{code-cell} ipython3
glm1 = glmboost(
    "DEXfat ~ hipcirc + kneebreadth + anthro3a",
    data=bodyfat,
    family=Gaussian(),
)
print(glm1.summary())
```

### Understanding the Intercept Gap

If you compare the intercept of `pymboost` with R's `mboost`, you will see a difference. This is not an error; it's a difference in parameterization.

- `pymboost` fit: $\hat f(x) = \text{offset} + \sum_j \beta_j (x_j - \bar x_j)$
- R `mboost` fit: $\hat f(x) = \alpha + \sum_j \beta_j x_j$

They are mathematically identical, with $\alpha = \text{offset} - \sum_j \beta_j \bar x_j$.

### Visual Parity: Python vs. R Coefficients

The following chart demonstrates that the slope coefficients (the scientific effects) match R to high precision.

```{code-cell} ipython3
ro.r('glm1_r <- glmboost(DEXfat ~ hipcirc + kneebreadth + anthro3a, data = bodyfat)')
r_glm1_table = r_named_vector("coef(glm1_r, off2int = TRUE)", value_name="coefficient")

glm1_coef = coef(glm1)
terms = ["hipcirc", "kneebreadth", "anthro3a"]
glm1_table = pl.DataFrame({
    "term": terms,
    "coefficient_python": [float(glm1_coef[name][0]) for name in terms],
    "coefficient_r": [
        float(r_glm1_table.filter(pl.col("term") == name)["coefficient"][0])
        for name in terms
    ],
})

glm1_compare_long = glm1_table.unpivot(
    index="term", on=["coefficient_python", "coefficient_r"],
    variable_name="source", value_name="coefficient",
).with_columns(
    pl.col("source").replace({"coefficient_python": "Python", "coefficient_r": "R mboost"})
)

coefficient_bar(glm1_compare_long, title="Linear model: Python vs R slope coefficients")
```

## Variable Selection in High Dimensions

Boosting is particularly powerful for variable selection. When we include all predictors, the algorithm automatically picks the most informative ones.

```{code-cell} ipython3
glm2 = glmboost("DEXfat ~ .", data=bodyfat, family=Gaussian())
print(glm2.summary())
```

We can visualize the **Selection Frequency**, which shows how many times each predictor "won" a boosting iteration.

```{code-cell} ipython3
selected_terms = [glm2.term_labels[idx] for idx in glm2.path.selected]
counts = pd.Series(selected_terms).value_counts().reset_index(name="count").rename(columns={"index": "term"})

alt.Chart(counts).mark_bar().encode(
    x=alt.X("count:Q", title="Selection iterations"),
    y=alt.Y("term:N", sort="-x", title=None),
    color=alt.value("#4c78a8"),
).properties(width=520, height=280, title="Variable selection frequency")
```
