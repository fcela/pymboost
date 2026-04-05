---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Families and Loss Functions

One of the key strengths of the `mboost` framework is the ability to change the loss function by swapping the `family`. This chapter demonstrates how to use built-in families and how to write your own.

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

from mboost import Gaussian, Quantile, boost_control, glmboost
from mboost.families.base import Family
```

## Built-in Families

`pymboost` supports common GLM families:
- `Gaussian()`: Squared loss (identity link).
- `Binomial()`: Log-loss (logit link).
- `Poisson()`: Poisson log-likelihood (log link).
- `Quantile(tau)`: Check loss for quantile regression.

### Quantile Regression with `Quantile(tau=0.5)`

Instead of predicting the mean, we can predict the median.

```{code-cell} ipython3
glm_median = glmboost(
    "DEXfat ~ hipcirc + kneebreadth + anthro3a",
    data=bodyfat,
    family=Quantile(tau=0.5),
    control=boost_control(mstop=500, nu=0.1),
)
print(glm_median.summary())
```

## Writing Your Own Family

You can create a custom family by subclassing `mboost.families.base.Family` and implementing three methods:

1.  `offset(y, weights)`: The initial guess.
2.  `negative_gradient(y, f)`: The pseudo-residuals.
3.  `risk(y, f, weights)`: The loss being minimized.

### Example: Asymmetric Squared Loss

Suppose you want to penalize underprediction more heavily than overprediction.

```{code-cell} ipython3
class AsymmetricSquared(Family):
    def __init__(self, tau: float = 0.75):
        self.tau = float(tau)

    def offset(self, y: np.ndarray, weights: np.ndarray) -> float:
        return float(np.average(y, weights=weights))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        diff = y - f
        weight = np.where(diff >= 0.0, self.tau, 1.0 - self.tau)
        return 2.0 * weight * diff

    def risk(self, y: np.ndarray, f: np.ndarray, weights: np.ndarray) -> float:
        diff = y - f
        weight = np.where(diff >= 0.0, self.tau, 1.0 - self.tau)
        return float(np.sum(weights * weight * diff * diff))

asym_fit = glmboost(
    "DEXfat ~ hipcirc + kneebreadth + anthro3a",
    data=bodyfat,
    family=AsymmetricSquared(tau=0.8),
    control=boost_control(mstop=200, nu=0.1),
)
print(asym_fit.summary())
```

This flexibility allows you to model any problem that can be framed as minimizing an empirical risk.
