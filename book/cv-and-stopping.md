---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Cross-Validation and Stopping

One of the most important hyperparameters in boosting is the number of iterations ($m_{stop}$). Too few, and the model underfits; too many, and it overfits. This chapter covers how to tune $m_{stop}$ using `cvrisk`.

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

from mboost import Gaussian, boost_control, gamboost, glmboost, cvrisk, mstop, plot, risk, coef
```

## Early Stopping with `cvrisk`

`cvrisk` performs cross-validation by fitting the model on a subset of the data and evaluating the risk on a hold-out set.

```{code-cell} ipython3
gam2 = gamboost(
    "DEXfat ~ .",
    data=bodyfat,
    family=Gaussian(),
    dfbase=4,
    control=boost_control(mstop=100, nu=0.1),
)

cvm = cvrisk(
    "DEXfat ~ .",
    data=bodyfat,
    family=Gaussian(),
    control=boost_control(mstop=100, nu=0.1),
    folds=25,
)
print(f"Optimal mstop: {cvm.best_mstop}")
```

### Visualizing the Risk Path

The cross-validated risk path typically shows a U-shape. The point where the hold-out risk is minimized is our chosen `mstop`.

```{code-cell} ipython3
plot(cvm)
```

## Regularization Trade-offs

We can see the trade-off between the complexity of each base-learner (`dfbase`) and the number of iterations (`mstop`).

```{code-cell} ipython3
gam2_loose = gamboost(
    "DEXfat ~ .",
    data=bodyfat,
    family=Gaussian(),
    dfbase=8,
    control=boost_control(mstop=400, nu=0.1),
)

regularization_table = pl.DataFrame({
    "model": ["conservative", "looser"],
    "dfbase": [4, 8],
    "mstop": [100, 400],
    "final_risk": [float(risk(gam2)[-1]), float(risk(gam2_loose)[-1])],
})
regularization_table
```

## A Realistic Workflow

The typical model-building workflow involves several stages:

1.  **Linear Baseline**: Fit a `glmboost`.
2.  **Additive Expansion**: Identify non-linear terms and use `gamboost`.
3.  **Cross-Validation**: Tune `mstop`.
4.  **Final Model**: Set the model to the optimal `mstop`.

```{code-cell} ipython3
glm1 = glmboost("DEXfat ~ hipcirc + kneebreadth + anthro3a", data=bodyfat, family=Gaussian())
gam2_opt = mstop(gam2, cvm.best_mstop)

workflow_table = pl.DataFrame({
    "model": ["Linear (glm1)", "Linear (glm2)", "Additive (gam2_opt)"],
    "iterations": [glm1.mstop, gam2.mstop, gam2_opt.mstop],
    "risk": [float(risk(glm1)[-1]), float(risk(gam2)[-1]), float(risk(gam2_opt)[-1])],
    "selected": [len(coef(glm1)), len(coef(gam2)), len(coef(gam2_opt))],
})
workflow_table
```
