---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting Started

In this chapter, we set up the `pymboost` environment and demonstrate the first principles of boosting using a manual step-by-step example.

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure, data_dir, r_assign_dataframe
import polars as pl
import pandas as pd
import numpy as np

ROOT = configure()
bodyfat = pl.read_csv(data_dir() / "bodyfat.csv")

# Initialize R reference
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
mboost_r = importr("mboost")
th_data_r = importr("TH.data")
ro.r('data(bodyfat, package = "TH.data")')
r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
ro.r("bodyfat <- bodyfat_py")
```

## The Dataset

The R tutorial uses the `bodyfat` data from `TH.data`. The response is `DEXfat`, a body fat estimate from dual-energy X-ray absorptiometry.

```{code-cell} ipython3
bodyfat.head()
```

## 1.1 One boosting iteration by hand

The theory is compact, but the intuition is easier to see on a manual example. Suppose we start with a Gaussian model and two centered linear base-learners.

```{code-cell} ipython3
toy = pd.DataFrame(
    {
        "x1": np.array([-1.0, -0.5, 0.5, 1.0]),
        "x2": np.array([1.0, -1.0, 1.0, -1.0]),
        "y": np.array([-1.0, -0.4, 0.6, 1.1]),
    }
)

offset0 = toy["y"].mean()
u1 = toy["y"] - offset0
x1c = toy["x1"] - toy["x1"].mean()
x2c = toy["x2"] - toy["x2"].mean()

beta1 = float(np.dot(x1c, u1) / np.dot(x1c, x1c))
beta2 = float(np.dot(x2c, u1) / np.dot(x2c, x2c))
pred1 = beta1 * x1c
pred2 = beta2 * x2c
sse1 = float(np.sum((u1 - pred1) ** 2))
sse2 = float(np.sum((u1 - pred2) ** 2))
nu = 0.1
f1 = offset0 + nu * (pred1 if sse1 < sse2 else pred2)

pd.DataFrame(
    {
        "quantity": ["initial_offset", "beta_x1", "beta_x2", "sse_x1", "sse_x2", "winner"],
        "value": [float(offset0), beta1, beta2, sse1, sse2, "x1" if sse1 < sse2 else "x2"],
    }
)
```

The iteration follows these steps:
1.  Initialize the predictor at the offset.
2.  Compute pseudo-residuals (negative gradient).
3.  Fit each learner separately.
4.  Update using the best learner and a small step size $\nu$.

```{code-cell} ipython3
pl.DataFrame(
    {
        "x1": toy["x1"],
        "x2": toy["x2"],
        "y": toy["y"],
        "gradient": u1,
        "updated_f": f1,
    }
)
```
