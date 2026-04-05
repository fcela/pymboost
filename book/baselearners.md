---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Base-Learner Handbook

Base-learners are the "weak learners" of model-based boosting. This chapter provides a deep dive into the semantics of each available learner.

```{code-cell} ipython3
:tags: [hide-cell]
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

import numpy as np
import altair as alt

from mboost import (
    Gaussian, bbs, bmono, bols, brandom, btree, gamboost, plot
)
```

## 1. `bols()`: Linear Effects

`bols()` is the building block for linear models. It can handle numeric covariates and categorical factors.

```{code-cell} ipython3
rng = np.random.default_rng(1907)
x = rng.normal(size=100)
y = 0.5 * x + rng.normal(scale=0.2, size=100)

mod_bols = gamboost("y ~ bols(x)", data={"x": x, "y": y}, family=Gaussian())
plot(mod_bols)
```

## 2. `bbs()`: Penalized Splines

`bbs()` fits smooth non-linear effects using P-splines. You can control the complexity via degrees of freedom (`df`).

```{code-cell} ipython3
x_smooth = np.linspace(-2, 2, 100)
y_smooth = np.sin(x_smooth) + rng.normal(scale=0.1, size=100)

mod_bbs = gamboost("y ~ bbs(x, df=4)", data={"x": x_smooth, "y": y_smooth}, family=Gaussian())
plot(mod_bbs)
```

## 3. `bmono()`: Shape-Constrained Splines

`bmono()` allows you to impose monotonicity or convexity constraints on the learned smooth relationship.

```{code-cell} ipython3
x_mono = np.linspace(0, 1, 100)
y_mono = np.exp(x_mono) + rng.normal(scale=0.1, size=100)

mod_mono = gamboost('y ~ bmono(x, constraint="increasing")', data={"x": x_mono, "y": y_mono}, family=Gaussian())
plot(mod_mono)
```

## 4. `btree()`: Piecewise-Constant Trees

`btree()` implements boosting with shallow trees (typically stumps). This is useful for capturing interactions and piecewise patterns.

```{code-cell} ipython3
x_tree = rng.uniform(-1, 1, 100)
y_tree = np.where(x_tree < 0, -1, 1) + rng.normal(scale=0.1, size=100)

mod_tree = gamboost("y ~ btree(x)", data={"x": x_tree, "y": y_tree}, family=Gaussian())
plot(mod_tree)
```

## 5. `brandom()`: Random Intercepts

`brandom()` handles grouped data by adding a random-intercept-like term.

```{code-cell} ipython3
group = np.repeat(["A", "B", "C", "D"], 25)
y_group = np.repeat([1, -1, 0.5, 2], 25) + rng.normal(scale=0.5, size=100)

mod_rand = gamboost("y ~ brandom(group)", data={"group": group, "y": y_group}, family=Gaussian())
plot(mod_rand)
```
