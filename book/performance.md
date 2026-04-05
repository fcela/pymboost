---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Performance and Scaling

A major motivation for `pymboost` is operational efficiency. This chapter explores how the Python implementation (accelerated by Numba) compares to the original R `mboost`.

```{code-cell} ipython3
:tags: [hide-cell]
from book_utils import configure
import pandas as pd
import altair as alt

configure()

# Setup benchmark data
benchmark_table = pd.DataFrame(
    [
        ("glmboost_gaussian_bols", 0.001722, 0.007192, 4.18),
        ("glmboost_binomial_bols", 0.002420, 0.014220, 5.88),
        ("glmboost_poisson_bols", 0.001907, 0.008789, 4.61),
        ("gamboost_gaussian_bbs_bols", 0.006936, 0.020010, 2.89),
        ("gamboost_gaussian_bmono", 0.067572, 0.036580, 0.54),
        ("gamboost_gaussian_btree", 0.043395, 0.092202, 2.12),
        ("cvrisk_gaussian_bols", 0.004185, 0.178415, 42.64),
        ("cvrisk_gaussian_bmono", 0.072833, 0.141196, 1.94),
        ("cvrisk_gaussian_btree", 0.110702, 0.298336, 2.69),
    ],
    columns=["case", "python_mean_s", "r_mean_s", "python_over_r"],
)
```

## Warm In-Process Benchmarks

The following chart compares the mean fitting time for various model types. These timings represent "warm" starts, excluding interpreter startup costs.

```{code-cell} ipython3
alt.Chart(
    benchmark_table.melt(
        id_vars="case",
        value_vars=["python_mean_s", "r_mean_s"],
        var_name="implementation",
        value_name="seconds",
    )
).mark_bar().encode(
    x=alt.X("seconds:Q", title="Mean warm fit time (seconds)"),
    y=alt.Y("case:N", sort="-x", title=None),
    color=alt.Color("implementation:N", title=None, scale=alt.Scale(range=["#4c78a8", "#e45756"])),
    xOffset="implementation:N",
).properties(width=620, height=260, title="Warm in-process performance: Python vs R")
```

### Observations
- **Linear Models**: `pymboost` is significantly faster than R for basic GLM boosting.
- **Cross-Validation**: Python is over 40x faster for `cvrisk` on linear models, largely due to efficient parallelization and low-overhead kernels.
- **Tree Models**: Even with the scikit-learn backend, Python remains competitive or faster than R's `btree`.
- **Monotone Splines**: This is the only area where R's specialized C implementation can be faster, though Python remains within an order of magnitude.

## Why Python is Competitive

The speed advantage comes from three architectural choices:

1.  **Numba Acceleration**: The hot inner loops (fitting and scoring learners) are JIT-compiled to machine code.
2.  **Precomputation**: Design matrices and penalty factors are computed once and reused across the boosting path.
3.  **Modern Array Core**: Efficient use of NumPy/SciPy linear algebra routines.
