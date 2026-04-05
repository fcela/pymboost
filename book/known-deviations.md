# Known Deviations from R

The goal of `pymboost` is parity with `mboost`, but the project is not there yet. This page is intentionally explicit about the remaining gaps.

## Missing Features

- cyclic splines
- `bspatial()`
- `bmrf()`
- survival families
- `stabsel()`

## Behavioral Differences

- centered linear learners expose the intercept differently from R
- cross-validation defaults differ from R's bootstrap-oriented defaults
- some spline default details still require broader parity coverage
- `btree` is CART-backed in Python, not a conditional inference tree

## Numerical Differences

- quantile coefficient paths may differ term by term even when fitted functions agree closely
- constrained-solver paths can differ slightly while preserving the same shape restriction
- bootstrap or CV-based procedures can diverge if the resampling setup differs

This page should shrink over time, not disappear. Honest documentation of the parity boundary is part of the project’s value.
