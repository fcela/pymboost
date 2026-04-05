# Inference and Confidence Intervals

`pymboost` now exposes `confint()` as a first-class feature. That matters because one reason to choose model-based boosting over black-box boosting is that partial effects can be accompanied by uncertainty summaries.

Current scope:

- `method="normal"`:
  - Gaussian only
  - pointwise intervals
  - hat-matrix / smoother based approximation
- `method="bootstrap"`:
  - available more broadly
  - refits the model on bootstrap-weighted samples
  - can re-tune `mstop` inside each replicate with `B_mstop`

The worked examples appear in [Boosted Linear Models](glmboost.md), [Boosted Additive Models](gamboost.md), and throughout the [Base-Learner Handbook](baselearners.md). This chapter collects the interpretation.

## What `confint()` Returns

- fitted-predictor intervals when `which=None`
- one-dimensional partial-effect intervals when `which` identifies a single learner

The result is a `ConfIntResult` object with:

- `estimate`
- `std_error`
- `lower`
- `upper`

and can be visualized directly with `plot(confint_result)`.

## Normal Approximation vs Bootstrap

Normal approximation is useful when:

- the family is Gaussian,
- the model is reasonably smooth,
- speed matters,
- and a pointwise approximation is acceptable.

Bootstrap is preferable when:

- the family is non-Gaussian,
- the effect is strongly nonlinear,
- the small-sample distribution is uncertain,
- or you want the stopping rule re-tuned inside each replicate.

## Inner `mstop` Tuning with `B_mstop`

The bootstrap path can optionally run inner `cvrisk` inside each bootstrap replicate. This is closer to what advanced `mboost::confint` workflows do, because it propagates stopping uncertainty into the interval width.

This is computationally expensive, but statistically more honest than pretending the original `mstop` is fixed without error.

## What Is Still Missing

- full `mboost::confint` parity across all learner classes
- tree-based intervals
- richer `newdata` and `which` semantics
- simultaneous bands rather than only pointwise bands
