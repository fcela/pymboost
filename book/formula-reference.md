# Formula Language Reference

The formula interface is a major part of the package experience. The frontend is Formulaic-backed, but the learner semantics are `pymboost`-specific.

## Basic Syntax

- `y ~ x1 + x2`
- `y ~ .`
- `y ~ x1 + x2 - x3`
- `y ~ 0 + x1`

## Learner Calls

- `bols(x)`
- `bols(x, lambda_=0.1)`
- `bols(x, df=1)`
- `bbs(x, df=4, knots=20)`
- `bmono(x, constraint="increasing")`
- `btree(x, z, maxdepth=2)`
- `brandom(group, df=4)`

## Modifiers

- `bbs(x, by=z)`
- `bmono(x, by=z)`
- `btree(x, z, by=g)`

## Interactions

Currently supported examples:

- `x1:x2`
- `x1*x2`

Current limitations remain for the broader R tensor-product algebra and multivariate spline calls.

## Factors

Factor handling is supported, including:

- automatic dummy expansion where appropriate
- explicit `bols(factor_var)`
- grouped effects with `brandom(group)`

But factor semantics still differ from R in some advanced corners, which are documented in [Known Deviations from R](known-deviations.md).
