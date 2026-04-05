# Practical Workflow and Gotchas

The most useful workflow is usually iterative rather than one-shot.

## A Typical Workflow

1. start with `glmboost` for a sparse linear screen
2. inspect selected terms and `varimp`
3. replace promising linear terms with `bbs` or `bmono`
4. tune `mstop` with `cvrisk` or `AIC`
5. inspect partial effects and confidence intervals
6. refit or simplify

## Common Gotchas

- Highly correlated predictors can split credit unstably across the path.
- Larger `mstop` is not always better; the path can overfit slowly.
- Smaller `nu` gives a smoother path but needs more iterations.
- Partial-effect plots are not standalone response curves; they are contributions to the linear predictor.
- Python and R expose the intercept differently for centered linear learners.
- Python and R may choose different stopping points if the resampling defaults differ.

## Weighted Fits

`weights=` is supported across fitting and resampling APIs. This matters for:

- survey weights
- inverse-probability weighting
- case-control or case-cohort adjustments

Weights do not just scale the final loss; they change the weak-learner fitting problem itself.
