# Model Diagnostics and Selection

Model-based boosting does not stop at fitted values. `pymboost` already exposes three complementary diagnostics that should be part of a normal workflow.

## `AIC()`

`AIC(model, method="corrected")` is currently implemented for Gaussian models. It returns:

- the optimal `mstop`
- the effective degrees of freedom at that stopping point
- the full AIC path

This is useful when you want a fast stopping heuristic without full resampling.

## `hatvalues()`

`hatvalues(model)` returns:

- the leverage values on the training observations
- the trace path, which acts like an effective-degrees-of-freedom path

This is the bridge between the boosting algorithm and classical smoother diagnostics. It is also the object behind the current Gaussian normal-approximation `confint`.

## `varimp()`

`varimp(model)` summarizes how much each learner or variable contributed along the boosting path. Two readings matter:

- risk-reduction importance
- selection frequency

These are not the same. A learner can win often but only make small improvements each time, or win rarely and make large risk reductions.

## When To Use What

- Use `cvrisk` when predictive stopping is the main goal.
- Use `AIC` when you want a fast Gaussian model-selection heuristic.
- Use `hatvalues` when you want a complexity diagnostic.
- Use `varimp` when you want an importance summary, not a stopping rule.
