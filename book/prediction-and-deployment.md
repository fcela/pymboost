# Prediction and Deployment

Every fitted model ultimately needs to score new observations. `pymboost` already supports this, but there are a few details worth making explicit.

## `predict(newdata=...)`

All fitted models expose:

```python
model.predict(newdata=...)
model.predict(newdata=..., type="response")
```

The default return value is the linear predictor (`type="link"`). For families with a non-identity link you can ask for response-scale output:

- Binomial: `type="response"` returns probabilities
- Poisson / Gamma: `type="response"` returns mean-scale predictions
- Gaussian, Laplace, Quantile, Expectile, and Huber currently use the identity mapping

## `with_mstop(...)` and `mstop(...)`

You can truncate a fitted boosting path cheaply:

- `mstop(model, k)`
- `model.with_mstop(k)`

This is valuable operationally because it avoids refitting the entire model when you already know the desired stopping point from `cvrisk`, `AIC`, or a manual inspection.

## Extrapolation Depends on the Learner

- linear learners extrapolate linearly
- splines follow the spline basis / boundary behavior
- trees predict the value of the terminal region they fall into
- constrained splines preserve the constraint shape but still depend on the basis at the boundary

So deployment behavior is not uniform across learners; it is part of the learner definition.
