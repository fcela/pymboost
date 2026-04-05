# Migration Guide: Coming from R `mboost`

Many users of `pymboost` will already know `mboost`. The biggest translations are straightforward, but not identical.

| R | Python | Notes |
|---|---|---|
| `glmboost(DEXfat ~ ., data=bodyfat)` | `glmboost("DEXfat ~ .", data=bodyfat)` | Formula is a string |
| `coef(model, off2int=TRUE)` | `coef(model)` plus explicit offset handling | Different intercept convention |
| `cvrisk(model)` | `cvrisk(formula, data=..., ...)` | Standalone function |
| `model[mstop(cv)]` | `mstop(model, cv.best_mstop)` | Returns truncated model |
| `predict(model, newdata, type="response")` | `model.predict(newdata=..., type="response")` | Response-scale prediction is available for the implemented families |
| `Family(...)` closures | subclass `Family` | Same conceptual contract, different syntax |

The important practical differences today are:

- Python stores centered linear effects with a separate offset.
- `cvrisk(...)` now defaults to bootstrap-style resampling when folds are omitted, but the API is still a standalone function rather than an S3 method on a fitted model.
- Some advanced learners and survival families are still missing.
