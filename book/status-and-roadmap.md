# Status and Roadmap

This is the canonical high-level status page for the current `pymboost` repository state.

## Current Implemented Surface

The repository already covers a substantial subset of `mboost`:

- APIs: `glmboost`, `gamboost`, provisional `blackboost`, `cv`, `cvrisk`, `mstop`
- base learners: `bols`, `bbs`, `bmono`, `brandom`, `btree`
- families: Gaussian, Binomial, Poisson, GammaReg, Laplace, Quantile, Expectile, Huber
- diagnostics and helpers: corrected Gaussian `AIC`, Gaussian `hatvalues`, `varimp`, `confint`, Altair plotting
- prediction: link-scale and response-scale prediction via `model.predict(type="link" | "response")`
- resampling: `cv()` supports k-fold, bootstrap, and subsampling generation, and `cvrisk()` now exposes the same resampling controls directly when folds are generated internally

## Verification Status

The project is verified against R `mboost` with direct `rpy2`-backed tests for the implemented surface.

Current verified slices include:

- core `glmboost` / `gamboost` path behavior
- Gaussian, Binomial, Poisson, GammaReg, Laplace, Quantile, Expectile, and Huber coverage where implemented
- explicit `bols(...)`, `bbs(...)`, covered `bmono(...)`, `brandom(...)`, and selected tree cases
- `cvrisk`, corrected Gaussian `AIC`, `varimp`, and covered bootstrap `confint(...)` slices
- prediction parity for supported learner/family combinations on the link scale, plus covered response-scale slices for Binomial, Poisson, and GammaReg

For the detailed gap inventory, see the repository file `PARITY_GAPS.md`.

## Performance Status

Performance depends on benchmark mode:

- warm in-process benchmarks currently favor Python on most measured core cases
- one-shot subprocess / CLI benchmarks still favor R because interpreter and process startup overhead dominate there

So the correct statement is not simply "Python is faster" or "Python is slower." It depends on whether you care about steady-state fitting inside a running process or whole-command execution.

## Main Remaining Gaps

The largest remaining package-surface gaps are:

- survival families such as CoxPH, Gehan, and Cindex
- spatial / graph learners such as `bspatial` and `bmrf`
- broader formula parity and cleaner semantics around complex R-style terms
- broader spline parity beyond the currently tested `bbs` / `bmono` slices
- fuller tree parity with `partykit` rather than the current CART-style backend
- stronger parity around R's default `cvrisk` behavior and broader resampling semantics
- broader extractor / diagnostic parity beyond the currently implemented subset

## Recommended Next Milestones

If the goal is the next highest-value delivery sequence:

1. implement the first survival family
2. broaden `bbs` and constrained-spline parity coverage
3. improve formula semantics and reduce special-casing
4. deepen tree parity or document the CART-vs-`partykit` divergence more sharply
5. keep expanding R-backed edge-case and multi-term tests
