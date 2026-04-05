# `pymboost` Documentation

This Jupyter Book is the primary documentation for `pymboost`, a Python implementation of `mboost`-style model-based boosting.

The goal of this book is to bridge the gap between R and Python by providing:
- **Mathematical Foundations**: The theory of functional gradient descent.
- **Hands-on Tutorials**: Step-by-step model building with real datasets.
- **Visual Parity**: Direct evidence of implementation fidelity against R.
- **Performance Benchmarks**: Quantifying the speed benefits of Numba acceleration.

## Navigation

- **{doc}`mathematical-foundations`**: The theory behind model-based boosting.
- **{doc}`getting-started`**: Setting up and the first boosting iteration.
- **{doc}`glmboost`**: Fitting and interpreting linear models.
- **{doc}`gamboost`**: Smoothing and additive effects.
- **{doc}`cv-and-stopping`**: Tuning the model via cross-validation.
- **{doc}`families`**: Exploring loss functions and custom gradients.
- **{doc}`baselearners`**: The semantics of various weak learners.
- **{doc}`performance`**: Python (Numba) vs. R benchmarking.
- **{doc}`parity-wall`**: Visual verification of R-parity.
