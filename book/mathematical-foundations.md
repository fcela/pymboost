---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Mathematical Foundations

Model-based boosting is a statistical framework that combines the principles of gradient boosting with structured, interpretable models like GLMs and GAMs.

## Why use model-based boosting?

The niche of `mboost` and `pymboost` is narrower and more statistical than generic gradient-boosted trees. You reach for this package when you want:

- **Interpretable Structure**: Additive models that stay interpretable term by term.
- **Variable Selection**: Automatic selection driven by the boosting path.
- **Heterogeneous Learners**: Mixing linear terms, splines, monotone splines, and trees.
- **Custom Losses**: Optimization beyond OLS using the same machinery.

## The Statistical Problem

Model-based boosting is framed as empirical risk minimization. Let

$$
f^\star = \arg\min_f \mathbb{E}_{Y, X}\left[\rho(Y, f(X))\right].
$$

Because the population expectation is unavailable, we minimize the empirical risk:

$$
\mathcal{R}(f) = \sum_{i=1}^n \rho(y_i, f(x_i)).
$$

### Functional Gradient Descent

The algorithm proceeds iteratively. In each step $m$:

1.  **Negative Gradient**: Calculate the "pseudo-residuals" $u_i$:
    $$
    u_i = - \left[ \frac{\partial}{\partial f} \rho(y_i, f) \right]_{f = \hat{f}_{m-1}(x_i)}
    $$
2.  **Base-learner Selection**: Fit all candidate base-learners $h_j(x)$ to $u_i$ using weighted least squares.
3.  **Update**: Choose the best-fitting learner $h_{j^\star}$ and update:
    $$
    \hat{f}_m = \hat{f}_{m-1} + \nu \hat{h}_{j^\star}
    $$
    where $\nu \in (0, 1]$ is the learning rate.

### L2 Loss Example
For Gaussian (L2) loss, $\rho(y, f) = \frac{1}{2}(y - f)^2$. The negative gradient is simply the residual:
$$
u_i = - \frac{\partial}{\partial f} \left[ \frac{1}{2}(y_i - f)^2 \right] = y_i - f(x_i)
$$
This makes the connection to traditional "residual boosting" explicit.
