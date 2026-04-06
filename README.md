# `pymboost`

Python implementation of `mboost`-style model-based boosting with a NumPy/SciPy core, Numba acceleration, and Formulaic-based formula parsing.

Current implemented subset:

- `glmboost`
- `gamboost`
- `bols`, `bbs`, `bmono`, `brandom`, `btree`, `blackboost`
- Gaussian, Binomial, Poisson, GammaReg, Laplace, Quantile, Expectile, and Huber families
- `cvrisk`, `mstop`, `confint`, corrected Gaussian `AIC`, and Gaussian `hatvalues`
- Altair-based plotting and variable importance

## Install

```bash
source ~/venv/lab/bin/activate
pip install -e '.[plot]'
```

For local development with tests, docs, and packaging tools:

```bash
pip install -e '.[all]'
```

## Install From Git

If you want to install directly from a Git repository rather than PyPI:

```bash
pip install "git+https://github.com/fcela/pymboost.git"
```

With plotting extras:

```bash
pip install "git+https://github.com/fcela/pymboost.git#egg=pymboost[plot]"
```

For editable development from a clone:

```bash
git clone https://github.com/fcela/pymboost.git
cd pymboost
source ~/venv/lab/bin/activate
pip install -e '.[all]'
```

## Quick Start

```python
import numpy as np
import polars as pl

from mboost import Gaussian, boost_control, gamboost, plot

x = np.linspace(0.0, 1.0, 100)
z = np.linspace(-1.0, 1.0, 100)
y = np.sin(2.0 * np.pi * x) + 0.5 * z

data = pl.DataFrame({"x": x, "z": z, "y": y})

fit = gamboost(
    "y ~ bbs(x, df=4) + bols(z)",
    data=data,
    family=Gaussian(),
    control=boost_control(mstop=60, nu=0.1),
)

print(fit)
plot(fit)
```

## Docs

- Live manual: https://fcela.github.io/pymboost/intro.html
- Canonical docs source: [book](/Users/fcela/src/public_github/pymboost/book)
- Status and roadmap: [status-and-roadmap.md](/Users/fcela/src/public_github/pymboost/book/status-and-roadmap.md)
- Hands-on tutorial: [hands-on-tutorial.md](/Users/fcela/src/public_github/pymboost/book/hands-on-tutorial.md)
- Migration guide: [migration-from-r.md](/Users/fcela/src/public_github/pymboost/book/migration-from-r.md)
- Plotting inventory: [PLOTTING_INVENTORY.md](/Users/fcela/src/public_github/pymboost/PLOTTING_INVENTORY.md)
- Parity gaps: [PARITY_GAPS.md](/Users/fcela/src/public_github/pymboost/PARITY_GAPS.md)

Render the book locally with Jupyter Book:

```bash
source ~/venv/lab/bin/activate
PYTHONPATH=. jupyter-book build book
```

The rendered site will be available under `book/_build/html/`.

## GitHub Pages

The repository now includes a GitHub Actions workflow for Pages deployment at
[github-pages.yml](/Users/fcela/src/public_github/pymboost/.github/workflows/github-pages.yml).

To publish the manual on GitHub Pages:

1. Push this repository to GitHub.
2. In GitHub, open `Settings -> Pages`.
3. Set the source to `GitHub Actions`.
4. Push to `main` (or `master` if that is your default branch), or run the
   `github-pages` workflow manually from the Actions tab.

The workflow builds the Jupyter Book from `book/` and publishes
`book/_build/html/` to GitHub Pages.

## Publishing

Publishing notes for TestPyPI / PyPI live in
[PUBLISHING.md](/Users/fcela/src/public_github/pymboost/PUBLISHING.md).

## Packaging

Build source and wheel distributions with:

```bash
pip install -e '.[build]'
python -m build
```

Artifacts will be created under `dist/`.

## GitLab CI/CD

The repository is configured for GitLab CI/CD via [.gitlab-ci.yml](/Users/fcela/src/public_github/pymboost/.gitlab-ci.yml):

- `test`: installs R + `mboost`, installs the Python package with test extras, and runs the full suite
- `build`: builds the source distribution and wheel into `dist/`
- `pages`: builds the Jupyter Book and publishes it as GitLab Pages from `public/`

The helper scripts used by the pipeline live under [scripts/ci](/Users/fcela/src/public_github/pymboost/scripts/ci).

## Status

This repo has strong R-backed parity for the currently implemented core subset, especially Gaussian `glmboost` / `gamboost`, explicit `bols(...)`, explicit `bbs(...)`, covered `bmono(...)` cases, `GammaReg`, `cvrisk`, corrected Gaussian `AIC`, `varimp`, and covered bootstrap `confint(...)` slices.

It is not yet full package parity with R `mboost`.

Performance depends on how it is measured. The warm in-process benchmark currently favors Python on most measured core cases, while the one-shot subprocess benchmark still favors R because interpreter and process startup overhead dominate there. Read both benchmark tables before making blanket speed claims.

Important caveat: `btree` is still provisional. The current Python implementation uses a CART-style sklearn backend, while R `mboost::btree` is built on conditional inference trees from `partykit`.

`blackboost` is available as a CART-backed wrapper over the same tree engine. It currently rewrites a plain formula such as `y ~ x + z` into a single multi-feature `btree(...)` learner with configurable depth/split controls, rather than reproducing full `mboost::blackboost` / `partykit` semantics.
