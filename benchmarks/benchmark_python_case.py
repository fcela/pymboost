from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Binomial, Gaussian, Poisson, boost_control, cvrisk, gamboost, glmboost


def _dataset_linear(n: int) -> pl.DataFrame:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    z = np.cos(3.0 * x)
    y = 1.5 * x - 0.25 * z
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _dataset_spline(n: int) -> pl.DataFrame:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x) + 0.5 * z
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _dataset_monotone(n: int) -> pl.DataFrame:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.log1p(5.0 * x) + 0.1 * x
    return pl.DataFrame({"x": x, "y": y})


def _dataset_tree(n: int) -> pl.DataFrame:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    z = np.sin(4.0 * np.pi * x)
    y = np.where(x < -0.2, -1.0, np.where(x < 0.4, 0.5, 1.25)) + 0.25 * (z > 0.0)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _dataset_binomial(n: int) -> pl.DataFrame:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    z = np.cos(2.5 * x)
    score = 1.4 * x - 0.8 * z + 0.15 * np.sin(5.0 * x)
    y = (score > 0.1).astype(np.float64)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _dataset_poisson(n: int) -> pl.DataFrame:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    z = np.sin(2.0 * x)
    mu = np.exp(0.25 + 0.45 * x - 0.25 * z)
    y = np.rint(mu).astype(np.float64)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _fold_ids(n: int, k: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % k


def run_case(case: str, n: int) -> None:
    linear = _dataset_linear(n)
    spline = _dataset_spline(n)
    monotone = _dataset_monotone(n)
    tree = _dataset_tree(n)
    binomial = _dataset_binomial(n)
    poisson = _dataset_poisson(n)
    folds = _fold_ids(n, 5)
    if case == "glmboost_gaussian_bols":
        glmboost(
            "y ~ x + z",
            data=linear,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "gamboost_gaussian_bbs_bols":
        gamboost(
            "y ~ bbs(x, knots=20, df=4, degree=3, differences=2) + bols(z)",
            data=spline,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "cvrisk_gaussian_bols":
        cvrisk(
            "y ~ x + z",
            data=linear,
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        )
        return
    if case == "cvrisk_gaussian_btree":
        cvrisk(
            "y ~ btree(x)",
            data=linear.select("x", "y"),
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        )
        return
    if case == "gamboost_gaussian_bmono":
        gamboost(
            'y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2)',
            data=monotone,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "gamboost_gaussian_btree":
        gamboost(
            "y ~ btree(x, z)",
            data=tree,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "glmboost_binomial_bols":
        glmboost(
            "y ~ x + z",
            data=binomial,
            family=Binomial(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "glmboost_poisson_bols":
        glmboost(
            "y ~ x + z",
            data=poisson,
            family=Poisson(),
            control=boost_control(mstop=100, nu=0.1),
        )
        return
    if case == "cvrisk_gaussian_bmono":
        cvrisk(
            'y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2)',
            data=monotone,
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        )
        return
    raise ValueError(f"unknown case: {case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    run_case(args.case, args.n)


if __name__ == "__main__":
    main()
