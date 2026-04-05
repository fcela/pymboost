from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import rpy2.robjects as ro

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Binomial, Gaussian, Poisson, boost_control, cvrisk, gamboost, glmboost


CASES = [
    "glmboost_gaussian_bols",
    "glmboost_binomial_bols",
    "glmboost_poisson_bols",
    "gamboost_gaussian_bbs_bols",
    "gamboost_gaussian_bmono",
    "gamboost_gaussian_btree",
    "cvrisk_gaussian_bols",
    "cvrisk_gaussian_bmono",
    "cvrisk_gaussian_btree",
]


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


def _summary(times: list[float]) -> dict[str, object]:
    return {
        "runs": len(times),
        "mean_s": float(statistics.mean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "stdev_s": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "raw_s": [float(t) for t in times],
    }


def _format_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Warm In-Process Benchmark Results",
        "",
        "| Case | Python mean (s) | R mean (s) | Python/R speedup |",
        "|---|---:|---:|---:|",
    ]
    for case, py in payload["python"].items():
        r = payload["r"][case]
        ratio = r["mean_s"] / py["mean_s"] if py["mean_s"] > 0.0 else float("inf")
        lines.append(f"| `{case}` | {py['mean_s']:.6f} | {r['mean_s']:.6f} | {ratio:.2f}x |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm in-process benchmark comparing Python and R fits.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()

    linear = _dataset_linear(args.n)
    spline = _dataset_spline(args.n)
    monotone = _dataset_monotone(args.n)
    tree = _dataset_tree(args.n)
    binomial = _dataset_binomial(args.n)
    poisson = _dataset_poisson(args.n)
    folds = _fold_ids(args.n, 5)
    fold_matrix = np.ones((args.n, 5), dtype=np.float64)
    for i in range(5):
        fold_matrix[folds == i, i] = 0.0

    r = ro.r
    fv = ro.FloatVector
    r("library(mboost)")
    r_glm = r(
        'function(x,z,y){glmboost(y ~ x + z, data=data.frame(x=x,z=z,y=y), family=Gaussian(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_glm_binomial = r(
        'function(x,z,y){glmboost(factor(y) ~ x + z, data=data.frame(x=x,z=z,y=y), family=Binomial(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_glm_poisson = r(
        'function(x,z,y){glmboost(y ~ x + z, data=data.frame(x=x,z=z,y=y), family=Poisson(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_gam = r(
        'function(x,z,y){gamboost(y ~ bbs(x, knots=20, df=4, degree=3, differences=2) + bols(z), data=data.frame(x=x,z=z,y=y), family=Gaussian(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_gam_bmono = r(
        'function(x,y){gamboost(y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2), data=data.frame(x=x,y=y), family=Gaussian(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_gam_btree = r(
        'function(x,z,y){gamboost(y ~ btree(x, z), data=data.frame(x=x,z=z,y=y), family=Gaussian(), control=boost_control(mstop=100, nu=0.1))}'
    )
    r_cv = r(
        'function(x,z,y,folds){cvrisk(gamboost(y ~ x + z, data=data.frame(x=x,z=z,y=y), family=Gaussian(), control=boost_control(mstop=50, nu=0.1)), folds=folds)}'
    )
    r_cv_bmono = r(
        'function(x,y,folds){cvrisk(gamboost(y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2), data=data.frame(x=x,y=y), family=Gaussian(), control=boost_control(mstop=50, nu=0.1)), folds=folds)}'
    )
    r_cv_btree = r(
        'function(x,y,folds){cvrisk(gamboost(y ~ btree(x), data=data.frame(x=x,y=y), family=Gaussian(), control=boost_control(mstop=50, nu=0.1)), folds=folds)}'
    )
    r_fold_matrix = r.matrix(fv(fold_matrix.reshape(-1)), nrow=args.n, ncol=5)

    python_cases = {
        "glmboost_gaussian_bols": lambda: glmboost(
            "y ~ x + z",
            data=linear,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "glmboost_binomial_bols": lambda: glmboost(
            "y ~ x + z",
            data=binomial,
            family=Binomial(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "glmboost_poisson_bols": lambda: glmboost(
            "y ~ x + z",
            data=poisson,
            family=Poisson(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "gamboost_gaussian_bbs_bols": lambda: gamboost(
            "y ~ bbs(x, knots=20, df=4, degree=3, differences=2) + bols(z)",
            data=spline,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "gamboost_gaussian_bmono": lambda: gamboost(
            'y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2)',
            data=monotone,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "gamboost_gaussian_btree": lambda: gamboost(
            "y ~ btree(x, z)",
            data=tree,
            family=Gaussian(),
            control=boost_control(mstop=100, nu=0.1),
        ),
        "cvrisk_gaussian_bols": lambda: cvrisk(
            "y ~ x + z",
            data=linear,
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        ),
        "cvrisk_gaussian_bmono": lambda: cvrisk(
            'y ~ bmono(x, constraint="increasing", knots=20, df=4, degree=3, differences=2)',
            data=monotone,
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        ),
        "cvrisk_gaussian_btree": lambda: cvrisk(
            "y ~ btree(x)",
            data=tree.select("x", "y"),
            family=Gaussian(),
            control=boost_control(mstop=50, nu=0.1),
            folds=folds,
        ),
    }
    r_cases = {
        "glmboost_gaussian_bols": lambda: r_glm(
            fv(np.asarray(linear["x"])),
            fv(np.asarray(linear["z"])),
            fv(np.asarray(linear["y"])),
        ),
        "glmboost_binomial_bols": lambda: r_glm_binomial(
            fv(np.asarray(binomial["x"])),
            fv(np.asarray(binomial["z"])),
            fv(np.asarray(binomial["y"])),
        ),
        "glmboost_poisson_bols": lambda: r_glm_poisson(
            fv(np.asarray(poisson["x"])),
            fv(np.asarray(poisson["z"])),
            fv(np.asarray(poisson["y"])),
        ),
        "gamboost_gaussian_bbs_bols": lambda: r_gam(
            fv(np.asarray(spline["x"])),
            fv(np.asarray(spline["z"])),
            fv(np.asarray(spline["y"])),
        ),
        "gamboost_gaussian_bmono": lambda: r_gam_bmono(
            fv(np.asarray(monotone["x"])),
            fv(np.asarray(monotone["y"])),
        ),
        "gamboost_gaussian_btree": lambda: r_gam_btree(
            fv(np.asarray(tree["x"])),
            fv(np.asarray(tree["z"])),
            fv(np.asarray(tree["y"])),
        ),
        "cvrisk_gaussian_bols": lambda: r_cv(
            fv(np.asarray(linear["x"])),
            fv(np.asarray(linear["z"])),
            fv(np.asarray(linear["y"])),
            r_fold_matrix,
        ),
        "cvrisk_gaussian_bmono": lambda: r_cv_bmono(
            fv(np.asarray(monotone["x"])),
            fv(np.asarray(monotone["y"])),
            r_fold_matrix,
        ),
        "cvrisk_gaussian_btree": lambda: r_cv_btree(
            fv(np.asarray(tree["x"])),
            fv(np.asarray(tree["y"])),
            r_fold_matrix,
        ),
    }

    unknown = [case for case in args.cases if case not in python_cases]
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {unknown}")
    python_cases = {name: python_cases[name] for name in args.cases}
    r_cases = {name: r_cases[name] for name in args.cases}

    for _ in range(2):
        for fn in python_cases.values():
            fn()
        for fn in r_cases.values():
            fn()

    def bench(fn):
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return _summary(times)

    payload = {
        "config": {"n": args.n, "runs": args.runs},
        "cases": list(args.cases),
        "python": {name: bench(fn) for name, fn in python_cases.items()},
        "r": {name: bench(fn) for name, fn in r_cases.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_warm_inprocess.json"
    md_path = args.output_dir / "benchmark_warm_inprocess.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_format_markdown(payload))
    print(json.dumps(payload, indent=2))
    print()
    try:
        print(md_path.relative_to(ROOT))
    except ValueError:
        print(md_path)


if __name__ == "__main__":
    main()
