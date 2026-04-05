from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "tutorial_assets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import AIC, Binomial, Gaussian, boost_control, cvrisk, gamboost, glmboost, plot, varimp


def load_cars() -> pl.DataFrame:
    speed = np.array(
        [
            4, 4, 7, 7, 8, 9, 10, 10, 10, 11,
            11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
            14, 14, 14, 15, 15, 15, 16, 16, 17, 17,
            17, 18, 18, 18, 18, 19, 19, 19, 20, 20,
            20, 20, 20, 22, 23, 24, 24, 24, 24, 25,
        ],
        dtype=np.float64,
    )
    dist = np.array(
        [
            2, 10, 4, 22, 16, 10, 18, 26, 34, 17,
            28, 14, 20, 24, 28, 26, 34, 34, 46, 26,
            36, 60, 80, 20, 26, 54, 32, 40, 32, 40,
            50, 42, 56, 76, 84, 36, 46, 68, 32, 48,
            52, 56, 64, 66, 54, 70, 92, 93, 120, 85,
        ],
        dtype=np.float64,
    )
    return pl.DataFrame({"speed": speed, "dist": dist})


def save_chart(name: str, chart) -> None:
    path = ASSET_DIR / f"{name}.svg"
    chart.save(path)
    print(path.relative_to(ROOT))


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    cars = load_cars()
    cars_fit = gamboost(
        "dist ~ speed",
        data=cars,
        dfbase=4,
        control=boost_control(mstop=50, nu=0.1),
    )
    cars_cv = cvrisk(
        "dist ~ speed",
        data=cars,
        family=Gaussian(),
        control=boost_control(mstop=50, nu=0.1),
        folds=5,
    )
    cars_aic = AIC(cars_fit, method="corrected")
    save_chart("cars_partial", plot(cars_fit, width=320, height=220))
    save_chart("cars_cvrisk", plot(cars_cv, width=500, height=260))
    save_chart("cars_aic", plot(cars_aic, width=500, height=260))

    x = np.linspace(0.0, 1.0, 120)
    z = np.linspace(-1.0, 1.0, 120)
    y = np.sin(2.0 * np.pi * x) + 0.6 * z
    additive = pl.DataFrame({"x": x, "z": z, "y": y})
    additive_fit = gamboost(
        "y ~ bbs(x, knots=8, df=4, degree=3, differences=2) + bols(z)",
        data=additive,
        family=Gaussian(),
        control=boost_control(mstop=60, nu=0.1),
    )
    save_chart("additive_partial", plot(additive_fit, width=260, height=200))
    save_chart("additive_varimp", plot(varimp(additive_fit), width=420, height=140))

    xm = np.linspace(0.0, 1.0, 120)
    ym = np.log1p(8.0 * xm)
    mono = pl.DataFrame({"x": xm, "y": ym})
    mono_fit = gamboost(
        "y ~ bmono(x, constraint='increasing', knots=7, lambda=0.5, degree=3, differences=2)",
        data=mono,
        family=Gaussian(),
        control=boost_control(mstop=40, nu=0.1),
    )
    save_chart("monotone_partial", plot(mono_fit, width=320, height=220))

    xt = np.linspace(-1.0, 1.0, 80)
    zt = np.sin(3.0 * xt)
    yt = xt * zt + 0.2 * xt
    tree2 = pl.DataFrame({"x": xt, "z": zt, "y": yt})
    tree2_fit = gamboost(
        "y ~ btree(x, z)",
        data=tree2,
        family=Gaussian(),
        control=boost_control(mstop=20, nu=0.1),
    )
    save_chart("tree_surface", plot(tree2_fit, width=260, height=220))

    wt = np.cos(4.0 * xt)
    y3 = xt - 0.5 * zt + 0.25 * wt
    tree3 = pl.DataFrame({"x": xt, "z": zt, "w": wt, "y": y3})
    tree3_fit = gamboost(
        "y ~ btree(x, z, w)",
        data=tree3,
        family=Gaussian(),
        control=boost_control(mstop=20, nu=0.1),
    )
    save_chart("tree_sensitivity", plot(tree3_fit, width=220, height=160))

    xb = np.linspace(-2.0, 2.0, 120)
    logits = 2.0 * xb
    probs = 1.0 / (1.0 + np.exp(-logits))
    yb = (probs > 0.5).astype(np.float64)
    binary = pl.DataFrame({"x": xb, "y": yb})
    binomial_fit = glmboost(
        "y ~ x",
        data=binary,
        family=Binomial(),
        control=boost_control(mstop=40, nu=0.1),
    )
    save_chart("binomial_partial", plot(binomial_fit, width=320, height=220))


if __name__ == "__main__":
    main()
