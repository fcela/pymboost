from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import (
    AIC,
    Gaussian,
    boost_control,
    cvrisk,
    gamboost,
    glmboost,
    plot,
    varimp,
)


def run_example() -> dict[str, object]:
    n = 120

    x = np.linspace(0.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, n)
    y_add = np.sin(2.0 * np.pi * x) + 0.75 * z
    additive_data = pl.DataFrame({"x": x, "z": z, "y": y_add})
    additive_fit = gamboost(
        "y ~ bbs(x, knots=8, df=4, degree=3, differences=2) + bols(z)",
        data=additive_data,
        family=Gaussian(),
        control=boost_control(mstop=60, nu=0.1),
    )

    groups = np.array(["slow", "medium", "fast", "very fast"] * 30, dtype=object)
    xg = np.linspace(-1.0, 1.0, groups.shape[0])
    group_effect = {"slow": -1.0, "medium": -0.15, "fast": 0.55, "very fast": 1.15}
    yg = np.array([group_effect[g] for g in groups], dtype=np.float64) + 0.15 * xg
    factor_data = {"x": xg, "group": groups, "y": yg}
    factor_fit = glmboost(
        "y ~ group",
        data=factor_data,
        family=Gaussian(),
        control=boost_control(mstop=30, nu=0.1),
    )

    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.cos(3.0 * np.pi * x1)
    y_surface = np.where(x1 + x2 > 0.0, 1.2, -0.8) + 0.2 * x1
    surface_data = {"x1": x1, "x2": x2, "y": y_surface}
    surface_fit = glmboost(
        "y ~ btree(x1, x2)",
        data=surface_data,
        family=Gaussian(),
        control=boost_control(mstop=30, nu=0.1),
    )

    x3 = np.sin(5.0 * np.pi * x1)
    y_sensitivity = np.where(x1 > 0.1, 1.0, -0.6) + 0.35 * (x2 > 0.0) - 0.2 * x3
    sensitivity_data = {"x1": x1, "x2": x2, "x3": x3, "y": y_sensitivity}
    sensitivity_fit = glmboost(
        "y ~ btree(x1, x2, x3)",
        data=sensitivity_data,
        family=Gaussian(),
        control=boost_control(mstop=30, nu=0.1),
    )

    cars_speed = np.array(
        [
            4, 4, 7, 7, 8, 9, 10, 10, 10, 11,
            11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
            14, 14, 14, 15, 15, 15, 16, 16, 17, 17,
            17, 18, 18, 18, 18, 19, 19, 19, 20, 20,
            20, 20, 20, 22, 23, 24, 24, 24, 24, 25,
        ],
        dtype=np.float64,
    )
    cars_dist = np.array(
        [
            2, 10, 4, 22, 16, 10, 18, 26, 34, 17,
            28, 14, 20, 24, 28, 26, 34, 34, 46, 26,
            36, 60, 80, 20, 26, 54, 32, 40, 32, 40,
            50, 42, 56, 76, 84, 36, 46, 68, 32, 48,
            52, 56, 64, 66, 54, 70, 92, 93, 120, 85,
        ],
        dtype=np.float64,
    )
    cars_data = pl.DataFrame({"speed": cars_speed, "dist": cars_dist})
    cars_fit = gamboost(
        "dist ~ speed",
        data=cars_data,
        dfbase=4,
        control=boost_control(mstop=50),
    )
    cars_cv = cvrisk(
        "dist ~ speed",
        data=cars_data,
        family=Gaussian(),
        control=boost_control(mstop=50),
        folds=5,
    )
    cars_aic = AIC(cars_fit, method="corrected")

    additive_varimp = varimp(additive_fit)
    additive_blearner_varimp = varimp(additive_fit, type="blearner")

    return {
        "additive_fit": additive_fit,
        "factor_fit": factor_fit,
        "surface_fit": surface_fit,
        "sensitivity_fit": sensitivity_fit,
        "cars_fit": cars_fit,
        "cars_cv": cars_cv,
        "cars_aic": cars_aic,
        "numeric_partial_chart": plot(additive_fit, which="bbs(x, knots=8, df=4, degree=3, differences=2)"),
        "categorical_partial_chart": plot(factor_fit),
        "surface_chart": plot(surface_fit),
        "sensitivity_chart": plot(sensitivity_fit),
        "variable_importance_chart": plot(additive_varimp),
        "blearner_importance_chart": plot(additive_blearner_varimp),
        "cvrisk_chart": plot(cars_cv),
        "aic_chart": plot(cars_aic),
        "variable_importance": additive_varimp.to_pandas(),
        "blearner_importance": additive_blearner_varimp.to_pandas(),
    }


def main() -> None:
    result = run_example()
    print("Charts:")
    print(
        ", ".join(
            [
                "numeric_partial_chart",
                "categorical_partial_chart",
                "surface_chart",
                "sensitivity_chart",
                "variable_importance_chart",
                "blearner_importance_chart",
                "cvrisk_chart",
                "aic_chart",
            ]
        )
    )
    print("AIC mstop:", result["cars_aic"].mstop)
    print("CV best mstop:", result["cars_cv"].best_mstop)


if __name__ == "__main__":
    main()
