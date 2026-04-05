from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import AIC, boost_control, gamboost


def load_cars() -> pl.DataFrame:
    # R datasets::cars
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


def run_example() -> dict[str, object]:
    cars = load_cars()

    cars_gb = gamboost(
        "dist ~ speed",
        data=cars,
        dfbase=4,
        control=boost_control(mstop=50),
    )
    aic = AIC(cars_gb, method="corrected")

    return {
        "data": cars,
        "fit": cars_gb,
        "aic": aic,
        "aic_value": float(aic),
        "aic_mstop": aic.mstop,
        "aic_df": float(aic.df),
        "first_selected": cars_gb.selected[0],
        "final_risk": float(cars_gb.risk_[-1]),
        "fitted_head": np.asarray(cars_gb.fitted_[:5], dtype=np.float64),
    }


def main() -> None:
    result = run_example()

    print(result["fit"])
    print(result["aic"])
    print("First selected learner:", result["first_selected"])
    print("Final empirical risk:", result["final_risk"])
    print("First 5 fitted values:", np.round(result["fitted_head"], 3))


if __name__ == "__main__":
    main()
