from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.cars_gamboost import load_cars
from mboost import AIC, Gaussian, boost_control, cvrisk, gamboost


def run_example() -> dict[str, object]:
    cars = load_cars()

    fit = gamboost(
        "dist ~ speed",
        data=cars,
        dfbase=4,
        control=boost_control(mstop=50),
    )
    aic = AIC(fit, method="corrected")
    cvm = cvrisk(
        "dist ~ speed",
        data=cars,
        family=Gaussian(),
        control=boost_control(mstop=50),
        folds=5,
    )
    return {
        "data": cars,
        "fit": fit,
        "aic": aic,
        "cvrisk": cvm,
        "aic_value": float(aic),
        "aic_mstop": aic.mstop,
        "aic_df": float(aic.df),
        "cv_best_mstop": cvm.best_mstop,
        "cv_risk_head": np.asarray(cvm.risk[:5], dtype=np.float64),
    }


def main() -> None:
    result = run_example()

    print("AIC:", result["aic_value"])
    print("AIC mstop:", result["aic_mstop"])
    print("AIC df:", result["aic_df"])
    print("CV best mstop:", result["cv_best_mstop"])
    print("CV first 5 risks:", result["cv_risk_head"])


if __name__ == "__main__":
    main()
