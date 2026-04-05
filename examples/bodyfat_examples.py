from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import AIC, Gaussian, Quantile, boost_control, cvrisk, gamboost, glmboost, plot, varimp

BODYFAT_PATH = ROOT / "data" / "bodyfat.csv"
BODYFAT_RESPONSE = "DEXfat"
BODYFAT_GARCIA_PREDICTORS = ["hipcirc", "kneebreadth", "anthro3a"]
BODYFAT_ALL_PREDICTORS = [
    "age",
    "waistcirc",
    "hipcirc",
    "elbowbreadth",
    "kneebreadth",
    "anthro3a",
    "anthro3b",
    "anthro3c",
    "anthro4",
]


def load_bodyfat() -> pl.DataFrame:
    return pl.read_csv(BODYFAT_PATH)


def _formula(response: str, predictors: list[str]) -> str:
    return f"{response} ~ " + " + ".join(predictors)


def _spline_formula(response: str, predictors: list[str], *, df: int = 4) -> str:
    terms = [f"bbs({name}, df={df}, knots=20)" for name in predictors]
    return f"{response} ~ " + " + ".join(terms)


def run_linear_bodyfat_example() -> dict[str, object]:
    bodyfat = load_bodyfat()
    formula = _formula(BODYFAT_RESPONSE, BODYFAT_GARCIA_PREDICTORS)
    fit = glmboost(
        formula,
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=200, nu=0.1),
    )
    aic = AIC(fit, method="corrected")
    importance = varimp(fit, type="blearner")
    return {
        "data": bodyfat,
        "formula": formula,
        "fit": fit,
        "aic": aic,
        "importance": importance,
        "plot": plot(fit),
        "importance_plot": plot(importance),
    }


def run_additive_bodyfat_example() -> dict[str, object]:
    bodyfat = load_bodyfat()
    formula = _spline_formula(BODYFAT_RESPONSE, BODYFAT_GARCIA_PREDICTORS, df=4)
    fit = gamboost(
        formula,
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=300, nu=0.1),
    )
    aic = AIC(fit, method="corrected")
    return {
        "data": bodyfat,
        "formula": formula,
        "fit": fit,
        "aic": aic,
        "partial_plot": plot(fit),
        "aic_plot": plot(aic),
    }


def run_bodyfat_model_selection_example() -> dict[str, object]:
    bodyfat = load_bodyfat()
    formula = _spline_formula(BODYFAT_RESPONSE, BODYFAT_ALL_PREDICTORS, df=4)
    fit = gamboost(
        formula,
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=400, nu=0.1),
    )
    cvm = cvrisk(
        formula,
        data=bodyfat,
        family=Gaussian(),
        control=boost_control(mstop=400, nu=0.1),
        folds=5,
    )
    pruned = fit.with_mstop(cvm.best_mstop)
    return {
        "data": bodyfat,
        "formula": formula,
        "fit": fit,
        "cvrisk": cvm,
        "pruned_fit": pruned,
        "cv_plot": plot(cvm),
        "partial_plot": plot(pruned),
        "selected_terms": pruned.selected,
    }


def run_bodyfat_quantile_example() -> dict[str, object]:
    bodyfat = load_bodyfat()
    formula = _formula(BODYFAT_RESPONSE, BODYFAT_GARCIA_PREDICTORS)
    fit = glmboost(
        formula,
        data=bodyfat,
        family=Quantile(0.5),
        control=boost_control(mstop=500, nu=0.1),
    )
    hipcirc = np.asarray(bodyfat["hipcirc"], dtype=np.float64)
    dexfat = np.asarray(bodyfat[BODYFAT_RESPONSE], dtype=np.float64)
    order = np.argsort(hipcirc)
    fitted_values = fit.fitted_[order]
    return {
        "data": bodyfat,
        "formula": formula,
        "fit": fit,
        "hipcirc_sorted": hipcirc[order],
        "dexfat_sorted": dexfat[order],
        "fitted_sorted": fitted_values,
    }


def main() -> None:
    linear = run_linear_bodyfat_example()
    additive = run_additive_bodyfat_example()
    selection = run_bodyfat_model_selection_example()
    quantile = run_bodyfat_quantile_example()
    print("Linear formula:", linear["formula"])
    print("Additive formula:", additive["formula"])
    print("Selected terms after CV:", selection["selected_terms"][:5])
    print("Quantile fit iterations:", quantile["fit"].mstop)


if __name__ == "__main__":
    main()
