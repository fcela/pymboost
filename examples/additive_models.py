from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Gaussian, boost_control, gamboost, plot, varimp


def run_example() -> dict[str, object]:
    x = np.linspace(0.0, 1.0, 120)
    z = np.linspace(-1.0, 1.0, 120)
    y = np.sin(2.0 * np.pi * x) + 0.6 * z
    data = pl.DataFrame({"x": x, "z": z, "y": y})

    fit = gamboost(
        "y ~ bbs(x, knots=8, df=4, degree=3, differences=2) + bols(z)",
        data=data,
        family=Gaussian(),
        control=boost_control(mstop=60, nu=0.1),
    )
    importance = varimp(fit)
    return {
        "data": data,
        "fit": fit,
        "selected_head": fit.selected[:10],
        "risk_head": np.asarray(fit.risk_[:5], dtype=np.float64),
        "partial_chart": plot(fit),
        "varimp": importance,
        "varimp_chart": plot(importance),
    }


def main() -> None:
    result = run_example()
    print(result["fit"].summary())
    print("Selected:", result["selected_head"])
    print("Risk head:", result["risk_head"])


if __name__ == "__main__":
    main()
