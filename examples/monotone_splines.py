from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Gaussian, boost_control, gamboost, plot


def run_example() -> dict[str, object]:
    x = np.linspace(0.0, 1.0, 120)
    y = np.exp(2.0 * x) + 0.05 * np.sin(8.0 * np.pi * x)
    data = pl.DataFrame({"x": x, "y": y})

    fit = gamboost(
        'y ~ bmono(x, constraint="increasing", df=4, knots=10)',
        data=data,
        family=Gaussian(),
        control=boost_control(mstop=40, nu=0.1),
    )
    return {
        "data": data,
        "fit": fit,
        "selected_head": fit.selected[:10],
        "final_risk": float(fit.risk_[-1]),
        "partial_chart": plot(fit),
    }


def main() -> None:
    result = run_example()
    print(result["fit"].summary())
    print("Selected:", result["selected_head"])
    print("Final risk:", result["final_risk"])


if __name__ == "__main__":
    main()
