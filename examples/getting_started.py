from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Gaussian, boost_control, gamboost


def run_example() -> dict[str, object]:
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

    return {
        "data": data,
        "fit": fit,
        "summary": fit.summary(),
        "selected_head": fit.selected[:5],
        "final_risk": float(fit.risk_[-1]),
    }


def main() -> None:
    result = run_example()

    print(result["summary"])
    print("Selected:", result["selected_head"])
    print("Final risk:", result["final_risk"])


if __name__ == "__main__":
    main()
