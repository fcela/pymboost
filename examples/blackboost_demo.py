from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mboost import Gaussian, TreeControls, blackboost, boost_control


def run_example() -> dict[str, object]:
    x = np.linspace(-1.0, 1.0, 80)
    z = np.sin(4.0 * np.pi * x)
    y = np.where(x < -0.2, -1.0, np.where(x < 0.4, 0.5, 1.25)) + 0.25 * (z > 0.0)

    fit = blackboost(
        "y ~ x + z",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=20, nu=0.1),
        tree_controls=TreeControls(maxdepth=2, minsplit=10, minbucket=4),
    )

    return {
        "fit": fit,
        "formula": fit.formula,
        "selected_head": fit.selected[:5],
        "final_risk": float(fit.risk_[-1]),
    }


def main() -> None:
    result = run_example()

    print(result["formula"])
    print("Selected:", result["selected_head"])
    print("Final risk:", result["final_risk"])


if __name__ == "__main__":
    main()
