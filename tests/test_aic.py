from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mboost import AIC, boost_control, gamboost


def _cars_frame() -> pl.DataFrame:
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


def test_corrected_aic_matches_r_cars_example():
    cars = _cars_frame()
    cars_gb = gamboost(
        "dist ~ speed",
        data=cars,
        dfbase=4,
        control=boost_control(mstop=50),
    )

    result = AIC(cars_gb, method="corrected")

    assert float(result) == pytest.approx(6.575865764262378, abs=1e-12)
    assert result.mstop == 26
    assert result.df == 3.905132697691835
