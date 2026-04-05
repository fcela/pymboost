from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mboost import Gaussian, boost_control, gamboost, glmboost


def test_glmboost_accepts_pandas_dataframe():
    df = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 20),
            "y": np.linspace(-1.0, 1.0, 20) * 2.0 + 0.1,
        }
    )

    model = glmboost(
        "y ~ x",
        data=df,
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.fitted_.shape == (20,)
    assert model.risk_[0] > model.risk_[-1]


def test_gamboost_accepts_polars_dataframe_and_dfbase():
    x = np.linspace(0.0, 1.0, 30)
    df = pl.DataFrame(
        {
            "x": x,
            "y": np.sin(2.0 * np.pi * x),
        }
    )

    model = gamboost(
        "y ~ x",
        data=df,
        dfbase=4,
        control=boost_control(mstop=8, nu=0.1),
    )

    assert "bbs(x, df=4, knots=20)" in model.formula
    assert model.fitted_.shape == (30,)
    assert model.risk_[0] > model.risk_[-1]
