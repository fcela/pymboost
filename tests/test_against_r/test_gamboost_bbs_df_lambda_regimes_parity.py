from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_low_df_bbs_regime(
    r_gamboost_bbs_runner,
):
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=8, df=2, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_runner(
        x,
        y,
        x_new=x,
        knots=8,
        lambda_value=None,
        df=2,
        degree=3,
        differences=2,
        center=False,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-6, rtol=1e-6)


def test_python_gamboost_matches_r_for_higher_df_bbs_regime_newdata_prediction(
    r_gamboost_bbs_runner,
):
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    x_new = np.linspace(0.1, 0.9, 9)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=8, df=6, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_runner(
        x,
        y,
        x_new=x_new,
        knots=8,
        lambda_value=None,
        df=6,
        degree=3,
        differences=2,
        center=False,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-10,
        rtol=1e-10,
    )
