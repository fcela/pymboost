from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_gaussian_bbs_with_df_and_explicit_knots(
    r_gamboost_bbs_runner,
):
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=5, df=4, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_runner(
        x,
        y,
        knots=5,
        lambda_value=None,
        degree=3,
        differences=2,
        df=4,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_model.coefficients_["bbs(x, knots=5, df=4, degree=3, differences=2)"],
        r_result["coef"],
        atol=1e-12,
        rtol=1e-12,
    )
