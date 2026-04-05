from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_centered_default_bbs_newdata_prediction(
    r_gamboost_bbs_runner,
):
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    x_new = np.linspace(0.1, 0.9, 9)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, center=True)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_runner(
        x,
        y,
        x_new=x_new,
        knots=None,
        lambda_value=None,
        df=4,
        degree=3,
        differences=2,
        center=True,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-11,
        rtol=1e-11,
    )
