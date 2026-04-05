from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost, glmboost


def test_python_gamboost_matches_r_for_gaussian_multiterm_newdata_prediction(
    r_gamboost_gaussian_predict_runner,
):
    x = np.linspace(0.0, 1.0, 50)
    z = np.linspace(-1.0, 1.0, 50)
    y = np.sin(2.0 * np.pi * x) + 0.5 * z
    x_new = np.linspace(0.1, 0.9, 9)
    z_new = np.linspace(-0.75, 0.75, 9)
    control = boost_control(mstop=8, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=5, df=4, degree=3, differences=2) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_gaussian_predict_runner(
        x,
        z,
        y,
        x_new=x_new,
        z_new=z_new,
        mstop=8,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new, "z": z_new}),
        r_result["pred"],
        atol=1e-11,
        rtol=1e-11,
    )


def test_python_gamboost_matches_r_for_factor_bols_newdata_prediction(
    r_gamboost_bols_factor_predict_runner,
):
    x = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    x_new = np.array(["c", "a", "b", "c", "a"], dtype=object)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bols_factor_predict_runner(
        x,
        y,
        x_new=x_new,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_python_glmboost_matches_r_for_bare_factor_newdata_prediction(
    r_factor_predict_runner,
):
    x = np.array(["a", "a", "b", "b", "c", "c", "a", "b", "c"], dtype=object)
    y = np.array([0.0, 0.2, 1.0, 1.1, 2.0, 2.2, 0.1, 0.9, 2.1], dtype=np.float64)
    x_new = np.array(["c", "a", "b", "c", "a"], dtype=object)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_factor_predict_runner(
        x,
        y,
        x_new=x_new,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )
