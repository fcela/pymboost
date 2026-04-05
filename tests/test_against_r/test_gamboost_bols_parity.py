from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_explicit_gaussian_bols(
    r_gamboost_bols_runner,
):
    z = np.linspace(-1.0, 1.0, 30)
    y = 1.0 + 2.0 * z
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bols(z)",
        data={"z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bols_runner(
        z,
        y,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.coefficients_["bols(z)"], r_result["coef"], atol=1e-12, rtol=1e-12)
    assert py_model.offset_ == r_result["offset"]


def test_python_gamboost_matches_r_for_df_penalized_gaussian_bols(
    r_gamboost_bols_runner,
):
    z = np.linspace(-1.0, 1.0, 30)
    y = 1.0 + 2.0 * z
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bols(z, df=1)",
        data={"z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bols_runner(
        z,
        y,
        mstop=5,
        nu=0.1,
        df_value=1.0,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(py_model.coefficients_["bols(z, df=1)"], r_result["coef"], atol=1e-10, rtol=1e-10)
