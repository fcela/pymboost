from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_factor_bols(
    r_gamboost_bols_factor_runner,
):
    x = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bols_factor_runner(
        x,
        y,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.coefficients_["bols(x)"], r_result["coef"], atol=1e-12, rtol=1e-12)
    assert py_model.offset_ == r_result["offset"]
