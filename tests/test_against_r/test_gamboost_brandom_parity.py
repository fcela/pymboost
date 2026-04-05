from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_brandom_factor_term(
    r_gamboost_brandom_runner,
):
    x = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    y = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ brandom(x, df=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_brandom_runner(
        x,
        y,
        df_value=2,
        lambda_value=None,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.coefficients_["brandom(x, df=2)"], r_result["coef"], atol=1e-12, rtol=1e-12)
    assert py_model.offset_ == r_result["offset"]
