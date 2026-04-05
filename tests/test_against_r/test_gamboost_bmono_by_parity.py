from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_monotone_spline_with_by_modifier(
    r_gamboost_bmono_by_runner,
):
    x = np.linspace(0.0, 1.0, 25)
    z = np.linspace(0.5, 1.5, 25)
    y = z * x**2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", by=z, knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_by_runner(
        x,
        z,
        y,
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=5e-4, rtol=5e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6
