from __future__ import annotations

import numpy as np

from mboost import Binomial, Gaussian, Poisson, boost_control, gamboost


def test_python_gamboost_matches_r_for_binomial_monotone_increasing_spline(
    r_gamboost_bmono_family_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    eta = -2.0 + 5.0 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Binomial(),
        control=control,
    )
    r_result = r_gamboost_bmono_family_runner(
        x,
        y,
        family="binomial",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=8e-5, rtol=8e-5)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=8e-5, rtol=8e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=2e-4, rtol=2e-4)


def test_python_gamboost_matches_r_for_poisson_monotone_increasing_spline(
    r_gamboost_bmono_family_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    mu = np.exp(0.2 + 1.2 * x)
    y = np.round(mu).astype(np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Poisson(),
        control=control,
    )
    r_result = r_gamboost_bmono_family_runner(
        x,
        y,
        family="poisson",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=3e-4, rtol=3e-4)
