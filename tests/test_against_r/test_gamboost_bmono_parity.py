from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_monotone_increasing_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = x**2
    x_new = np.linspace(0.05, 0.95, 9)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        x_new=x_new,
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
    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=5e-5,
        rtol=5e-5,
    )
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=5e-4, rtol=5e-4)


def test_python_gamboost_matches_r_for_monotone_increasing_spline_with_boundary_constraints(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = x**2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2, boundary_constraints=True)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        boundary_constraints=True,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=5e-4, rtol=5e-4)


def test_python_gamboost_matches_r_for_monotone_decreasing_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = (1.0 - x) ** 2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="decreasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="decreasing",
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


def test_python_gamboost_matches_r_for_convex_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = x**2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="convex", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="convex",
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


def test_python_gamboost_matches_r_for_concave_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = 1.0 - (1.0 - x) ** 2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="concave", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="concave",
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


def test_python_gamboost_matches_r_for_positive_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = 0.25 + x**2
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="positive", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="positive",
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


def test_python_gamboost_matches_r_for_negative_spline(
    r_gamboost_bmono_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = -(0.25 + x**2)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="negative", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_runner(
        x,
        y,
        constraint="negative",
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
