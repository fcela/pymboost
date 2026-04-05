from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_monotone_factor_increasing(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float64)
    x_new = np.array(["c", "a", "b", "c"], dtype=object)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        x_new=x_new,
        constraint="increasing",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=3e-4,
        rtol=3e-4,
    )
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=7e-4, rtol=7e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6


def test_python_gamboost_matches_r_for_monotone_factor_decreasing(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([2.2, 2.1, 2.0, 1.2, 1.1, 1.0, 0.2, 0.1, 0.0], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="decreasing", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        constraint="decreasing",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=7e-4, rtol=7e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6


def test_python_gamboost_matches_r_for_positive_factor_constraint(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="positive", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        constraint="positive",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=7e-4, rtol=7e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6


def test_python_gamboost_matches_r_for_negative_factor_constraint(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([-0.2, -0.3, -0.4, -1.0, -1.1, -1.2, -2.0, -2.1, -2.2], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="negative", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        constraint="negative",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=7e-4, rtol=7e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6


def test_python_gamboost_matches_r_for_convex_factor_constraint(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, dtype=object)
    y = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 2.2, 2.3, 2.4], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="convex", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        constraint="convex",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=5e-4, rtol=5e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6


def test_python_gamboost_matches_r_for_concave_factor_constraint(
    r_gamboost_bmono_factor_runner,
):
    x = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, dtype=object)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="concave", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bmono_factor_runner(
        x,
        y,
        constraint="concave",
        lambda_value=1.0,
        mstop=5,
        nu=0.1,
    )
    py_coef = next(iter(py_model.coefficients_.values()))

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=7e-5, rtol=7e-5)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=7e-5, rtol=7e-5)
    np.testing.assert_allclose(py_coef, r_result["coef"], atol=6e-4, rtol=6e-4)
    assert abs(py_model.offset_ - r_result["offset"]) <= 5e-6
