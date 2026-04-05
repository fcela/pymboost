from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, confint, gamboost


def test_python_confint_bootstrap_matches_r_for_fitted_values(
    r_confint_bootstrap_runner,
):
    x = np.linspace(-1.0, 1.0, 20)
    y = 1.0 + 0.5 * x
    control = boost_control(mstop=5, nu=0.1)
    boot_weights = np.array(
        [
            [0, 2, 0, 1, 1, 0],
            [1, 0, 2, 0, 1, 1],
            [2, 1, 0, 1, 0, 1],
            [0, 1, 1, 2, 0, 0],
            [1, 0, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 2],
            [0, 1, 2, 0, 0, 1],
            [1, 1, 0, 2, 0, 0],
            [2, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 2, 1],
            [1, 0, 1, 0, 1, 2],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 2, 1],
            [1, 0, 2, 0, 0, 1],
            [0, 1, 0, 2, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 2, 0],
            [1, 1, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_result = confint(
        py_model,
        method="bootstrap",
        B=boot_weights.shape[1],
        bootstrap_weights=boot_weights,
        level=0.95,
    ).to_pandas()
    r_result = r_confint_bootstrap_runner(
        x,
        y,
        boot_weights=boot_weights,
        level=0.95,
        mode="fitted",
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result["estimate"], r_result["estimate"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["lower"], r_result["lower"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["upper"], r_result["upper"], atol=1e-12, rtol=1e-12)


def test_python_confint_bootstrap_matches_r_for_partial_linear_effect(
    r_confint_bootstrap_runner,
):
    x = np.linspace(-1.0, 1.0, 20)
    y = 1.0 + 0.5 * x
    x_new = np.linspace(-0.9, 0.9, 9)
    control = boost_control(mstop=5, nu=0.1)
    boot_weights = np.array(
        [
            [0, 2, 0, 1, 1, 0],
            [1, 0, 2, 0, 1, 1],
            [2, 1, 0, 1, 0, 1],
            [0, 1, 1, 2, 0, 0],
            [1, 0, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 2],
            [0, 1, 2, 0, 0, 1],
            [1, 1, 0, 2, 0, 0],
            [2, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 2, 1],
            [1, 0, 1, 0, 1, 2],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 2, 1],
            [1, 0, 2, 0, 0, 1],
            [0, 1, 0, 2, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 2, 0],
            [1, 1, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_result = confint(
        py_model,
        which=0,
        method="bootstrap",
        B=boot_weights.shape[1],
        bootstrap_weights=boot_weights,
        newdata={"x": x_new},
        level=0.95,
    ).to_pandas()
    r_result = r_confint_bootstrap_runner(
        x,
        y,
        boot_weights=boot_weights,
        x_new=x_new,
        level=0.95,
        mode="partial",
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result["estimate"], r_result["estimate"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["lower"], r_result["lower"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["upper"], r_result["upper"], atol=1e-12, rtol=1e-12)


def test_python_confint_bootstrap_matches_r_for_partial_spline_effect(
    r_confint_bootstrap_bbs_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    y = np.sin(2.0 * np.pi * x)
    x_new = np.linspace(0.05, 0.95, 9)
    control = boost_control(mstop=5, nu=0.1)
    boot_weights = np.array(
        [
            [0, 2, 0, 1, 1, 0],
            [1, 0, 2, 0, 1, 1],
            [2, 1, 0, 1, 0, 1],
            [0, 1, 1, 2, 0, 0],
            [1, 0, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 2],
            [0, 1, 2, 0, 0, 1],
            [1, 1, 0, 2, 0, 0],
            [2, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 2, 1],
            [1, 0, 1, 0, 1, 2],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 2, 1],
            [1, 0, 2, 0, 0, 1],
            [0, 1, 0, 2, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 2, 0],
            [1, 1, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )

    py_model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_result = confint(
        py_model,
        which=0,
        method="bootstrap",
        B=boot_weights.shape[1],
        bootstrap_weights=boot_weights,
        newdata={"x": x_new},
        level=0.95,
    ).to_pandas()
    r_result = r_confint_bootstrap_bbs_runner(
        x,
        y,
        boot_weights=boot_weights,
        x_new=x_new,
        level=0.95,
        mode="partial",
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result["estimate"], r_result["estimate"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_result["lower"], r_result["lower"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_result["upper"], r_result["upper"], atol=5e-5, rtol=5e-5)


def test_python_confint_bootstrap_with_inner_mstop_matches_r_for_partial_linear_effect(
    r_confint_bootstrap_bmstop_runner,
):
    x = np.linspace(-1.0, 1.0, 20)
    y = 1.0 + 0.5 * x + 0.1 * np.sin(np.pi * x)
    x_new = np.linspace(-0.9, 0.9, 9)
    control = boost_control(mstop=5, nu=0.1)
    boot_weights = np.array(
        [
            [0, 2, 0, 1],
            [1, 0, 2, 0],
            [2, 1, 0, 1],
            [0, 1, 1, 2],
            [1, 0, 0, 1],
            [0, 2, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 2, 0],
            [1, 1, 0, 2],
            [2, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 2, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 2, 0],
            [0, 1, 0, 2],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    inner_boot = np.array(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
        ],
        dtype=np.float64,
    )

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_result = confint(
        py_model,
        which=0,
        method="bootstrap",
        B=boot_weights.shape[1],
        B_mstop=inner_boot.shape[1],
        bootstrap_weights=boot_weights,
        inner_bootstrap_weights=inner_boot,
        newdata={"x": x_new},
        level=0.95,
    ).to_pandas()
    r_result = r_confint_bootstrap_bmstop_runner(
        x,
        y,
        boot_weights=boot_weights,
        inner_boot=inner_boot,
        x_new=x_new,
        level=0.95,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result["lower"], r_result["lower"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["upper"], r_result["upper"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result["std_error"], r_result["std_error"], atol=1e-12, rtol=1e-12)
