from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, glmboost


def test_python_glmboost_matches_r_for_transformed_numeric_factor_star_formula(
    r_transform_factor_star_reference_runner,
):
    x = np.linspace(1.0, 5.0, 18)
    g = np.array(["a", "b", "c"] * 6, dtype=object)
    intercepts = {"a": -0.2, "b": 0.0, "c": 0.25}
    slopes = {"a": 0.4, "b": 0.8, "c": 1.2}
    y = np.array([intercepts[level] + slopes[level] * np.log(value) for value, level in zip(x, g)], dtype=np.float64)
    x_new = np.array([1.25, 2.5, 4.75, 3.0], dtype=np.float64)
    g_new = np.array(["c", "a", "b", "c"], dtype=object)
    control = boost_control(mstop=6, nu=0.1)

    py_model = glmboost(
        "y ~ np.log(x) * g",
        data={"x": x, "g": g, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_transform_factor_star_reference_runner(
        x,
        g,
        y,
        x_new=x_new,
        g_new=g_new,
        mstop=6,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new, "g": g_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )
