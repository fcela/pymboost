from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, glmboost


def test_python_glmboost_matches_r_for_numeric_factor_interaction_term(
    r_factor_interaction_reference_runner,
):
    x = np.linspace(-1.0, 1.0, 18)
    g = np.array(["a", "b", "c"] * 6, dtype=object)
    slopes = {"a": -0.5, "b": 0.75, "c": 1.5}
    y = np.array([slopes[level] * value for value, level in zip(x, g)], dtype=np.float64)
    x_new = np.array([-0.8, -0.2, 0.4, 0.9], dtype=np.float64)
    g_new = np.array(["c", "a", "b", "c"], dtype=object)
    control = boost_control(mstop=6, nu=0.1)

    py_model = glmboost(
        "y ~ x:g",
        data={"x": x, "g": g, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_factor_interaction_reference_runner(
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
