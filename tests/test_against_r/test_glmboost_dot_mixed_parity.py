from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, glmboost


def test_python_glmboost_matches_r_for_dot_expansion_with_mixed_numeric_and_factor_terms(
    r_dot_mixed_reference_runner,
):
    x = np.linspace(-1.0, 1.0, 18)
    z = np.cos(2.0 * x)
    g = np.array(["a", "b", "c"] * 6, dtype=object)
    offsets = {"a": -0.3, "b": 0.1, "c": 0.4}
    y = np.array([0.8 * xv - 0.25 * zv + offsets[level] for xv, zv, level in zip(x, z, g)], dtype=np.float64)

    x_new = np.array([-0.8, -0.2, 0.4, 0.9], dtype=np.float64)
    z_new = np.cos(2.0 * x_new)
    g_new = np.array(["c", "a", "b", "c"], dtype=object)
    control = boost_control(mstop=6, nu=0.1)

    py_model = glmboost(
        "y ~ .",
        data={"x": x, "z": z, "g": g, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_dot_mixed_reference_runner(
        x,
        z,
        g,
        y,
        x_new=x_new,
        z_new=z_new,
        g_new=g_new,
        mstop=6,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new, "z": z_new, "g": g_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )
