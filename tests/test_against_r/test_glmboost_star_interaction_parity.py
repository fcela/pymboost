from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, glmboost


def test_python_glmboost_matches_r_for_star_interaction_formula(
    r_star_interaction_reference_runner,
):
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.0, -1.0, 40)
    y = 0.3 * x1 - 0.2 * x2 + 1.5 * x1 * x2 + 0.1
    x1_new = np.linspace(-0.8, 0.8, 7)
    x2_new = np.linspace(0.9, -0.9, 7)
    control = boost_control(mstop=8, nu=0.1)

    py_model = glmboost(
        "y ~ x1 * x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_star_interaction_reference_runner(
        x1,
        x2,
        y,
        x1_new=x1_new,
        x2_new=x2_new,
        mstop=8,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_model.predict(newdata={"x1": x1_new, "x2": x2_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )
