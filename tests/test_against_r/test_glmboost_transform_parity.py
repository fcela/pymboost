from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, glmboost


def test_python_glmboost_matches_r_for_log_transform_term(
    r_transform_reference_runner,
):
    x = np.linspace(1.0, 5.0, 40)
    y = 0.8 * np.log(x) + 0.25
    x_new = np.array([1.25, 2.5, 4.75], dtype=np.float64)
    control = boost_control(mstop=6, nu=0.1)

    py_model = glmboost(
        "y ~ np.log(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_transform_reference_runner(
        x,
        y,
        x_new=x_new,
        mstop=6,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )
