from __future__ import annotations

import numpy as np

from mboost import boost_control, glmboost


def test_python_glmboost_matches_r_for_single_gaussian_linear_term(r_reference_runner):
    x = np.linspace(-1.0, 1.0, 12)
    y = 1.5 * x - 0.25
    control = boost_control(mstop=4, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, control=control)
    r_result = r_reference_runner(x, y, family="gaussian", mstop=4, nu=0.1)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
