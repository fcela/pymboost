from __future__ import annotations

import numpy as np

from mboost import Laplace, boost_control, glmboost


def test_python_glmboost_matches_r_for_single_laplace_linear_term(r_reference_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Laplace(), control=control)
    r_result = r_reference_runner(x, y, family="laplace", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=4e-3, rtol=1e-3)
