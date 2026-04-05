from __future__ import annotations

import numpy as np

from mboost import Huber, boost_control, glmboost


def test_python_glmboost_matches_r_for_single_huber_linear_term(r_reference_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = x * x
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Huber(), control=control)
    r_result = r_reference_runner(x, y, family="huber", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=3e-4, rtol=3e-4)
