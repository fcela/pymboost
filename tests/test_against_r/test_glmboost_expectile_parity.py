from __future__ import annotations

import numpy as np

from mboost import Expectile, boost_control, glmboost


def test_python_glmboost_matches_r_for_single_expectile_linear_term(r_reference_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = x * x
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Expectile(tau=0.25), control=control)
    r_result = r_reference_runner(x, y, family="expectile", mstop=5, nu=0.1, tau=0.25)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
