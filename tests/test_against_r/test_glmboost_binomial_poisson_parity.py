from __future__ import annotations

import numpy as np

from mboost import Binomial, Poisson, boost_control, glmboost


def test_python_glmboost_matches_r_for_single_binomial_linear_term(r_reference_runner):
    x = np.linspace(-2.0, 2.0, 16)
    y = (x > 0.0).astype(np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Binomial(), control=control)
    r_result = r_reference_runner(x, y, family="binomial", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-10, rtol=1e-10)


def test_python_glmboost_matches_r_for_single_poisson_linear_term(r_reference_runner):
    x = np.linspace(0.0, 1.2, 16)
    y = np.round(np.exp(0.5 * x)).astype(np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Poisson(), control=control)
    r_result = r_reference_runner(x, y, family="poisson", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-10, rtol=1e-10)
