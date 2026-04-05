from __future__ import annotations

import numpy as np

from mboost import Binomial, Expectile, GammaReg, Huber, Laplace, Poisson, Quantile, boost_control, glmboost


def test_python_glmboost_matches_r_for_binomial_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(-1.5, 1.5, 18)
    y = (x > 0.0).astype(np.float64)
    x_new = np.linspace(-1.0, 1.0, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Binomial(), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="binomial", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=1e-12, rtol=1e-12)


def test_python_glmboost_matches_r_for_poisson_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(0.0, 1.5, 18)
    y = np.round(np.exp(0.6 * x)).astype(np.float64)
    x_new = np.linspace(0.1, 1.4, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Poisson(), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="poisson", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=1e-12, rtol=1e-12)


def test_python_glmboost_matches_r_for_laplace_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
    x_new = np.linspace(-0.8, 0.8, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Laplace(), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="laplace", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=3e-3, rtol=3e-3)


def test_python_glmboost_matches_r_for_quantile_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = x * x
    x_new = np.linspace(-0.9, 0.9, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Quantile(tau=0.25), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="quantile", mstop=5, nu=0.1, tau=0.25)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=1e-12, rtol=1e-12)


def test_python_glmboost_matches_r_for_expectile_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = x * x
    x_new = np.linspace(-0.9, 0.9, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Expectile(tau=0.25), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="expectile", mstop=5, nu=0.1, tau=0.25)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=1e-12, rtol=1e-12)


def test_python_glmboost_matches_r_for_huber_newdata_prediction(r_reference_predict_runner):
    x = np.linspace(-1.0, 1.0, 16)
    y = x * x
    x_new = np.linspace(-0.9, 0.9, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Huber(), control=control)
    r_result = r_reference_predict_runner(x, y, x_new=x_new, family="huber", mstop=5, nu=0.1)

    np.testing.assert_allclose(py_model.predict(newdata={"x": x_new}), r_result["pred"], atol=3e-4, rtol=3e-4)


def test_python_glmboost_matches_r_for_binomial_response_prediction(r_reference_predict_runner):
    x = np.linspace(-1.5, 1.5, 18)
    y = (x > 0.0).astype(np.float64)
    x_new = np.linspace(-1.0, 1.0, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Binomial(), control=control)
    r_result = r_reference_predict_runner(
        x,
        y,
        x_new=x_new,
        family="binomial",
        mstop=5,
        nu=0.1,
        pred_type="response",
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}, type="response"),
        r_result["pred"],
        atol=5e-4,
        rtol=5e-4,
    )


def test_python_glmboost_matches_r_for_poisson_response_prediction(r_reference_predict_runner):
    x = np.linspace(0.0, 1.5, 18)
    y = np.round(np.exp(0.6 * x)).astype(np.float64)
    x_new = np.linspace(0.1, 1.4, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=Poisson(), control=control)
    r_result = r_reference_predict_runner(
        x,
        y,
        x_new=x_new,
        family="poisson",
        mstop=5,
        nu=0.1,
        pred_type="response",
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}, type="response"),
        r_result["pred"],
        atol=5e-4,
        rtol=5e-4,
    )


def test_python_glmboost_matches_r_for_gamma_response_prediction(r_reference_predict_runner):
    x = np.linspace(0.2, 1.5, 18)
    y = np.exp(0.3 + 0.5 * x)
    x_new = np.linspace(0.25, 1.45, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = glmboost("y ~ x", data={"x": x, "y": y}, family=GammaReg(), control=control)
    r_result = r_reference_predict_runner(
        x,
        y,
        x_new=x_new,
        family="gamma",
        mstop=5,
        nu=0.1,
        pred_type="response",
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}, type="response"),
        r_result["pred"],
        atol=5e-4,
        rtol=5e-4,
    )
