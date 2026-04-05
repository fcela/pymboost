from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, coef, fitted, glmboost, risk, selected


def test_top_level_extractors_return_model_state():
    x = np.linspace(-1.0, 1.0, 20)
    y = 2.0 * x + 0.5
    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    np.testing.assert_allclose(fitted(model), model.fitted_)
    np.testing.assert_allclose(risk(model), model.risk_)
    assert selected(model) == model.selected
    assert coef(model).keys() == model.coefficients_.keys()
    for key in coef(model):
        np.testing.assert_allclose(coef(model)[key], model.coefficients_[key])
