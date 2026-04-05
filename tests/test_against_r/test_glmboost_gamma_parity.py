from __future__ import annotations

import numpy as np

from mboost import GammaReg, boost_control, glmboost


def test_glmboost_gamma_matches_r_reference(r_reference_runner):
    x = np.linspace(0.1, 2.0, 25)
    y = np.exp(0.3 + 0.5 * x)

    reference = r_reference_runner(
        x,
        y,
        family="gamma",
        mstop=6,
        nu=0.1,
    )

    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=GammaReg(),
        control=boost_control(mstop=6, nu=0.1),
    )

    np.testing.assert_allclose(model.fitted_, reference["fitted"], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(model.risk_, reference["risk"], rtol=3e-4, atol=5e-3)
