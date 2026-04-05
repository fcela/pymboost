from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_gaussian_bbs_by_term(
    r_gamboost_bbs_by_runner,
):
    x = np.linspace(0.0, 1.0, 20)
    z = np.linspace(-1.0, 1.0, 20)
    y = np.sin(2.0 * np.pi * x) * z
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, df=4, knots=5, by=z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_by_runner(
        x,
        z,
        y,
        knots=5,
        df_value=4,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-9, rtol=1e-9)
