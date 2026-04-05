from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_gaussian_bbs_plus_bols(
    r_gamboost_gaussian_runner,
):
    x = np.linspace(0.0, 1.0, 50)
    z = np.linspace(-1.0, 1.0, 50)
    y = np.sin(2.0 * np.pi * x) + 0.5 * z
    control = boost_control(mstop=8, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=5, df=4, degree=3, differences=2) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_gaussian_runner(
        x,
        z,
        y,
        mstop=8,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-11, rtol=1e-11)
    np.testing.assert_array_equal(np.asarray(py_model.path.selected, dtype=np.int64) + 1, r_result["selected"])
