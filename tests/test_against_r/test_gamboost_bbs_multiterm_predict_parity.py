from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_multiterm_default_bbs_newdata_prediction(
    r_gamboost_bbs_multiterm_predict_runner,
):
    x1 = np.linspace(0.0, 1.0, 50)
    x2 = np.linspace(-1.0, 1.0, 50)
    y = np.sin(2.0 * np.pi * x1) + 0.3 * x2**2
    x1_new = np.linspace(0.1, 0.9, 9)
    x2_new = np.linspace(-0.8, 0.8, 9)
    control = boost_control(mstop=8, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x1) + bbs(x2)",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_bbs_multiterm_predict_runner(
        x1,
        x2,
        y,
        x1_new=x1_new,
        x2_new=x2_new,
        mstop=8,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x1": x1_new, "x2": x2_new}),
        r_result["pred"],
        atol=1e-11,
        rtol=1e-11,
    )
