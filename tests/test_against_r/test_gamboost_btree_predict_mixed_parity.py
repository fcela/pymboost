from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_btree_newdata_on_training_support(
    r_gamboost_btree_predict_runner,
):
    x = np.linspace(-1.0, 1.0, 20)
    y = (x > 0.0).astype(np.float64)
    x_new = x[[1, 3, 5, 7, 9, 11, 13, 15, 17]]
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ btree(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_btree_predict_runner(
        x,
        y,
        x_new=x_new,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_python_gamboost_matches_r_for_mixed_btree_bols_risk_and_selection(
    r_gamboost_btree_mixed_runner,
):
    x = np.linspace(-1.0, 1.0, 40)
    z = np.linspace(-1.0, 1.0, 40)
    y = (x > 0.0).astype(np.float64) + 0.2 * z
    control = boost_control(mstop=6, nu=0.1)

    py_model = gamboost(
        "y ~ btree(x) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_btree_mixed_runner(
        x,
        z,
        y,
        mstop=6,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(py_model.path.selected, dtype=np.int64) + 1, r_result["selected"])
