from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, gamboost


def test_python_gamboost_matches_r_for_btree_stump_term(
    r_gamboost_btree_runner,
):
    x = np.linspace(-1.0, 1.0, 20)
    y = (x > 0.0).astype(np.float64)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        "y ~ btree(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_gamboost_btree_runner(
        x,
        y,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_model.fitted_, r_result["fitted"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_model.risk_, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(py_model.path.selected, dtype=np.int64) + 1, r_result["selected"])
