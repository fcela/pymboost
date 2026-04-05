from __future__ import annotations

import numpy as np

from mboost import Gaussian, bols, boost_control, gamboost, varimp


def test_python_varimp_matches_r_for_gaussian_multiterm_model(r_varimp_runner):
    x = np.linspace(0.0, 1.0, 50)
    z = np.linspace(-1.0, 1.0, 50)
    y = np.sin(2.0 * np.pi * x) + 0.5 * z
    control = boost_control(mstop=8, nu=0.1)

    py_model = gamboost(
        "y ~ bmono(x, constraint='increasing', knots=5, lambda=1.0, degree=3, differences=2) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    r_result = r_varimp_runner(x, z, y, mstop=8, nu=0.1)

    py_blearner = varimp(py_model, percent=False, type="blearner").to_pandas().sort_values("label")
    py_variable = varimp(py_model, percent=False, type="variable").to_pandas().sort_values("label")

    assert list(py_blearner["label"]) == sorted(r_result["baselearner_names"])
    assert list(py_variable["label"]) == sorted(r_result["variable_names"])
    np.testing.assert_allclose(
        py_blearner["reduction"].to_numpy(),
        np.asarray([value for _, value in sorted(zip(r_result["baselearner_names"], r_result["baselearner"]))]),
        atol=1e-7,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        py_variable["reduction"].to_numpy(),
        np.asarray([value for _, value in sorted(zip(r_result["variable_names"], r_result["variable"]))]),
        atol=1e-7,
        rtol=1e-7,
    )
