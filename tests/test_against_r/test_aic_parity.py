from __future__ import annotations

import numpy as np
import pytest

from mboost import AIC, Gaussian, boost_control, gamboost


def test_corrected_aic_matches_r_for_gaussian_bbs_gamboost(r_gamboost_aic_runner):
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    control = boost_control(mstop=20, nu=0.1)

    py_model = gamboost(
        "y ~ bbs(x, knots=5, df=4, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_aic = AIC(py_model, method="corrected")
    r_aic = r_gamboost_aic_runner(x, y, mode="bbs", mstop=20, nu=0.1)

    assert float(py_aic) == pytest.approx(r_aic["value"])
    assert py_aic.mstop == r_aic["mstop"]
    assert py_aic.df == pytest.approx(r_aic["df"])
    np.testing.assert_allclose(py_aic.aic_path, r_aic["aic_path"], atol=1e-12, rtol=1e-12)


def test_corrected_aic_matches_r_for_explicit_gaussian_bols_gamboost(r_gamboost_aic_runner):
    x = np.linspace(-1.0, 1.0, 30)
    y = 1.0 + 2.0 * x
    control = boost_control(mstop=15, nu=0.1)

    py_model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    py_aic = AIC(py_model, method="corrected")
    r_aic = r_gamboost_aic_runner(x, y, mode="bols", mstop=15, nu=0.1)

    assert float(py_aic) == pytest.approx(r_aic["value"])
    assert py_aic.mstop == r_aic["mstop"]
    assert py_aic.df == pytest.approx(r_aic["df"])
    np.testing.assert_allclose(py_aic.aic_path, r_aic["aic_path"], atol=1e-12, rtol=1e-12)
