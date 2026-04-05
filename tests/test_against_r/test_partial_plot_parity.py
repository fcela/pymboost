from __future__ import annotations

import numpy as np
import polars as pl

from mboost import Gaussian, boost_control, gamboost, partial_plot_data


def test_python_partial_plot_data_matches_r_which_predictions_for_bodyfat_bbs_terms(
    r_gamboost_three_bbs_partial_runner,
):
    bodyfat = pl.read_csv("data/bodyfat.csv").to_pandas()
    control = boost_control(mstop=100, nu=0.1)

    py_model = gamboost(
        "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
        data=bodyfat,
        family=Gaussian(),
        control=control,
    )
    py_partial = partial_plot_data(py_model, grid_size=80)

    for feature in ["hipcirc", "kneebreadth", "anthro3a"]:
        grid = np.linspace(float(bodyfat[feature].min()), float(bodyfat[feature].max()), 80)
        r_result = r_gamboost_three_bbs_partial_runner(
            bodyfat["hipcirc"].to_numpy(),
            bodyfat["kneebreadth"].to_numpy(),
            bodyfat["anthro3a"].to_numpy(),
            bodyfat["DEXfat"].to_numpy(),
            grid=grid,
            which_name={"hipcirc": "x1", "kneebreadth": "x2", "anthro3a": "x3"}[feature],
            mstop=100,
            nu=0.1,
        )
        py_term = py_partial[py_partial["term"] == f"bbs({feature})"]

        np.testing.assert_allclose(py_term["x"].to_numpy(), grid, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            py_term["effect"].to_numpy(),
            r_result["pred"],
            atol=1e-7,
            rtol=1e-7,
        )


def test_python_partial_plot_data_observed_support_matches_r_for_bodyfat_bbs_terms(
    r_gamboost_three_bbs_partial_runner,
):
    bodyfat = pl.read_csv("data/bodyfat.csv").to_pandas()
    control = boost_control(mstop=100, nu=0.1)

    py_model = gamboost(
        "DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
        data=bodyfat,
        family=Gaussian(),
        control=control,
    )
    py_partial = partial_plot_data(py_model, grid_size=None)

    for feature in ["hipcirc", "kneebreadth", "anthro3a"]:
        observed = np.sort(bodyfat[feature].to_numpy())
        r_result = r_gamboost_three_bbs_partial_runner(
            bodyfat["hipcirc"].to_numpy(),
            bodyfat["kneebreadth"].to_numpy(),
            bodyfat["anthro3a"].to_numpy(),
            bodyfat["DEXfat"].to_numpy(),
            grid=observed,
            which_name={"hipcirc": "x1", "kneebreadth": "x2", "anthro3a": "x3"}[feature],
            mstop=100,
            nu=0.1,
        )
        py_term = py_partial[py_partial["term"] == f"bbs({feature})"]

        np.testing.assert_allclose(py_term["x"].to_numpy(), observed, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            py_term["effect"].to_numpy(),
            r_result["pred"],
            atol=1e-7,
            rtol=1e-7,
        )
