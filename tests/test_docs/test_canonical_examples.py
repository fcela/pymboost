from __future__ import annotations

import numpy as np

from examples.additive_models import run_example as run_additive
from examples.blackboost_demo import run_example as run_blackboost
from examples.bodyfat_examples import (
    load_bodyfat,
    run_additive_bodyfat_example,
    run_bodyfat_model_selection_example,
    run_bodyfat_quantile_example,
    run_linear_bodyfat_example,
)
from examples.cars_cvrisk import run_example as run_cars_cvrisk
from examples.cars_gamboost import run_example as run_cars
from examples.chart_gallery import run_example as run_chart_gallery
from examples.getting_started import run_example as run_getting_started
from examples.monotone_splines import run_example as run_monotone


def test_getting_started_example_returns_stable_surface() -> None:
    result = run_getting_started()
    assert "Model-based boosting fit" in result["summary"]
    assert result["selected_head"]
    assert np.isfinite(result["final_risk"])


def test_cars_example_matches_known_aic_result() -> None:
    result = run_cars()
    assert result["first_selected"] == "speed"
    assert result["aic_mstop"] == 26
    assert np.isclose(result["aic_value"], 6.575865764262378, atol=1e-12)
    assert np.isclose(result["aic_df"], 3.905132697691835, atol=1e-12)


def test_cars_cvrisk_example_matches_known_surface() -> None:
    result = run_cars_cvrisk()
    assert result["aic_mstop"] == 26
    assert result["cv_best_mstop"] > 0
    assert result["cv_risk_head"].shape == (5,)
    assert np.all(np.isfinite(result["cv_risk_head"]))


def test_additive_example_produces_plotting_objects() -> None:
    result = run_additive()
    assert result["selected_head"]
    assert result["risk_head"].shape == (5,)
    assert hasattr(result["partial_chart"], "to_dict")
    assert hasattr(result["varimp_chart"], "to_dict")


def test_monotone_example_runs() -> None:
    result = run_monotone()
    assert result["selected_head"]
    assert np.isfinite(result["final_risk"])
    assert hasattr(result["partial_chart"], "to_dict")


def test_blackboost_example_runs() -> None:
    result = run_blackboost()
    assert result["formula"].startswith("y ~ btree(")
    assert result["selected_head"]
    assert np.isfinite(result["final_risk"])


def test_chart_gallery_exercises_all_plot_types() -> None:
    result = run_chart_gallery()
    assert hasattr(result["numeric_partial_chart"], "to_dict")
    assert hasattr(result["categorical_partial_chart"], "to_dict")
    assert hasattr(result["surface_chart"], "to_dict")
    assert hasattr(result["sensitivity_chart"], "to_dict")
    assert hasattr(result["variable_importance_chart"], "to_dict")
    assert hasattr(result["blearner_importance_chart"], "to_dict")
    assert hasattr(result["cvrisk_chart"], "to_dict")
    assert hasattr(result["aic_chart"], "to_dict")
    assert not result["variable_importance"].empty
    assert not result["blearner_importance"].empty


def test_bodyfat_dataset_is_available() -> None:
    data = load_bodyfat()
    assert data.shape == (71, 10)
    assert "DEXfat" in data.columns
    assert "hipcirc" in data.columns


def test_bodyfat_linear_example_runs() -> None:
    result = run_linear_bodyfat_example()
    assert "hipcirc" in result["formula"]
    assert np.isfinite(float(result["aic"]))
    assert hasattr(result["plot"], "to_dict")
    assert hasattr(result["importance_plot"], "to_dict")


def test_bodyfat_additive_example_runs() -> None:
    result = run_additive_bodyfat_example()
    assert "bbs(hipcirc" in result["formula"]
    assert hasattr(result["partial_plot"], "to_dict")
    assert hasattr(result["aic_plot"], "to_dict")


def test_bodyfat_model_selection_example_runs() -> None:
    result = run_bodyfat_model_selection_example()
    assert result["cvrisk"].best_mstop > 0
    assert result["selected_terms"]
    assert hasattr(result["cv_plot"], "to_dict")
    assert hasattr(result["partial_plot"], "to_dict")


def test_bodyfat_quantile_example_runs() -> None:
    result = run_bodyfat_quantile_example()
    assert result["hipcirc_sorted"].shape == result["dexfat_sorted"].shape
    assert result["fitted_sorted"].shape == result["dexfat_sorted"].shape
