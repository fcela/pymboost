from __future__ import annotations

import warnings

import numpy as np

from mboost import AIC, Gaussian, boost_control, confint, cvrisk, gamboost, glmboost, plot, partial_plot_data, varimp


def test_partial_plot_data_returns_numeric_effect_grid_for_spline_model():
    x = np.linspace(0.0, 1.0, 30)
    y = np.sin(2.0 * np.pi * x)
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    data = partial_plot_data(model)

    assert set(data.columns) >= {"term", "feature", "x", "effect", "kind"}
    assert set(data["kind"].unique()) == {"numeric"}
    assert data["term"].nunique() == 1


def test_plot_uses_level_segments_for_categorical_partial_effects():
    x = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 0.9, 1.9, 2.9], dtype=np.float64)
    model = gamboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    chart = plot(model)
    spec = chart.to_dict()

    assert spec["spec"]["layer"][0]["mark"]["type"] == "rule"


def test_partial_plot_data_returns_surface_for_two_feature_tree_term():
    x = np.linspace(-1.0, 1.0, 40)
    z = np.cos(3.0 * x)
    y = x + z
    model = gamboost(
        "y ~ btree(x, z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    data = partial_plot_data(model, grid_size=36)

    assert set(data["kind"].unique()) == {"surface"}
    assert {"x", "y", "effect", "feature_x", "feature_y"} <= set(data.columns)


def test_partial_plot_data_returns_one_factor_at_a_time_sensitivity_for_multi_feature_tree_term():
    x = np.linspace(-1.0, 1.0, 36)
    z = np.cos(2.0 * x)
    w = np.sin(3.0 * x)
    y = x - z + 0.5 * w
    model = gamboost(
        "y ~ btree(x, z, w)",
        data={"x": x, "z": z, "w": w, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    data = partial_plot_data(model, grid_size=12)

    assert set(data["kind"].unique()) == {"sensitivity_numeric"}
    assert set(data["feature"].unique()) == {"x", "z", "w"}


def test_plot_smoke_covers_model_cvrisk_aic_and_varimp():
    x = np.linspace(0.0, 1.0, 30)
    z = np.linspace(-1.0, 1.0, 30)
    y = np.sin(2.0 * np.pi * x) + 0.25 * z
    control = boost_control(mstop=6, nu=0.1)
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
    )
    cv_result = cvrisk(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2) + bols(z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
        folds=3,
    )
    aic_result = AIC(model, method="corrected")
    vi_result = varimp(model)

    model_chart = plot(model)
    cv_chart = plot(cv_result)
    aic_chart = plot(aic_result)
    vi_chart = plot(vi_result)

    assert model_chart.to_dict()["facet"]["field"] == "term"
    assert model_chart.to_dict()["resolve"]["scale"]["x"] == "independent"
    assert model_chart.to_dict()["resolve"]["scale"]["y"] == "independent"
    assert model_chart.to_dict()["spec"]["layer"][0]["mark"]["type"] == "line"
    assert model_chart.to_dict()["spec"]["layer"][1]["mark"]["type"] == "point"
    assert cv_chart.to_dict()["width"] == 420
    assert aic_chart.to_dict()["height"] == 260
    assert vi_chart.to_dict()["mark"]["type"] == "bar"


def test_confint_returns_fitted_and_partial_intervals_for_gaussian_models():
    x = np.linspace(0.0, 1.0, 30)
    y = np.sin(2.0 * np.pi * x)
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=6, nu=0.1),
    )

    fitted_ci = confint(model)
    partial_ci = confint(model, which=0)

    fitted_df = fitted_ci.to_pandas()
    partial_df = partial_ci.to_pandas()

    assert {"estimate", "lower", "upper", "std_error"} <= set(fitted_df.columns)
    assert {"term", "feature", "x", "estimate", "lower", "upper"} <= set(partial_df.columns)
    assert np.all(fitted_df["lower"] <= fitted_df["estimate"])
    assert np.all(fitted_df["estimate"] <= fitted_df["upper"])
    assert np.all(partial_df["lower"] <= partial_df["estimate"])
    assert np.all(partial_df["estimate"] <= partial_df["upper"])


def test_confint_supports_bootstrap_intervals():
    x = np.linspace(0.0, 1.0, 24)
    y = np.sin(2.0 * np.pi * x)
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    fitted_ci = confint(model, method="bootstrap", B=12, random_state=1)
    partial_ci = confint(model, which=0, method="bootstrap", B=12, grid_size=24, random_state=1)

    fitted_df = fitted_ci.to_pandas()
    partial_df = partial_ci.to_pandas()

    assert fitted_ci.method == "bootstrap"
    assert partial_ci.method == "bootstrap"
    assert not fitted_ci.approximate
    assert not partial_ci.approximate
    assert np.all(fitted_df["lower"] <= fitted_df["upper"])
    assert np.all(partial_df["lower"] <= partial_df["upper"])


def test_plot_smoke_covers_confint():
    x = np.linspace(-1.0, 1.0, 24)
    y = x**2
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    fitted_chart = plot(confint(model))
    partial_chart = plot(confint(model, which=0))

    assert fitted_chart.to_dict()["layer"][0]["mark"]["type"] == "area"
    assert partial_chart.to_dict()["facet"]["field"] == "term"


def test_varimp_percentages_sum_to_hundred():
    x = np.linspace(-1.0, 1.0, 24)
    z = x**2
    y = 0.5 * x - 0.2 * z
    model = glmboost(
        "y ~ x + z",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    result = varimp(model, percent=True, type="variable").to_pandas()

    np.testing.assert_allclose(result["reduction"].sum(), 100.0, atol=1e-12, rtol=1e-12)


def test_plotting_spline_model_does_not_emit_runtime_warning():
    x = np.linspace(0.0, 1.0, 30)
    y = np.sin(2.0 * np.pi * x)
    model = gamboost(
        "y ~ bbs(x, knots=5, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        chart = plot(model)
        chart.to_dict()
