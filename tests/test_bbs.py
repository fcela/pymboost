from __future__ import annotations

import warnings

import numpy as np

from mboost import Binomial, Gaussian, Poisson, gamboost, glmboost, boost_control


def test_glmboost_gaussian_with_bbs_runs_and_reduces_risk():
    x = np.linspace(0.0, 1.0, 60)
    y = np.sin(2.0 * np.pi * x)

    model = glmboost(
        "y ~ bbs(x, df=6, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=15, nu=0.1),
    )

    assert model.selected == ["x"] * 15
    assert model.risk_[0] > model.risk_[-1]
    assert model.fitted_.shape == y.shape


def test_bbs_beats_linear_on_smooth_signal():
    x = np.linspace(0.0, 1.0, 80)
    y = np.sin(2.0 * np.pi * x)
    control = boost_control(mstop=20, nu=0.1)

    linear = glmboost("y ~ x", data={"x": x, "y": y}, family=Gaussian(), control=control)
    spline = glmboost(
        "y ~ bbs(x, df=8, lambda=1.0, degree=3, differences=2)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )

    assert spline.risk_[-1] < linear.risk_[-1]


def test_gamboost_alias_supports_bbs_formula():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)

    model = gamboost(
        "y ~ bbs(x, df=6, lambda=1.0)",
        data={"x": x, "y": y},
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.selected[0] == "x"
    assert model.risk_[0] > model.risk_[-1]


def test_gamboost_accepts_bbs_by_formula():
    x = np.linspace(0.0, 1.0, 25)
    z = np.linspace(-1.0, 1.0, 25)
    y = np.sin(2.0 * np.pi * x) * z

    model = gamboost(
        "y ~ bbs(x, df=4, knots=5, by=z)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.risk_[0] > model.risk_[-1]
    assert model.fitted_.shape == y.shape


def test_gamboost_dfbase_rewrites_bare_terms_to_bbs():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)

    model = gamboost(
        "y ~ x",
        data={"x": x, "y": y},
        dfbase=4,
        control=boost_control(mstop=10, nu=0.1),
    )

    assert "bbs(x, df=4, knots=20)" in model.formula
    assert model.risk_[0] > model.risk_[-1]


def test_gamboost_centered_default_bbs_predicts_without_runtime_warning():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x)
    x_new = np.linspace(0.1, 0.9, 9)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = gamboost(
            "y ~ bbs(x, center=True)",
            data={"x": x, "y": y},
            family=Gaussian(),
            control=boost_control(mstop=5, nu=0.1),
        )
        pred = model.predict(newdata={"x": x_new})

    assert pred.shape == x_new.shape


def test_glmboost_gaussian_with_bmono_is_monotone_increasing():
    x = np.linspace(0.0, 1.0, 60)
    y = x**2

    model = glmboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(pred) >= -1e-8)


def test_glmboost_gaussian_with_bmono_is_monotone_decreasing():
    x = np.linspace(0.0, 1.0, 60)
    y = (1.0 - x) ** 2

    model = glmboost(
        'y ~ bmono(x, constraint="decreasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(pred) <= 1e-8)


def test_glmboost_gaussian_with_bmono_is_convex():
    x = np.linspace(0.0, 1.0, 60)
    y = x**2

    model = glmboost(
        'y ~ bmono(x, constraint="convex", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(pred, n=2) >= -1e-8)


def test_glmboost_gaussian_with_bmono_is_concave():
    x = np.linspace(0.0, 1.0, 60)
    y = 1.0 - (1.0 - x) ** 2

    model = glmboost(
        'y ~ bmono(x, constraint="concave", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(pred, n=2) <= 1e-8)


def test_glmboost_gaussian_with_bmono_is_positive():
    x = np.linspace(0.0, 1.0, 60)
    y = 0.25 + x**2

    model = glmboost(
        'y ~ bmono(x, constraint="positive", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(pred >= -1e-8)


def test_glmboost_gaussian_with_bmono_is_negative():
    x = np.linspace(0.0, 1.0, 60)
    y = -(0.25 + x**2)

    model = glmboost(
        'y ~ bmono(x, constraint="negative", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(pred <= 1e-8)


def test_glmboost_gaussian_with_bmono_iterative_solver_is_monotone():
    x = np.linspace(0.0, 1.0, 60)
    y = x**2

    model = glmboost(
        'y ~ bmono(x, constraint="increasing", type="iterative", niter=20, knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=10, nu=0.1),
    )

    pred = model.predict()
    assert model.selected == ["x"] * 10
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(pred) >= -1e-8)


def test_glmboost_gaussian_with_bmono_iterative_solver_stays_close_to_quad_prog():
    x = np.linspace(0.0, 1.0, 60)
    y = x**2
    control = boost_control(mstop=10, nu=0.1)

    quad_prog = glmboost(
        'y ~ bmono(x, constraint="increasing", type="quad.prog", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )
    iterative = glmboost(
        'y ~ bmono(x, constraint="increasing", type="iterative", niter=20, knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )

    np.testing.assert_allclose(iterative.predict(), quad_prog.predict(), atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(iterative.risk_, quad_prog.risk_, atol=5e-4, rtol=5e-4)


def test_glmboost_gaussian_with_bmono_accepts_boundary_constraints_for_quad_prog():
    x = np.linspace(0.0, 1.0, 60)
    y = x**2

    model = glmboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2, boundary_constraints=True)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]


def test_glmboost_gaussian_with_bmono_rejects_boundary_constraints_for_iterative():
    x = np.linspace(0.0, 1.0, 20)
    y = x**2

    try:
        glmboost(
            'y ~ bmono(x, constraint="increasing", type="iterative", niter=20, knots=5, lambda=1.0, degree=3, differences=2, boundary_constraints=True)',
            data={"x": x, "y": y},
            family=Gaussian(),
            control=boost_control(mstop=3, nu=0.1),
        )
    except NotImplementedError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected NotImplementedError for iterative boundary_constraints")


def test_glmboost_gaussian_with_bmono_accepts_by_modifier():
    x = np.linspace(0.0, 1.0, 25)
    z = np.linspace(0.5, 1.5, 25)
    y = z * x**2

    model = glmboost(
        'y ~ bmono(x, constraint="increasing", by=z, knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]


def test_glmboost_gaussian_with_factor_bmono_is_monotone_increasing():
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float64)

    model = glmboost(
        'y ~ bmono(x, constraint="increasing", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    pred = model.predict()
    level_means = [float(np.mean(pred[x == level])) for level in ["a", "b", "c"]]
    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert level_means[0] <= level_means[1] <= level_means[2]


def test_glmboost_gaussian_with_factor_bmono_is_monotone_decreasing():
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([2.2, 2.1, 2.0, 1.2, 1.1, 1.0, 0.2, 0.1, 0.0], dtype=np.float64)

    model = glmboost(
        'y ~ bmono(x, constraint="decreasing", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    pred = model.predict()
    level_means = [float(np.mean(pred[x == level])) for level in ["a", "b", "c"]]
    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert level_means[0] >= level_means[1] >= level_means[2]


def test_glmboost_gaussian_with_factor_bmono_is_convex():
    x = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, dtype=object)
    y = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 2.2, 2.3, 2.4], dtype=np.float64)

    model = glmboost(
        'y ~ bmono(x, constraint="convex", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    pred = model.predict()
    level_means = np.array([float(np.mean(pred[x == level])) for level in ["a", "b", "c", "d"]])
    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(level_means, n=2) >= -1e-8)


def test_glmboost_gaussian_with_factor_bmono_is_concave():
    x = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, dtype=object)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4], dtype=np.float64)

    model = glmboost(
        'y ~ bmono(x, constraint="concave", lambda=1.0)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    pred = model.predict()
    level_means = np.array([float(np.mean(pred[x == level])) for level in ["a", "b", "c", "d"]])
    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert np.all(np.diff(level_means, n=2) <= 1e-8)


def test_gamboost_binomial_with_bmono_runs_and_reduces_risk():
    x = np.linspace(0.0, 1.0, 30)
    eta = -2.0 + 5.0 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)

    model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Binomial(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert np.isfinite(model.fitted_).all()


def test_gamboost_poisson_with_bmono_runs_and_reduces_risk():
    x = np.linspace(0.0, 1.0, 30)
    y = np.round(np.exp(0.2 + 1.2 * x)).astype(np.float64)

    model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Poisson(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert np.isfinite(model.fitted_).all()
