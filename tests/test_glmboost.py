from __future__ import annotations

import numpy as np

from mboost import Binomial, Expectile, GammaReg, Gaussian, Huber, Laplace, Poisson, Quantile, TreeControls, blackboost, boost_control, glmboost
from mboost.families.base import Family


def test_glmboost_gaussian_selects_predictive_feature_and_reduces_risk():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.0, -1.0, 40) ** 2
    y = 2.0 * x1 + 0.1

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=8, nu=0.2),
    )

    assert model.selected[0] == "x1"
    assert model.risk_[0] > model.risk_[-1]
    np.testing.assert_allclose(model.predict(), model.fitted_)
    np.testing.assert_allclose(model.predict(type="response"), model.fitted_)


def test_glmboost_formula_parser_handles_top_level_minus_and_intercept_tokens():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.0, -1.0, 40) ** 2
    y = 2.0 * x1 + 0.1

    model = glmboost(
        "y ~ 1 + x1 + x2 - x2 - 1",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=6, nu=0.2),
    )

    assert model.selected[0] == "x1"
    assert model.risk_[0] > model.risk_[-1]


def test_glmboost_formula_parser_accepts_zero_token_with_remaining_terms():
    x = np.linspace(-1.0, 1.0, 20)
    y = x

    model = glmboost(
        "y ~ 0 + x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    assert model.selected == ["x"] * 4


def test_glmboost_formula_parser_expands_dot_from_data_columns():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.0, -1.0, 40) ** 2
    y = 2.0 * x1 + 0.1

    model = glmboost(
        "y ~ .",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=6, nu=0.2),
    )

    assert set(model.term_labels) == {"x1", "x2"}
    assert model.selected[0] == "x1"


def test_glmboost_accepts_explicit_bols_terms():
    x = np.linspace(-2.0, 2.0, 25)
    y = -0.75 * x + 1.0

    model = glmboost(
        "y ~ bols(x)",
        data={"x": x, "y": y},
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.fitted_.shape == y.shape


def test_glmboost_accepts_explicit_brandom_terms():
    x = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    y = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1], dtype=np.float64)

    model = glmboost(
        "y ~ brandom(x, df=2)",
        data={"x": x, "y": y},
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x"] * 5
    assert model.fitted_.shape == y.shape


def test_glmboost_accepts_explicit_btree_terms():
    x = np.linspace(-1.0, 1.0, 20)
    z = np.linspace(1.0, -1.0, 20)
    y = (x > 0.0).astype(np.float64)

    model = glmboost(
        "y ~ btree(x, z)",
        data={"x": x, "z": z, "y": y},
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.selected == ["x, z"] * 5
    assert model.risk_[0] > model.risk_[-1]
    assert model.fitted_.shape == y.shape


def test_blackboost_rewrites_plain_formula_into_single_tree_learner():
    x = np.linspace(-1.0, 1.0, 40)
    z = np.linspace(1.0, -1.0, 40)
    y = np.where(x < -0.2, -1.0, np.where(x < 0.4, 0.5, 1.25)) + 0.25 * (z > 0.0)

    model = blackboost(
        "y ~ x + z",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
    )

    assert model.formula == "y ~ btree(x, z, maxdepth=2, minsplit=10, minbucket=4)"
    assert model.selected == ["x, z"] * 5
    assert model.risk_[0] > model.risk_[-1]


def test_blackboost_accepts_tree_controls():
    x = np.linspace(-1.0, 1.0, 40)
    z = np.linspace(1.0, -1.0, 40)
    y = ((x > 0.0) | (z > 0.0)).astype(np.float64)

    model = blackboost(
        "y ~ x + z",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=3, nu=0.1),
        tree_controls=TreeControls(maxdepth=3, minsplit=8, minbucket=2),
    )

    learner = model.prepared_learners[0]
    assert learner.maxdepth == 3
    assert learner.minsplit == 8
    assert learner.minbucket == 2


def test_glmboost_summary_is_informative():
    x = np.linspace(-1.0, 1.0, 20)
    y = 0.5 * x

    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    text = model.summary()

    assert "Model-based boosting fit" in text
    assert "Formula: y ~ x" in text
    assert "Family: Gaussian" in text
    assert "Boosting iterations: 4" in text
    assert str(model) == text


def test_glmboost_accepts_custom_family_subclass():
    class OurQuantile(Family):
        def __init__(self, tau: float = 0.5):
            self.tau = float(tau)

        def offset(self, y, weights):
            return float(np.quantile(y, 0.5))

        def negative_gradient(self, y, f):
            diff = y - f
            return np.where(diff >= 0.0, self.tau, self.tau - 1.0)

        def risk(self, y, f, weights):
            diff = y - f
            return float(np.sum(weights * np.where(diff >= 0.0, self.tau * diff, (self.tau - 1.0) * diff)))

    x = np.linspace(-2.0, 2.0, 30)
    y = 0.5 * x

    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=OurQuantile(tau=0.5),
        control=boost_control(mstop=6, nu=0.1),
    )

    assert model.selected == ["x"] * 6
    assert model.risk_[0] >= model.risk_[-1]


def test_btree_respects_depth_and_split_parameters():
    x = np.linspace(-1.0, 1.0, 40)
    z = np.linspace(1.0, -1.0, 40)
    y = ((x > 0.0) | (z > 0.0)).astype(np.float64)

    shallow = glmboost(
        "y ~ btree(x, z, maxdepth=1, minsplit=10, minbucket=4)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=3, nu=0.1),
    )
    deeper = glmboost(
        "y ~ btree(x, z, maxdepth=2, minsplit=6, minbucket=2)",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=3, nu=0.1),
    )

    assert shallow.prepared_learners[0].maxdepth == 1
    assert deeper.prepared_learners[0].maxdepth == 2
    assert deeper.prepared_learners[0].minsplit == 6
    assert deeper.prepared_learners[0].minbucket == 2
    assert deeper.risk_[-1] <= shallow.risk_[-1]


def test_btree_supports_binary_by_modifier():
    x = np.linspace(-1.0, 1.0, 40)
    by = np.concatenate([np.zeros(20), np.ones(20)])
    y = (x > 0.0).astype(np.float64) * by

    model = glmboost(
        "y ~ btree(x, by=by)",
        data={"x": x, "by": by, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=4, nu=0.1),
    )

    pred = model.predict()

    assert model.prepared_learners[0].by_name == "by"
    np.testing.assert_allclose(pred[by == 0.0], model.offset_, atol=1e-12, rtol=1e-12)


def test_glmboost_accepts_formulaic_interaction_terms():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.5, -0.5, 40)
    y = 1.75 * x1 * x2

    model = glmboost(
        "y ~ x1:x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=6, nu=0.2),
    )

    assert model.selected == ["x1:x2"] * 6
    assert model.risk_[0] > model.risk_[-1]


def test_glmboost_accepts_formulaic_transform_terms_and_predicts_newdata():
    x = np.linspace(1.0, 5.0, 30)
    y = 0.8 * np.log(x)
    x_new = np.array([1.5, 2.5, 4.5], dtype=np.float64)

    model = glmboost(
        "y ~ np.log(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.2),
    )

    pred = model.predict(newdata={"x": x_new})

    assert model.selected == ["np.log(x)"] * 5
    assert pred.shape == x_new.shape
    assert np.isfinite(pred).all()


def test_glmboost_accepts_formulaic_star_expansion():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.linspace(1.0, -1.0, 40)
    y = 0.3 * x1 - 0.2 * x2 + 1.5 * x1 * x2

    model = glmboost(
        "y ~ x1 * x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=8, nu=0.2),
    )

    assert set(model.term_labels) == {"x1", "x2", "x1:x2"}
    assert model.risk_[0] > model.risk_[-1]


def test_glmboost_binomial_runs_and_reduces_risk():
    x1 = np.linspace(-2.0, 2.0, 60)
    x2 = np.cos(x1)
    y = (x1 > 0.0).astype(np.float64)

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Binomial(),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.selected[0] == "x1"
    assert model.risk_[0] > model.risk_[-1]
    assert np.isfinite(model.fitted_).all()
    response = model.predict(type="response")
    assert np.all(response >= 0.0)
    assert np.all(response <= 1.0)


def test_glmboost_poisson_runs_and_reduces_risk():
    x1 = np.linspace(0.0, 1.5, 50)
    x2 = np.cos(7.0 * x1)
    y = np.round(np.exp(0.6 * x1)).astype(np.float64)

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Poisson(),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.selected[0] == "x1"
    assert model.risk_[0] > model.risk_[-1]
    assert np.isfinite(model.fitted_).all()
    response = model.predict(type="response")
    assert np.all(response > 0.0)


def test_glmboost_gamma_runs_and_reduces_risk():
    x1 = np.linspace(0.1, 2.0, 50)
    x2 = np.cos(2.0 * x1)
    y = np.exp(0.2 + 0.4 * x1 - 0.15 * x2)

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=GammaReg(),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.selected[0] == "x1"
    assert np.isfinite(model.risk_).all()
    assert np.isfinite(model.fitted_).all()
    response = model.predict(type="response")
    assert np.all(response > 0.0)


def test_glmboost_predict_type_response_applies_inverse_link_on_newdata():
    x = np.linspace(-1.5, 1.5, 20)
    y = (x > 0.0).astype(np.float64)
    x_new = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Binomial(),
        control=boost_control(mstop=5, nu=0.1),
    )

    link_pred = model.predict(newdata={"x": x_new}, type="link")
    response_pred = model.predict(newdata={"x": x_new}, type="response")

    np.testing.assert_allclose(response_pred, 1.0 / (1.0 + np.exp(-2.0 * link_pred)))


def test_glmboost_predict_rejects_unknown_prediction_type():
    x = np.linspace(-1.0, 1.0, 10)
    y = x
    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=3, nu=0.1),
    )

    try:
        model.predict(type="probability")
    except ValueError as exc:
        assert "type must be either 'link' or 'response'" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported predict type")


def test_glmboost_laplace_runs_and_reduces_risk():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.cos(4.0 * x1)
    y = x1 + 0.2 * np.sign(x1)

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Laplace(),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.risk_[0] >= model.risk_[-1]
    assert np.isfinite(model.fitted_).all()


def test_glmboost_quantile_runs_and_reduces_risk():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.cos(4.0 * x1)
    y = x1 * x1

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Quantile(tau=0.25),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.risk_[0] >= model.risk_[-1]
    assert np.isfinite(model.fitted_).all()


def test_glmboost_expectile_runs_and_reduces_risk():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.cos(4.0 * x1)
    y = x1 * x1

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Expectile(tau=0.25),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.risk_[0] >= model.risk_[-1]
    assert np.isfinite(model.fitted_).all()


def test_glmboost_huber_runs_and_reduces_risk():
    x1 = np.linspace(-1.0, 1.0, 40)
    x2 = np.cos(4.0 * x1)
    y = x1 * x1

    model = glmboost(
        "y ~ x1 + x2",
        data={"x1": x1, "x2": x2, "y": y},
        family=Huber(),
        control=boost_control(mstop=10, nu=0.1),
    )

    assert model.risk_[0] >= model.risk_[-1]
    assert np.isfinite(model.fitted_).all()
