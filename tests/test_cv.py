from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, cv, cvrisk, glmboost, mstop


def test_cv_generates_balanced_deterministic_fold_ids():
    folds = cv(12, folds=3)
    assert folds.shape == (12,)
    assert set(folds.tolist()) == {0, 1, 2}
    counts = np.bincount(folds)
    assert counts.tolist() == [4, 4, 4]


def test_cv_supports_subsampling_mask_matrix():
    masks = cv(20, type="subsampling", B=4, fraction=0.5, random_state=1)
    assert masks.shape == (20, 4)
    assert set(np.unique(masks).tolist()) <= {0.0, 1.0}
    assert np.all(masks.sum(axis=0) == 10)


def test_cv_supports_bootstrap_oob_mask_matrix():
    masks = cv(20, type="bootstrap", B=3, random_state=1)
    assert masks.shape == (20, 3)
    assert set(np.unique(masks).tolist()) <= {0.0, 1.0}
    assert np.all(masks.sum(axis=0) > 0)


def test_cv_accepts_weight_vector_and_defaults_to_bootstrap_training_weights():
    weights = np.ones(12, dtype=np.float64)
    folds = cv(weights, random_state=1)

    assert folds.shape == (12, 25)
    assert np.all(folds >= 0.0)
    assert np.any(folds > 1.0)


def test_cv_weight_vector_kfold_returns_training_weight_matrix():
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    folds = cv(weights, type="kfold", B=3)

    assert folds.shape == (6, 3)
    for row_idx, weight in enumerate(weights):
        row = folds[row_idx]
        assert np.count_nonzero(row == 0.0) == 1
        assert np.count_nonzero(row == weight) == 2


def test_cv_weight_vector_supports_stratified_subsampling():
    weights = np.ones(8, dtype=np.float64)
    strata = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    folds = cv(weights, type="subsampling", B=2, prob=0.5, strata=strata, random_state=3)

    assert folds.shape == (8, 2)
    assert np.all(np.sum(folds[:4] > 0.0, axis=0) == 2)
    assert np.all(np.sum(folds[4:] > 0.0, axis=0) == 2)


def test_mstop_returns_truncated_model_path():
    x = np.linspace(-1.0, 1.0, 20)
    y = 1.25 * x + 0.2
    model = glmboost(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=6, nu=0.1),
    )

    short = mstop(model, 2)

    assert mstop(model) == 6
    assert mstop(short) == 2
    np.testing.assert_allclose(short.fitted_, model.predict(mstop=2))
    np.testing.assert_allclose(short.risk_, model.risk_[:3])


def test_cvrisk_returns_mean_foldwise_risk_path():
    x = np.linspace(-1.0, 1.0, 18)
    z = np.cos(5.0 * x)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)

    result = cvrisk(
        "y ~ x + z",
        data={"x": x, "z": z, "y": y},
        family=Gaussian(),
        control=control,
        folds=3,
    )

    assert result.fold_risk.shape == (3, 6)
    assert result.risk.shape == (6,)
    np.testing.assert_allclose(result.risk, result.fold_risk.mean(axis=0))
    assert 0 <= result.best_mstop <= control.mstop


def test_cvrisk_defaults_to_25_bootstrap_runs_when_folds_are_omitted():
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)

    result = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
    )

    assert result.fold_risk.shape == (25, 6)
    assert result.folds.shape == (18, 25)
    assert np.any(result.folds > 1.0)
    np.testing.assert_allclose(result.risk, result.fold_risk.mean(axis=0))


def test_cvrisk_accepts_holdout_mask_matrix():
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)
    masks = cv(x.shape[0], type="subsampling", B=4, fraction=0.5, random_state=2)

    result = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=masks,
    )

    assert result.fold_risk.shape == (4, 6)
    np.testing.assert_allclose(result.risk, result.fold_risk.mean(axis=0))


def test_cvrisk_can_generate_subsampling_masks_directly():
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)

    generated = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=4,
        type="subsampling",
        B=4,
        fraction=0.5,
        random_state=2,
    )
    explicit_masks = cv(x.shape[0], type="subsampling", B=4, fraction=0.5, random_state=2)
    explicit = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=explicit_masks,
    )

    np.testing.assert_allclose(generated.folds, 1.0 - explicit_masks)
    np.testing.assert_allclose(generated.fold_risk, explicit.fold_risk)
    np.testing.assert_allclose(generated.risk, explicit.risk)


def test_cvrisk_can_generate_bootstrap_masks_directly():
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)

    generated = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=3,
        type="bootstrap",
        B=3,
        random_state=3,
    )
    explicit_masks = cv(x.shape[0], type="bootstrap", B=3, random_state=3)
    explicit = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=explicit_masks,
    )

    np.testing.assert_allclose((generated.folds == 0.0).astype(np.float64), explicit_masks)
    assert generated.fold_risk.shape == explicit.fold_risk.shape
    assert np.all(np.isfinite(generated.fold_risk))
    assert np.all(np.isfinite(generated.risk))


def test_cvrisk_accepts_r_style_bootstrap_training_weight_matrix():
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)
    train_weights = np.array(
        [
            [2, 0, 1],
            [0, 1, 0],
            [1, 2, 1],
            [0, 0, 2],
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 2],
            [1, 1, 1],
            [0, 0, 1],
            [2, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )

    result = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=train_weights,
    )

    assert result.fold_risk.shape == (3, 6)
    np.testing.assert_allclose(result.folds, train_weights)
