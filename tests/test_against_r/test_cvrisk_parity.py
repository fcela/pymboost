from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, cvrisk


def _fold_matrix_from_ids(fold_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(fold_ids)
    folds = np.ones((fold_ids.shape[0], unique.shape[0]), dtype=np.float64)
    for idx, fold_id in enumerate(unique):
        folds[fold_ids == fold_id, idx] = 0.0
    return folds


def test_python_cvrisk_matches_r_for_gaussian_linear_term(r_cvrisk_runner):
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    fold_ids = np.arange(x.shape[0]) % 3
    fold_matrix = _fold_matrix_from_ids(fold_ids)
    control = boost_control(mstop=5, nu=0.1)

    py_result = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=fold_ids,
    )
    r_result = r_cvrisk_runner(
        x,
        y,
        family="gaussian",
        folds=fold_matrix,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        py_result.fold_risk,
        r_result["fold_risk"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_python_cvrisk_matches_r_for_explicit_bootstrap_training_weights(r_cvrisk_bootstrap_runner):
    x = np.linspace(-1.0, 1.0, 18)
    y = 1.5 * x - 0.1
    control = boost_control(mstop=5, nu=0.1)

    r_result = r_cvrisk_bootstrap_runner(
        x,
        y,
        B=4,
        mstop=5,
        nu=0.1,
        seed=7,
    )
    py_result = cvrisk(
        "y ~ x",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=r_result["folds"],
    )

    np.testing.assert_allclose(py_result.folds, r_result["folds"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result.fold_risk, r_result["fold_risk"], atol=1e-12, rtol=1e-12)
