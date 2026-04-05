from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, cvrisk


def _fold_matrix_from_ids(fold_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(fold_ids)
    folds = np.ones((fold_ids.shape[0], unique.shape[0]), dtype=np.float64)
    for idx, fold_id in enumerate(unique):
        folds[fold_ids == fold_id, idx] = 0.0
    return folds


def test_python_cvrisk_matches_r_for_gaussian_factor_term(r_factor_cvrisk_runner):
    x = np.array(["a", "a", "b", "b", "c", "c", "a", "b", "c"], dtype=object)
    y = np.array([0.0, 0.2, 1.0, 1.1, 2.0, 2.2, 0.1, 0.9, 2.1], dtype=np.float64)
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
    r_result = r_factor_cvrisk_runner(
        x,
        y,
        folds=fold_matrix,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(py_result.fold_risk, r_result["fold_risk"], atol=1e-12, rtol=1e-12)
