from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, cvrisk


def _fold_matrix_from_ids(fold_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(fold_ids)
    folds = np.ones((fold_ids.shape[0], unique.shape[0]), dtype=np.float64)
    for idx, fold_id in enumerate(unique):
        folds[fold_ids == fold_id, idx] = 0.0
    return folds


def test_python_cvrisk_matches_r_for_monotone_increasing_spline(
    r_gamboost_bmono_cvrisk_runner,
):
    x = np.linspace(0.0, 1.0, 24)
    y = x**2
    fold_ids = np.arange(x.shape[0]) % 4
    fold_matrix = _fold_matrix_from_ids(fold_ids)
    control = boost_control(mstop=5, nu=0.1)

    py_result = cvrisk(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Gaussian(),
        control=control,
        folds=fold_ids,
    )
    r_result = r_gamboost_bmono_cvrisk_runner(
        x,
        y,
        folds=fold_matrix,
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(py_result.fold_risk, r_result["fold_risk"], atol=5e-5, rtol=5e-5)
