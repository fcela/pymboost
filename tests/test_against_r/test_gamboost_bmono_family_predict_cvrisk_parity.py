from __future__ import annotations

import numpy as np

from mboost import Binomial, Poisson, boost_control, cvrisk, gamboost


def _fold_matrix_from_ids(fold_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(fold_ids)
    folds = np.ones((fold_ids.shape[0], unique.shape[0]), dtype=np.float64)
    for idx, fold_id in enumerate(unique):
        folds[fold_ids == fold_id, idx] = 0.0
    return folds


def test_python_gamboost_matches_r_for_binomial_bmono_newdata_prediction(
    r_gamboost_bmono_family_predict_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    eta = -2.0 + 5.0 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)
    x_new = np.linspace(0.1, 0.9, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Binomial(),
        control=control,
    )
    r_result = r_gamboost_bmono_family_predict_runner(
        x,
        y,
        x_new=x_new,
        family="binomial",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-4,
        rtol=1e-4,
    )


def test_python_gamboost_matches_r_for_poisson_bmono_newdata_prediction(
    r_gamboost_bmono_family_predict_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    y = np.round(np.exp(0.2 + 1.2 * x)).astype(np.float64)
    x_new = np.linspace(0.1, 0.9, 7)
    control = boost_control(mstop=5, nu=0.1)

    py_model = gamboost(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Poisson(),
        control=control,
    )
    r_result = r_gamboost_bmono_family_predict_runner(
        x,
        y,
        x_new=x_new,
        family="poisson",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(
        py_model.predict(newdata={"x": x_new}),
        r_result["pred"],
        atol=1e-4,
        rtol=1e-4,
    )


def test_python_cvrisk_matches_r_for_binomial_bmono(
    r_gamboost_bmono_family_cvrisk_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    eta = -2.0 + 5.0 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)
    fold_ids = np.arange(x.shape[0]) % 3
    fold_matrix = _fold_matrix_from_ids(fold_ids)
    control = boost_control(mstop=5, nu=0.1)

    py_result = cvrisk(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Binomial(),
        control=control,
        folds=fold_ids,
    )
    r_result = r_gamboost_bmono_family_cvrisk_runner(
        x,
        y,
        folds=fold_matrix,
        family="binomial",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(py_result.fold_risk, r_result["fold_risk"], atol=1e-4, rtol=1e-4)


def test_python_cvrisk_matches_r_for_poisson_bmono(
    r_gamboost_bmono_family_cvrisk_runner,
):
    x = np.linspace(0.0, 1.0, 30)
    y = np.round(np.exp(0.2 + 1.2 * x)).astype(np.float64)
    fold_ids = np.arange(x.shape[0]) % 3
    fold_matrix = _fold_matrix_from_ids(fold_ids)
    control = boost_control(mstop=5, nu=0.1)

    py_result = cvrisk(
        'y ~ bmono(x, constraint="increasing", knots=5, lambda=1.0, degree=3, differences=2)',
        data={"x": x, "y": y},
        family=Poisson(),
        control=control,
        folds=fold_ids,
    )
    r_result = r_gamboost_bmono_family_cvrisk_runner(
        x,
        y,
        folds=fold_matrix,
        family="poisson",
        constraint="increasing",
        knots=5,
        lambda_value=1.0,
        degree=3,
        differences=2,
        mstop=5,
        nu=0.1,
    )

    np.testing.assert_allclose(py_result.risk, r_result["risk"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(py_result.fold_risk, r_result["fold_risk"], atol=1e-4, rtol=1e-4)
