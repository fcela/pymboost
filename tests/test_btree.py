from __future__ import annotations

import numpy as np

from mboost import Gaussian, boost_control, cvrisk


def test_btree_cvrisk_returns_finite_risk_path():
    x = np.linspace(-1.0, 1.0, 24)
    y = (x > 0.0).astype(np.float64)
    fold_ids = np.arange(x.shape[0]) % 4

    result = cvrisk(
        "y ~ btree(x)",
        data={"x": x, "y": y},
        family=Gaussian(),
        control=boost_control(mstop=5, nu=0.1),
        folds=fold_ids,
    )

    assert result.fold_risk.shape == (4, 6)
    assert result.risk.shape == (6,)
    assert np.isfinite(result.fold_risk).all()
    assert np.isfinite(result.risk).all()
    np.testing.assert_allclose(result.risk, result.fold_risk.mean(axis=0))
