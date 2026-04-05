from __future__ import annotations

import numpy as np


def test_r_mboost_gaussian_bols_reference_path(r_reference_runner):
    x = np.linspace(-1.0, 1.0, 12)
    y = 1.5 * x - 0.25

    result = r_reference_runner(x, y, family="gaussian", mstop=4, nu=0.1)

    assert set(result) == {"fitted", "risk"}
    assert result["fitted"].shape == y.shape
    assert result["risk"].ndim == 1
    assert result["risk"].size >= 1
    assert np.isfinite(result["fitted"]).all()
    assert np.isfinite(result["risk"]).all()
