from __future__ import annotations

import numpy as np
import pytest

from mboost import Gaussian, boost_control


def test_boost_control_validates_bounds():
    with pytest.raises(ValueError):
        boost_control(mstop=0)
    with pytest.raises(ValueError):
        boost_control(nu=0.0)
    with pytest.raises(ValueError):
        boost_control(nu=1.5)


def test_gaussian_family_offset_gradient_and_risk():
    family = Gaussian()
    y = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    w = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    f = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(3.5)
    np.testing.assert_allclose(family.negative_gradient(y, f), np.array([-1.0, 1.0, 3.0]))
    assert family.risk(y, f, w) == pytest.approx(20.0)
