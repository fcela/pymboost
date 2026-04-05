from __future__ import annotations

import math

import numpy as np
import pytest

from mboost import Binomial, Expectile, GammaReg, Gaussian, Huber, Laplace, Poisson, Quantile


def test_binomial_family_offset_gradient_and_risk():
    family = Binomial()
    y = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    f = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(0.5 * np.log(3.0))
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([-1.0, 1.0, 1.0]) / np.log(2.0),
    )
    assert family.risk(y, f, w) == pytest.approx(4.0 * np.log(2.0))


def test_poisson_family_offset_gradient_and_risk():
    family = Poisson()
    y = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    w = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    f = np.zeros(3, dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(np.log(2.75))
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([0.0, 1.0, 3.0]),
    )
    assert family.risk(y, f, w) == pytest.approx(4.0 + np.log(2.0) + 2.0 * np.log(24.0))


def test_laplace_family_offset_gradient_and_risk():
    family = Laplace()
    y = np.array([0.0, 1.0, 10.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    f = np.array([1.0, 1.0, 5.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(1.0)
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([-1.0, 0.0, 1.0]),
    )
    assert family.risk(y, f, w) == pytest.approx(6.0)


def test_quantile_family_offset_gradient_and_risk():
    family = Quantile(tau=0.25)
    y = np.array([0.0, 1.0, 10.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    f = np.array([1.0, 1.0, 5.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(1.0)
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([-0.75, 0.25, 0.25]),
    )
    assert family.risk(y, f, w) == pytest.approx(2.0)


def test_expectile_family_offset_gradient_and_risk():
    family = Expectile(tau=0.25)
    y = np.array([-1.0, 0.0, 2.0], dtype=np.float64)
    w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    f = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([-1.5, 0.0, 1.0]),
    )
    assert family.risk(y, f, w) == pytest.approx(1.75)


def test_huber_family_offset_gradient_and_risk():
    family = Huber(d=1.0)
    y = np.array([0.0, 1.0, 10.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    f = np.array([1.0, 1.0, 5.0], dtype=np.float64)

    assert family.offset(y, w) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(
        family.negative_gradient(y, f),
        np.array([-1.0, 0.0, 1.0]),
    )
    assert family.risk(y, f, w) == pytest.approx(5.0)


def test_gamma_family_offset_gradient_and_risk():
    family = GammaReg()
    y = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    w = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    offset = family.offset(y, w)

    assert offset == pytest.approx(1.011603, abs=5e-5)

    f = np.zeros(3, dtype=np.float64)
    family.calibrate(y, f, w)
    grad = family.negative_gradient(y, f)
    sigma = family.sigma
    np.testing.assert_allclose(
        grad,
        sigma * y - sigma,
        rtol=1e-8,
        atol=1e-8,
    )
    expected_risk = (
        math.lgamma(sigma) + sigma * 1.0 - sigma * math.log(1.0) - sigma * math.log(sigma)
        + math.lgamma(sigma) + sigma * 2.0 - sigma * math.log(2.0) - sigma * math.log(sigma)
        + 2.0 * (math.lgamma(sigma) + sigma * 4.0 - sigma * math.log(4.0) - sigma * math.log(sigma))
    )
    assert family.risk(y, f, w) == pytest.approx(expected_risk, abs=1e-6)
