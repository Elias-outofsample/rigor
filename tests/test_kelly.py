"""Tests for Kelly-criterion position sizing (rigor.analysis.kelly).

The five variants are checked against their closed forms and against each
other:

  * ``kelly_full`` equals ``W - (1-W)/R`` and is long-only clipped;
  * ``kelly_half`` / ``kelly_fractional`` are exact multiples of the full bet;
  * ``kelly_continuous`` equals ``mu / sigma^2`` for a known series and is signed;
  * ``kelly_multi_asset`` equals ``Sigma^{-1} mu`` (recovers a closed-form case);
  * ``kelly_shrinkage`` is a contraction of the full vector toward zero, with a
    factor in ``[0, 1)`` that grows with the sample, and is finite on a
    near-singular covariance;
  * every degenerate input (zero vol, single asset, empty, singular Sigma)
    returns a finite value or a zero vector — never NaN, never an exception.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rigor.analysis import (
    kelly_continuous,
    kelly_fractional,
    kelly_full,
    kelly_half,
    kelly_multi_asset,
    kelly_shrinkage,
)

# ---------------------------------------------------------------------------
# kelly_full / kelly_half / kelly_fractional — discrete bet
# ---------------------------------------------------------------------------

def test_kelly_full_closed_form() -> None:
    # W=0.6, R=2 -> 0.6 - 0.4/2 = 0.4
    assert kelly_full(0.6, 2.0) == pytest.approx(0.4)
    # W=0.55, R=1 (even money) -> 0.55 - 0.45 = 0.10
    assert kelly_full(0.55, 1.0) == pytest.approx(0.10)


def test_kelly_full_long_only_clip() -> None:
    # No edge: W=0.4, R=1 -> 0.4 - 0.6 = -0.2, clipped to 0.0.
    assert kelly_full(0.4, 1.0) == 0.0


def test_kelly_full_degenerate_payoff() -> None:
    assert kelly_full(0.6, 0.0) == 0.0
    assert kelly_full(0.6, -2.0) == 0.0
    assert kelly_full(0.6, float("inf")) == 0.0


def test_kelly_full_clamps_win_rate() -> None:
    # Out-of-range win rates are clamped to [0, 1], result stays finite.
    assert kelly_full(1.5, 2.0) == pytest.approx(1.0)  # W clamped to 1 -> 1 - 0 = 1
    assert kelly_full(-0.5, 2.0) == 0.0                 # W clamped to 0 -> max(-0.5, 0)


def test_kelly_half_is_exact_half() -> None:
    for w, r in [(0.6, 2.0), (0.55, 1.0), (0.7, 3.0)]:
        assert kelly_half(w, r) == pytest.approx(0.5 * kelly_full(w, r))
    assert kelly_half(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_fractional_multiples() -> None:
    full = kelly_full(0.6, 2.0)
    assert kelly_fractional(0.6, 2.0, 0.5) == pytest.approx(0.5 * full)
    assert kelly_fractional(0.6, 2.0, 0.25) == pytest.approx(0.25 * full)
    # Default fraction is half-Kelly.
    assert kelly_fractional(0.6, 2.0) == pytest.approx(kelly_half(0.6, 2.0))


def test_kelly_fractional_clamps_fraction() -> None:
    full = kelly_full(0.6, 2.0)
    assert kelly_fractional(0.6, 2.0, 2.0) == pytest.approx(full)   # clamped to 1
    assert kelly_fractional(0.6, 2.0, -1.0) == 0.0                  # clamped to 0


# ---------------------------------------------------------------------------
# kelly_continuous — f* = mu / sigma^2
# ---------------------------------------------------------------------------

def test_kelly_continuous_closed_form() -> None:
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    mu = r.mean()
    var = r.var(ddof=1)
    assert kelly_continuous(r) == pytest.approx(mu / var)


def test_kelly_continuous_signed() -> None:
    # A net-negative mean gives a negative (short) fraction.
    r = np.array([-0.02, -0.01, 0.005, -0.03, -0.01])
    assert kelly_continuous(r) < 0.0
    assert math.isfinite(kelly_continuous(r))


def test_kelly_continuous_zero_vol() -> None:
    # Constant series -> zero variance -> 0.0, not inf/NaN.
    assert kelly_continuous(np.full(10, 0.01)) == 0.0


def test_kelly_continuous_too_few_obs() -> None:
    assert kelly_continuous([]) == 0.0
    assert kelly_continuous([0.01]) == 0.0


def test_kelly_continuous_drops_non_finite() -> None:
    clean = np.array([0.01, 0.02, -0.01])
    noisy = np.array([0.01, np.nan, 0.02, np.inf, -0.01])
    assert kelly_continuous(noisy) == pytest.approx(kelly_continuous(clean))


# ---------------------------------------------------------------------------
# kelly_multi_asset — f* = Sigma^{-1} mu
# ---------------------------------------------------------------------------

def test_kelly_multi_asset_equals_sigma_inv_mu() -> None:
    mu = np.array([0.10, 0.05])
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    expected = np.linalg.inv(sigma) @ mu
    np.testing.assert_allclose(kelly_multi_asset(mu, sigma), expected)


def test_kelly_multi_asset_diagonal_closed_form() -> None:
    # Diagonal covariance decouples into the per-asset f = mu_i / var_i.
    mu = np.array([0.08, 0.06, 0.04])
    sigma = np.diag([0.04, 0.09, 0.01])
    expected = mu / np.array([0.04, 0.09, 0.01])
    np.testing.assert_allclose(kelly_multi_asset(mu, sigma), expected)


def test_kelly_multi_asset_singular_returns_zeros() -> None:
    mu = np.array([0.1, 0.2])
    sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1, singular
    f = kelly_multi_asset(mu, sigma)
    np.testing.assert_array_equal(f, np.zeros(2))


def test_kelly_multi_asset_single_asset() -> None:
    f = kelly_multi_asset(np.array([0.1]), np.array([[0.04]]))
    assert f.shape == (1,)
    assert f[0] == pytest.approx(0.1 / 0.04)


def test_kelly_multi_asset_empty() -> None:
    f = kelly_multi_asset(np.array([]), np.empty((0, 0)))
    assert f.shape == (0,)


def test_kelly_multi_asset_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        kelly_multi_asset(np.array([0.1, 0.2]), np.array([[0.04]]))
    with pytest.raises(ValueError):
        kelly_multi_asset(np.array([[0.1]]), np.array([[0.04]]))  # mu not 1-D


# ---------------------------------------------------------------------------
# kelly_shrinkage — Kan & Zhou (2007)
# ---------------------------------------------------------------------------

def _toy_returns(seed: int = 0, t: int = 500, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Small positive drift so the in-sample Sharpe is positive.
    return rng.normal(loc=0.001, scale=0.01, size=(t, n))


def test_shrinkage_is_contraction_of_full() -> None:
    r = _toy_returns()
    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False, ddof=1)
    f_full = np.linalg.inv(sigma) @ mu
    f_shrunk = kelly_shrinkage(r)

    # Same direction, strictly smaller magnitude (shrunk toward the zero-edge
    # naive estimate) — i.e. f_shrunk = c * f_full with 0 < c < 1.
    ratios = f_shrunk / f_full
    assert np.all(np.isfinite(f_shrunk))
    np.testing.assert_allclose(ratios, ratios[0])      # constant scalar factor
    c = float(ratios[0])
    assert 0.0 < c < 1.0
    assert np.all(np.abs(f_shrunk) < np.abs(f_full))


def test_shrinkage_factor_grows_with_sample() -> None:
    # More data -> more confidence -> less shrinkage (factor closer to 1).
    short = _toy_returns(t=60)
    long = _toy_returns(t=4000)

    def _factor(r: np.ndarray) -> float:
        mu = r.mean(axis=0)
        sigma = np.cov(r, rowvar=False, ddof=1)
        f_full = np.linalg.inv(sigma) @ mu
        return float((kelly_shrinkage(r) / f_full)[0])

    assert _factor(short) < _factor(long)


def test_shrinkage_finite_on_near_singular_cov() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.001, 0.01, size=(800, 1))
    # Three assets that are near-perfect copies -> covariance near singular.
    r = np.hstack([base, base + 1e-9 * rng.normal(size=(800, 1)),
                   base + 1e-9 * rng.normal(size=(800, 1))])
    f = kelly_shrinkage(r)
    assert f.shape == (3,)
    assert np.all(np.isfinite(f))


def test_shrinkage_too_few_obs_returns_zeros() -> None:
    # T <= N + 2 is below Kan & Zhou's regularity condition.
    r = np.random.default_rng(2).normal(size=(4, 3))  # T=4, N=3 -> 4 <= 5
    np.testing.assert_array_equal(kelly_shrinkage(r), np.zeros(3))


def test_shrinkage_negative_sharpe_returns_zeros() -> None:
    # All-negative drift -> in-sample squared Sharpe is still positive (it is a
    # quadratic form), so to force the <=0 guard we use a degenerate zero-mean
    # series whose mean rounds to exactly the naive no-edge case.
    r = np.zeros((100, 2))
    np.testing.assert_array_equal(kelly_shrinkage(r), np.zeros(2))


def test_shrinkage_single_asset_1d_input() -> None:
    r = np.random.default_rng(3).normal(0.001, 0.01, size=200)
    f = kelly_shrinkage(r)
    assert f.shape == (1,)
    assert np.all(np.isfinite(f))


def test_shrinkage_n_obs_override() -> None:
    r = _toy_returns(t=500)
    # A smaller effective sample shrinks harder than the full T.
    f_small = kelly_shrinkage(r, n_obs=30)
    f_full = kelly_shrinkage(r, n_obs=500)
    assert np.all(np.abs(f_small) < np.abs(f_full))


def test_shrinkage_drops_non_finite_returns_zeros() -> None:
    r = _toy_returns()
    r[5, 1] = np.nan
    np.testing.assert_array_equal(kelly_shrinkage(r), np.zeros(3))


def test_shrinkage_empty_matrix() -> None:
    f = kelly_shrinkage(np.empty((0, 0)))
    assert f.shape == (0,)
