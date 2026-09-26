"""Tests for the risk_budget weight solver and its allocate_weights integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.weight_methods import (
    METHODS,
    allocate_weights,
    risk_budget,
    risk_parity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_psd_cov(n: int, seed: int = 42) -> np.ndarray:
    """Return a well-conditioned (n × n) covariance matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n * 4, n))
    return np.cov(a.T) + 0.05 * np.eye(n)


def _risk_contrib_shares(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Return normalised risk-contribution shares RCᵢ / σ_p for weight vector w."""
    portfolio_var = float(w @ cov @ w)
    if portfolio_var <= 0:
        return np.ones(len(w)) / len(w)
    sigma_p = portfolio_var**0.5
    rc = w * (cov @ w) / sigma_p        # absolute risk contributions
    return rc / sigma_p                 # normalised shares (sum to 1)


def _returns_df(n: int = 3, bars: int = 500, seed: int = 0) -> pd.DataFrame:
    """Random daily-returns DataFrame with n columns."""
    rng = np.random.default_rng(seed)
    data = 0.01 * rng.standard_normal((bars, n))
    idx = pd.bdate_range("2020-01-01", periods=bars)
    return pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# Test 1 — uniform budget ≈ risk_parity
# ---------------------------------------------------------------------------

def test_uniform_budget_matches_risk_parity() -> None:
    """risk_budget with 1/n budget should produce weights close to risk_parity."""
    n = 5
    cov = _make_psd_cov(n, seed=1)
    budget = np.ones(n) / n

    w_rb = risk_budget(cov, budget)
    w_rp = risk_parity(cov)

    assert w_rb.shape == (n,)
    assert abs(w_rb.sum() - 1.0) < 1e-6
    np.testing.assert_allclose(w_rb, w_rp, atol=0.02,
                               err_msg="Uniform risk_budget should be close to risk_parity")


# ---------------------------------------------------------------------------
# Test 2 — skewed budget — realised RC shares match target
# ---------------------------------------------------------------------------

def test_skewed_budget_rc_shares_match() -> None:
    """Realised risk-contribution shares should be within 0.05 of the target budget."""
    n = 3
    cov = _make_psd_cov(n, seed=2)
    target = np.array([0.6, 0.3, 0.1])

    w = risk_budget(cov, target)

    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-10).all(), "weights must be non-negative"

    realised = _risk_contrib_shares(w, cov)
    target_norm = target / target.sum()

    np.testing.assert_allclose(realised, target_norm, atol=0.05,
                               err_msg="Realised RC shares deviate too far from target budget")


# ---------------------------------------------------------------------------
# Test 3 — long-only, sums to 1, respects max_weight cap
# ---------------------------------------------------------------------------

def test_long_only_sum_to_one_and_max_weight() -> None:
    """Weights are non-negative, sum to 1, and respect the max_weight cap."""
    n = 4
    cov = _make_psd_cov(n, seed=3)
    budget = np.array([0.5, 0.2, 0.2, 0.1])
    cap = 0.40

    w = risk_budget(cov, budget, max_weight=cap)

    assert w.shape == (n,)
    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}, not 1"
    assert (w >= -1e-10).all(), "negative weight detected"
    assert w.max() <= cap + 1e-6, f"max weight {w.max()} exceeds cap {cap}"


# ---------------------------------------------------------------------------
# Test 4 — degenerate covariance → fallback without raising
# ---------------------------------------------------------------------------

def test_degenerate_covariance_fallback() -> None:
    """A zero (degenerate) covariance should not raise; result is a valid simplex vector."""
    n = 3
    cov_zero = np.zeros((n, n))
    budget = np.array([0.5, 0.3, 0.2])

    # Must not raise
    w = risk_budget(cov_zero, budget)

    assert w.shape == (n,)
    assert abs(w.sum() - 1.0) < 1e-6, f"fallback weights sum to {w.sum()}, not 1"
    assert (w >= -1e-10).all(), "negative weight in fallback"


# ---------------------------------------------------------------------------
# Test 5 — allocate_weights integration
# ---------------------------------------------------------------------------

def test_allocate_weights_risk_budget_sums_to_one() -> None:
    """allocate_weights with method='risk_budget' must return weights summing to 1."""
    rm = _returns_df(n=4, bars=600, seed=10)
    budget = np.array([0.4, 0.3, 0.2, 0.1])

    w = allocate_weights(rm, "risk_budget", target_budget=budget)

    assert isinstance(w, np.ndarray)
    assert w.shape == (4,)
    assert abs(w.sum() - 1.0) < 1e-6, f"allocate_weights returned weights summing to {w.sum()}"
    assert (w >= -1e-10).all(), "allocate_weights returned negative weight"


def test_allocate_weights_risk_budget_default_uniform() -> None:
    """allocate_weights with method='risk_budget' and no budget defaults to equal risk parity."""
    rm = _returns_df(n=3, bars=400, seed=20)

    w_rb = allocate_weights(rm, "risk_budget")
    cov = rm.dropna().cov().to_numpy()
    w_rp = risk_parity(cov)

    assert abs(w_rb.sum() - 1.0) < 1e-6
    np.testing.assert_allclose(
        w_rb, w_rp, atol=0.02,
        err_msg="Default risk_budget in allocate_weights should ≈ risk_parity",
    )


# ---------------------------------------------------------------------------
# Test 6 — every method survives a degenerate book (no NaN / blow-up weights)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_all_methods_handle_degenerate_covariance(method: str) -> None:
    """A zero-variance book (constant returns → zero covariance, undefined
    correlation) must never yield NaN or blow-up weights: every allocation method
    returns a finite long-only simplex."""
    rm = pd.DataFrame(
        np.full((300, 3), 0.001),
        index=pd.bdate_range("2020-01-01", periods=300),
        columns=["a", "b", "c"],
    )
    w = allocate_weights(rm, method)
    assert w.shape == (3,)
    assert np.all(np.isfinite(w)), f"{method} produced non-finite weights"
    assert abs(w.sum() - 1.0) < 1e-6, f"{method} weights sum to {w.sum()}, not 1"
    assert (w >= -1e-10).all(), f"{method} produced a negative weight"


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_single_asset(method: str) -> None:
    """A one-strategy book allocates the whole weight to it without dividing by zero."""
    rm = pd.DataFrame(
        0.01 * np.random.default_rng(0).standard_normal((300, 1)),
        index=pd.bdate_range("2020-01-01", periods=300),
        columns=["only"],
    )
    w = allocate_weights(rm, method)
    assert w.shape == (1,)
    assert np.isfinite(w).all()
    assert abs(w.sum() - 1.0) < 1e-6
