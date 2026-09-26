"""Tests for the strategy-longevity heuristics (rigor.analysis.survival).

These cover the three honest priors — crowding, life-cycle phase, forward
survival — with known-answer cases (a clearly-crowded vs clearly-niche input
scores accordingly; a young rising edge vs a decayed one labels accordingly) and
degenerate / missing-input cases that must return a neutral result without
raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.analysis.survival import compute_crowding, compute_lifecycle, compute_survival

# ---------------------------------------------------------------------------
# Crowding
# ---------------------------------------------------------------------------

def test_crowding_high_vs_low_ordering():
    """A textbook, scalable factor bet must score far above a niche idiosyncratic edge."""
    crowded = compute_crowding(
        factor_betas={"momentum": 1.2, "value": 0.9},
        family="Momentum",
        capacity_usd=2_000_000_000,  # ~$2B, very scalable
        annual_turnover=1.0,         # low turnover
    )
    niche = compute_crowding(
        factor_betas={"momentum": 0.02},  # no real factor exposure
        family="Seasonality",
        capacity_usd=5_000_000,           # tiny capacity
        annual_turnover=30.0,             # frantic turnover
    )
    assert crowded["verdict"] == "HIGH"
    assert niche["verdict"] == "LOW"
    assert crowded["score"] > niche["score"]
    assert crowded["score"] >= 70.0
    assert niche["score"] < 45.0
    # Components are reported and bounded in [0, 1].
    for comp in crowded["components"].values():
        assert 0.0 <= comp <= 1.0


def test_crowding_factor_popularity_caps_at_one():
    """A huge factor beta saturates factor_popularity at 1.0 (clipped)."""
    res = compute_crowding({"value": 5.0}, "Factor", None, None)
    assert res["components"]["factor_popularity"] == 1.0


def test_crowding_degenerate_inputs_are_neutral():
    """All-None input must not raise and must land in a defined band."""
    res = compute_crowding(None, None, None, None)
    assert res["verdict"] in {"LOW", "MODERATE", "HIGH"}
    assert 0.0 <= res["score"] <= 100.0
    # No factor data and no family -> factor_popularity 0, recipe_popularity 0.5.
    assert res["components"]["factor_popularity"] == 0.0
    assert res["components"]["recipe_popularity"] == 0.5


def test_crowding_ignores_nonfinite_betas():
    """NaN/inf betas are skipped, not propagated into the score."""
    res = compute_crowding(
        {"momentum": float("nan"), "value": float("inf"), "size": None},
        "Trend", None, None,
    )
    assert res["components"]["factor_popularity"] == 0.0
    assert np.isfinite(res["score"])


# ---------------------------------------------------------------------------
# Life-cycle
# ---------------------------------------------------------------------------

def _series(daily_returns: np.ndarray) -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=len(daily_returns), freq="B")
    return pd.Series(daily_returns, index=idx)


def test_lifecycle_young_edge_is_emergence():
    """A short (<4y), non-falling history reads as EMERGENCE."""
    rng = np.random.default_rng(0)
    # Gently rising drift, low noise -> recent >= early (not falling); young -> EMERGENCE.
    drift = np.linspace(0.0004, 0.0010, 400)  # ~1.6y of daily bars
    r = _series(drift + 0.003 * rng.standard_normal(400))
    out = compute_lifecycle(r, periods_per_year=252)
    assert out["phase"] == "EMERGENCE"


def test_lifecycle_declining_edge_via_decay_slope():
    """A strongly-negative decay slope forces DECLINE regardless of length."""
    rng = np.random.default_rng(1)
    r = _series(0.0003 + 0.01 * rng.standard_normal(252 * 6))  # 6y so 'young' is False
    out = compute_lifecycle(r, periods_per_year=252, edge_decay_slope=-0.5)
    assert out["phase"] == "DECLINE"
    assert out["edge_decay_slope_per_yr"] == -0.5


def test_lifecycle_rising_recent_edge_is_exploitation():
    """Recent Sharpe materially above the early third reads as EXPLOITATION.

    6y so 'young' is False; low-noise weak-early / strong-late drift makes the
    last-2y Sharpe exceed the first-third Sharpe by well over the +0.2 threshold.
    """
    rng = np.random.default_rng(2)
    n = 252 * 6
    early = 0.0001 + 0.004 * rng.standard_normal(n // 2)    # weak early edge
    late = 0.0010 + 0.004 * rng.standard_normal(n - n // 2)  # strong recent edge
    r = _series(np.concatenate([early, late]))
    out = compute_lifecycle(r, periods_per_year=252)
    assert out["phase"] == "EXPLOITATION"


def test_lifecycle_insufficient_history():
    """A history shorter than one year (or ppy<=0) returns INSUFFICIENT_DATA."""
    short = compute_lifecycle(_series(np.zeros(50)), periods_per_year=252)
    assert short["phase"] == "INSUFFICIENT_DATA"
    bad_ppy = compute_lifecycle(_series(np.ones(500) * 0.001), periods_per_year=0)
    assert bad_ppy["phase"] == "INSUFFICIENT_DATA"


def test_lifecycle_accepts_plain_array():
    """A bare numpy array (no DatetimeIndex) must work, not raise."""
    out = compute_lifecycle(np.full(500, 0.001), periods_per_year=252)
    assert out["phase"] in {"EMERGENCE", "EXPLOITATION", "MATURITY", "DECLINE"}


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

def test_survival_durable_vs_at_risk_ordering():
    """A clean edge scores LIKELY_DURABLE; a flawed one scores AT_RISK below it."""
    durable = compute_survival(
        pbo=0.05, psr=0.95, edge_decay_slope=0.1, crash_ratio=1.2,
        crowding_score=20.0, lifecycle_phase="EXPLOITATION",
    )
    at_risk = compute_survival(
        pbo=0.85, psr=0.10, edge_decay_slope=-0.45, crash_ratio=0.1,
        crowding_score=90.0, lifecycle_phase="DECLINE",
    )
    assert durable["verdict"] == "LIKELY_DURABLE"
    assert at_risk["verdict"] == "AT_RISK"
    assert durable["probability"] > at_risk["probability"]
    assert 0.0 <= at_risk["probability"] <= 1.0


def test_survival_weakest_link_is_the_worst_component():
    """The reported weakest link is the lowest survival contribution."""
    out = compute_survival(
        pbo=0.05,            # not_overfit ~0.95
        psr=0.90,            # significant 0.90
        edge_decay_slope=-0.5,  # not_decaying 0.0 -> the weakest
        crash_ratio=1.0,
        crowding_score=10.0,
        lifecycle_phase="MATURITY",
    )
    assert out["weakest_link"] == "not_decaying"
    assert out["components"]["not_decaying"] == 0.0


def test_survival_geometric_mean_punishes_single_fatal_flaw():
    """One near-zero contribution drags the aggregate below a plain mean."""
    out = compute_survival(
        pbo=0.0, psr=1.0, edge_decay_slope=0.2, crash_ratio=2.0,
        crowding_score=0.0, lifecycle_phase="EXPLOITATION",
    )
    flawed = compute_survival(
        pbo=0.0, psr=1.0, edge_decay_slope=-0.5, crash_ratio=2.0,
        crowding_score=0.0, lifecycle_phase="EXPLOITATION",
    )
    assert flawed["probability"] < out["probability"]


def test_survival_no_inputs_is_neutral():
    """No usable input -> INSUFFICIENT_DATA, None probability, no raise."""
    out = compute_survival(None, None, None, None, None, None)
    assert out["verdict"] == "INSUFFICIENT_DATA"
    assert out["probability"] is None
    assert out["components"] == {}
    assert out["weakest_link"] is None


def test_survival_ignores_nonfinite_inputs():
    """NaN/inf inputs are dropped; only the finite ones contribute."""
    out = compute_survival(
        pbo=float("nan"), psr=0.8, edge_decay_slope=float("inf"),
        crash_ratio=None, crowding_score=30.0, lifecycle_phase=None,
    )
    assert set(out["components"]) == {"significant", "uncrowded"}
    assert out["probability"] is not None
