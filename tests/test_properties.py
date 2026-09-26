"""Property-based tests using hypothesis.

Invariants that must hold for *any* finite returns array, regardless of shape or sign:
  P1  max_drawdown is non-positive.
  P2  volatility is non-negative and finite.
  P3  ES (CVaR) <= VaR + epsilon.
  P4  omega, ulcer_index, pain_index, cdar are all non-negative.
  P5  Sharpe sign tracks mean sign for arrays with positive mean and positive std.
  P6  Sharpe is scale-invariant: compute_sharpe(c*r) ≈ compute_sharpe(r).
  P7  PSR and DSR are always in [0, 1].
  P8  verdict.score in [0, 100] and verdict.verdict is one of the four labels.
"""
from __future__ import annotations

import numpy as np
import pytest

# hypothesis ships in the [dev] extra; skip these property tests cleanly on a
# base install so `pytest` never hard-fails for anyone who hasn't installed it.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from rigor.analysis import risk
from rigor.analysis.verdict import compute_strategy_verdict
from rigor.metrics import compute_max_drawdown, compute_sharpe, compute_volatility
from rigor.validation.overfit import deflated_sharpe_ratio, probabilistic_sharpe_ratio

# ---------------------------------------------------------------------------
# Shared strategy: realistic daily-return arrays, 40–400 observations.
# ---------------------------------------------------------------------------
returns_arrays = hnp.arrays(
    np.float64,
    st.integers(40, 400),
    elements=st.floats(-0.2, 0.2, allow_nan=False, allow_infinity=False),
)

_EPS = 1e-9
_VALID_VERDICTS = {"ROBUST", "MODERATE", "FRAGILE", "OVERFIT"}


# ---------------------------------------------------------------------------
# P1 — max drawdown is non-positive
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays)
def test_p1_drawdown_nonpositive(r: np.ndarray) -> None:
    """compute_max_drawdown(r) <= 1e-9 for all finite returns."""
    assert compute_max_drawdown(r) <= _EPS


# ---------------------------------------------------------------------------
# P2 — volatility is non-negative and finite
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays)
def test_p2_volatility_nonneg_finite(r: np.ndarray) -> None:
    """compute_volatility(r) >= 0 and is finite."""
    vol = compute_volatility(r)
    assert vol >= 0.0
    assert np.isfinite(vol)


# ---------------------------------------------------------------------------
# P3 — ES (CVaR) <= VaR + epsilon
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays)
def test_p3_es_le_var(r: np.ndarray) -> None:
    """conditional_var(r) <= value_at_risk(r) + 1e-9 (ES is at least as bad as VaR)."""
    es = risk.conditional_var(r)
    var = risk.value_at_risk(r)
    assert es <= var + _EPS


# ---------------------------------------------------------------------------
# P4 — omega, ulcer_index, pain_index, cdar are non-negative
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays)
def test_p4_risk_metrics_nonneg(r: np.ndarray) -> None:
    """omega, ulcer_index, pain_index, cdar are all >= 0."""
    assert risk.omega(r) >= 0.0
    assert risk.ulcer_index(r) >= 0.0
    assert risk.pain_index(r) >= 0.0
    assert risk.cdar(r) >= 0.0


# ---------------------------------------------------------------------------
# P5 — Sharpe sign tracks mean sign (positive drift, positive std)
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays)
def test_p5_sharpe_sign_tracks_mean(r: np.ndarray) -> None:
    """When r.mean() > eps and r.std() > eps, sign(compute_sharpe(r)) == sign(r.mean())."""
    from hypothesis import assume

    assume(float(r.mean()) > _EPS)
    assume(float(r.std(ddof=1)) > _EPS)
    sharpe = compute_sharpe(r)
    assert sharpe > 0.0


# ---------------------------------------------------------------------------
# P6 — Sharpe is scale-invariant
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(r=returns_arrays, c=st.sampled_from([0.5, 2.0, 10.0]))
def test_p6_sharpe_scale_invariant(r: np.ndarray, c: float) -> None:
    """compute_sharpe(c*r) ≈ compute_sharpe(r) (within 1e-6 relative) when std > eps."""
    from hypothesis import assume

    assume(float(r.std(ddof=1)) > _EPS)
    # Guard against values that would overflow float64 after scaling
    assume(float(np.abs(r).max()) * c <= 0.5)

    base = compute_sharpe(r)
    scaled = compute_sharpe(c * r)

    if abs(base) < _EPS:
        # Both should be near-zero
        assert abs(scaled) < 1e-4
    else:
        rel_err = abs(scaled - base) / abs(base)
        assert rel_err < 1e-6


# ---------------------------------------------------------------------------
# P7 — PSR and DSR are always in [0, 1]
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(
    sharpe=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    benchmark=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    n_obs=st.integers(10, 5000),
    skew=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    kurt=st.floats(-1.9, 10.0, allow_nan=False, allow_infinity=False),
    n_trials=st.integers(1, 500),
)
def test_p7_psr_dsr_in_unit_interval(
    sharpe: float,
    benchmark: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
) -> None:
    """probabilistic_sharpe_ratio and deflated_sharpe_ratio are always in [0, 1]."""
    psr = probabilistic_sharpe_ratio(sharpe, benchmark, n_obs, skew, kurt)
    dsr = deflated_sharpe_ratio(sharpe, n_trials, n_obs, skew, kurt)
    assert 0.0 <= psr <= 1.0, f"PSR={psr} out of [0,1]"
    assert 0.0 <= dsr <= 1.0, f"DSR={dsr} out of [0,1]"


# ---------------------------------------------------------------------------
# P8 — verdict.score in [0, 100] and verdict in the four labels
# ---------------------------------------------------------------------------
@settings(max_examples=50, deadline=None)
@given(
    sharpe=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    n_obs=st.integers(10, 5000),
    cagr=st.floats(-0.5, 2.0, allow_nan=False, allow_infinity=False),
    mdd=st.floats(-1.0, 0.0, allow_nan=False, allow_infinity=False),
)
def test_p8_verdict_score_and_label(
    sharpe: float, n_obs: int, cagr: float, mdd: float
) -> None:
    """verdict.score in [0, 100] and verdict in {ROBUST, MODERATE, FRAGILE, OVERFIT}."""
    metrics = {
        "sharpe": sharpe,
        "n_obs": n_obs,
        "cagr": cagr,
        "max_drawdown": mdd,
        "periods_per_year": 252,
    }
    result = compute_strategy_verdict(metrics)
    assert 0.0 <= result.score <= 100.0, f"score={result.score} out of [0, 100]"
    assert result.verdict in _VALID_VERDICTS, f"unexpected verdict: {result.verdict!r}"
