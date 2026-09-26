"""Tests for the Pezier & White (2006) skew/kurtosis-adjusted Sharpe.

The adjusted Sharpe (ASR) applies the P&W correction to the per-period Sharpe::

    ASR = SR * (1 + (S/6)*SR - (K_excess/24)*SR**2)

then annualises by ``sqrt(periods_per_year)`` (house convention, ddof=1 std).
``S`` is Fisher-Pearson skewness, ``K_excess`` is *excess* kurtosis (normal = 0),
both via ``rigor.metrics.compute_skewness`` / ``compute_kurtosis``.

Behavioural contract under test:
  * ASR ≈ SR for ~normal returns,
  * ASR < SR for negatively-skewed / fat-tailed returns,
  * ASR > SR for positively-skewed returns,
  * deterministic, and edge cases (zero variance, tiny sample) never crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigor.metrics import compute_kurtosis, compute_sharpe, compute_skewness
from rigor.validation.overfit import pezier_white_adjusted_sharpe

_PPY = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asr(returns, ppy: int = _PPY) -> float:
    return pezier_white_adjusted_sharpe(returns, ppy)


def _plain_sharpe(returns, ppy: int = _PPY) -> float:
    return compute_sharpe(returns, ppy)


# ---------------------------------------------------------------------------
# 1.  Near-normal returns: ASR ≈ SR
# ---------------------------------------------------------------------------

def test_asr_approx_sharpe_for_normal_returns():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 20000)  # large, ~symmetric, ~mesokurtic
    sr = _plain_sharpe(r)
    asr = _asr(r)
    # Sample skew/kurt are tiny but non-zero, so allow a small relative gap.
    assert asr == pytest.approx(sr, rel=0.03)
    assert abs(compute_skewness(r)) < 0.1
    assert abs(compute_kurtosis(r)) < 0.1


# ---------------------------------------------------------------------------
# 2.  Negative skew / fat tails: ASR < SR
# ---------------------------------------------------------------------------

def test_asr_below_sharpe_for_negative_skew():
    # Smooth small gains punctuated by rare large losses -> strong negative skew,
    # while keeping a net-positive mean (positive-expectancy strategy).
    rng = np.random.default_rng(1)
    n = 5000
    r = np.full(n, 0.001)
    crash_idx = rng.choice(n, size=n // 100, replace=False)  # ~1% crash days
    r[crash_idx] = -0.03
    assert compute_skewness(r) < 0  # construction sanity check
    sr = _plain_sharpe(r)
    asr = _asr(r)
    assert sr > 0  # positive expectancy strategy
    assert asr < sr  # penalised for the left-tail crash risk


def test_asr_below_sharpe_for_fat_tails():
    # Symmetric but heavy-tailed (excess kurtosis > 0) -> still penalised.
    rng = np.random.default_rng(2)
    r = rng.standard_t(df=3, size=20000) * 0.005 + 0.0005
    assert compute_kurtosis(r) > 0.5  # genuinely fat-tailed
    sr = _plain_sharpe(r)
    asr = _asr(r)
    assert asr < sr


# ---------------------------------------------------------------------------
# 3.  Positive skew: ASR > SR
# ---------------------------------------------------------------------------

def test_asr_above_sharpe_for_positive_skew():
    # Small steady costs with rare large gains -> positive skew, lottery-like.
    rng = np.random.default_rng(3)
    n = 5000
    r = np.full(n, -0.0008)
    jackpot_idx = rng.choice(n, size=n // 50, replace=False)
    r[jackpot_idx] = 0.06
    assert compute_skewness(r) > 0  # construction sanity check
    sr = _plain_sharpe(r)
    asr = _asr(r)
    assert sr > 0
    assert asr > sr  # rewarded for the right-tail convexity


# ---------------------------------------------------------------------------
# 4.  Determinism
# ---------------------------------------------------------------------------

def test_deterministic():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0004, 0.01, 1000)
    assert _asr(r) == _asr(r)
    assert _asr(r.copy()) == _asr(r)


# ---------------------------------------------------------------------------
# 5.  Annualisation convention matches compute_sharpe
# ---------------------------------------------------------------------------

def test_annualisation_scales_with_sqrt_ppy():
    # For ~normal returns ASR ≈ SR, and both scale by sqrt(ppy). Check the ratio
    # between two annualisation factors is sqrt(ppy_a / ppy_b).
    rng = np.random.default_rng(11)
    r = rng.normal(0.0006, 0.01, 20000)
    asr_252 = _asr(r, 252)
    asr_52 = _asr(r, 52)
    assert asr_252 / asr_52 == pytest.approx(np.sqrt(252 / 52), rel=1e-9)


def test_matches_manual_formula():
    rng = np.random.default_rng(13)
    r = rng.normal(0.0005, 0.01, 3000)
    sd = r.std(ddof=1)
    sr_pp = r.mean() / sd
    s = compute_skewness(r)
    k_excess = compute_kurtosis(r)
    expected = sr_pp * (1 + (s / 6) * sr_pp - (k_excess / 24) * sr_pp**2) * np.sqrt(252)
    assert _asr(r) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 6.  Guarded edge cases never crash
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_variance(self):
        r = np.full(500, 0.001)  # constant -> std == 0
        assert _asr(r) == 0.0

    def test_all_zeros(self):
        assert _asr(np.zeros(500)) == 0.0

    def test_empty(self):
        assert _asr(np.array([])) == 0.0

    def test_single_obs(self):
        assert _asr(np.array([0.01])) == 0.0

    def test_two_obs(self):
        # Below the 3-obs floor -> guarded to 0.0, no crash.
        assert _asr(np.array([0.01, -0.02])) == 0.0

    def test_all_nan(self):
        assert _asr(np.array([np.nan, np.nan, np.nan])) == 0.0

    def test_nan_dropped(self):
        clean = np.array([0.01, -0.005, 0.02, 0.003, -0.007, 0.011])
        with_nan = np.array([0.01, np.nan, -0.005, 0.02, 0.003, np.nan, -0.007, 0.011])
        assert _asr(with_nan) == pytest.approx(_asr(clean), rel=1e-12)

    def test_list_input(self):
        # Array-like (list) input must work like a numpy array.
        r = [0.01, -0.005, 0.02, 0.003, -0.007, 0.011, 0.004]
        assert _asr(r) == _asr(np.asarray(r, dtype=np.float64))

    def test_returns_float(self):
        rng = np.random.default_rng(5)
        out = _asr(rng.normal(0.0005, 0.01, 500))
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# 7.  Surfaced in the validation AO metrics dict
# ---------------------------------------------------------------------------

def test_wired_into_validation_overfit_block():
    import pandas as pd

    from rigor.engine import result_from_returns
    from rigor.validation import validate

    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    r = pd.Series(rng.normal(0.0006, 0.01, 3000), index=idx)
    rep = validate(result_from_returns(r), n_trials=1)
    assert "pezier_white_adjusted_sharpe" in rep["overfit"]
    val = rep["overfit"]["pezier_white_adjusted_sharpe"]
    assert isinstance(val, float)
    assert np.isfinite(val)
    # For these ~normal returns it should track the plain Sharpe closely.
    sr = _plain_sharpe(r.to_numpy())
    assert val == pytest.approx(sr, rel=0.05)
    # Adding the metric must not change the promotion decision (additive only).
    assert rep["verdict"]["verdict"] in ("PROMOTE", "CONDITIONAL", "REJECT")
