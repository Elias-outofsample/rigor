"""Tests for the new Gaussian VaR and Expected Excess Return additions.

Covers:
  A. compute_expected_excess_return in rigor.metrics
  B. gaussian_var in rigor.analysis.risk
  C. compute_core_metrics now contains ev_excess_annual and var_normal_95
  D. Existing metrics are unchanged when risk_free_annual=0 (byte-identical)
  E. The inline _gaussian_var_inline (via compute_core_metrics) agrees with
     the canonical rigor.analysis.risk.gaussian_var for the same input
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from rigor import metrics as m
from rigor.analysis.risk import gaussian_var, value_at_risk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PPY = 252


def _make_returns(n: int = 500, seed: int = 42, mu: float = 0.0004,
                  sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


# ---------------------------------------------------------------------------
# A. compute_expected_excess_return
# ---------------------------------------------------------------------------

class TestComputeExpectedExcessReturn:

    def test_ev_excess_annual_equals_ev_annual_minus_rf(self):
        """Core definition: ev_excess_annual == ev_annual - risk_free_annual."""
        r = _make_returns()
        rf = 0.04  # 4% annualised
        result = m.compute_expected_excess_return(r, _PPY, risk_free_annual=rf)
        ev_result = m.compute_expected_value(r, _PPY)
        assert result["ev_excess_annual"] == pytest.approx(
            ev_result["ev_annual"] - rf, rel=1e-10
        )

    def test_ev_excess_per_period_equals_excess_annual_over_ppy(self):
        r = _make_returns()
        rf = 0.03
        result = m.compute_expected_excess_return(r, _PPY, risk_free_annual=rf)
        assert result["ev_excess"] == pytest.approx(
            result["ev_excess_annual"] / _PPY, rel=1e-10
        )

    def test_zero_rf_gives_ev_excess_equal_ev(self):
        """At rf=0 the excess return equals the total return (default behaviour)."""
        r = _make_returns()
        result = m.compute_expected_excess_return(r, _PPY, risk_free_annual=0.0)
        ev_result = m.compute_expected_value(r, _PPY)
        assert result["ev_excess_annual"] == pytest.approx(ev_result["ev_annual"], rel=1e-10)

    def test_empty_returns_gives_zero(self):
        result = m.compute_expected_excess_return(np.array([]), _PPY)
        assert result["ev_excess"] == 0.0
        assert result["ev_excess_annual"] == 0.0

    def test_all_nan_gives_zero(self):
        result = m.compute_expected_excess_return(
            np.array([float("nan"), float("nan")]), _PPY
        )
        assert result["ev_excess"] == 0.0
        assert result["ev_excess_annual"] == 0.0

    def test_positive_alpha_gives_positive_excess(self):
        """A strategy earning well above rf should show positive ev_excess_annual."""
        r = np.full(500, 0.002)  # 0.2%/day = ~50% annualised — well above any rf
        result = m.compute_expected_excess_return(r, _PPY, risk_free_annual=0.05)
        assert result["ev_excess_annual"] > 0.0

    def test_underperforming_strategy_gives_negative_excess(self):
        """A strategy earning less than rf should show negative ev_excess_annual."""
        r = np.full(500, 0.00001)  # ~0.25% annualised — below 5% rf
        result = m.compute_expected_excess_return(r, _PPY, risk_free_annual=0.05)
        assert result["ev_excess_annual"] < 0.0

    def test_different_ppy(self):
        """Weekly series (ppy=52): ev_excess_annual = weekly_mean*52 - rf."""
        r = np.array([0.01, 0.02, -0.005, 0.015, 0.003])
        rf = 0.02
        result = m.compute_expected_excess_return(r, periods_per_year=52,
                                                   risk_free_annual=rf)
        expected = r.mean() * 52 - rf
        assert result["ev_excess_annual"] == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# B. gaussian_var in rigor.analysis.risk
# ---------------------------------------------------------------------------

class TestGaussianVar:

    def test_known_formula_matches(self):
        """gaussian_var must equal μ + z_alpha*σ (negative for typical alpha<0.5)."""
        r = _make_returns()
        alpha = 0.05
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        z = statistics.NormalDist().inv_cdf(alpha)
        expected = mu + z * sigma
        assert gaussian_var(r, alpha) == pytest.approx(expected, rel=1e-10)

    def test_returns_negative_number_for_typical_series(self):
        """At alpha=0.05 a zero-mean return series should give a negative VaR."""
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, 1000)
        assert gaussian_var(r, 0.05) < 0.0

    def test_sign_convention_matches_historical_var(self):
        """Both gaussian_var and value_at_risk should return negative numbers
        for a typical daily return distribution."""
        r = _make_returns()
        assert gaussian_var(r, 0.05) < 0.0
        assert value_at_risk(r, 0.05) < 0.0

    def test_larger_alpha_gives_less_negative_var(self):
        """alpha=0.10 cuts at a less extreme quantile than alpha=0.05 → closer to 0."""
        r = _make_returns()
        var_05 = gaussian_var(r, 0.05)
        var_10 = gaussian_var(r, 0.10)
        assert var_10 > var_05  # both negative; 10% quantile is less negative

    def test_zero_variance_returns_zero(self):
        """Constant return series has σ=0 → VaR equals μ (no spread)."""
        r = np.full(100, 0.001)
        result = gaussian_var(r, 0.05)
        # μ + z*0 = μ; for mu=0.001 this is positive (no downside)
        assert result == pytest.approx(0.001, rel=1e-9)

    def test_empty_returns_zero(self):
        assert gaussian_var(np.array([]), 0.05) == 0.0

    def test_single_element_returns_zero(self):
        assert gaussian_var(np.array([0.01]), 0.05) == 0.0

    def test_pandas_series_accepted(self):
        """Function must accept a pandas Series (via _arr helper)."""
        import pandas as pd
        r_arr = _make_returns()
        r_ser = pd.Series(r_arr)
        assert gaussian_var(r_ser, 0.05) == pytest.approx(gaussian_var(r_arr, 0.05),
                                                           rel=1e-10)

    def test_higher_vol_gives_more_negative_var(self):
        """Doubling σ should roughly double the magnitude of the tail loss."""
        rng = np.random.default_rng(7)
        r_low = rng.normal(0.0, 0.01, 2000)
        r_high = rng.normal(0.0, 0.02, 2000)
        # Both zero-mean; higher vol → more negative VaR
        assert gaussian_var(r_high, 0.05) < gaussian_var(r_low, 0.05)


# ---------------------------------------------------------------------------
# C. compute_core_metrics: new keys present and correct
# ---------------------------------------------------------------------------

class TestCoreMetricsNewKeys:

    def test_ev_excess_annual_key_present(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert "ev_excess_annual" in out

    def test_var_normal_95_key_present(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert "var_normal_95" in out

    def test_ev_excess_annual_default_equals_ev_annual(self):
        """At default rf=0, ev_excess_annual must be byte-identical to ev_annual."""
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert out["ev_excess_annual"] == out["ev_annual"]

    def test_ev_excess_annual_nonzero_rf(self):
        """With rf>0, ev_excess_annual < ev_annual."""
        r = _make_returns()
        out = m.compute_core_metrics(r, risk_free_annual=0.04)
        assert out["ev_excess_annual"] == pytest.approx(out["ev_annual"] - 0.04,
                                                        rel=1e-10)

    def test_var_normal_95_matches_formula(self):
        """var_normal_95 in core metrics must agree with the standalone gaussian_var."""
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert out["var_normal_95"] == pytest.approx(gaussian_var(r, 0.05), rel=1e-10)

    def test_var_normal_95_is_negative(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert out["var_normal_95"] < 0.0

    def test_empty_returns_new_keys(self):
        out = m.compute_core_metrics(np.array([]))
        assert out["ev_excess_annual"] == 0.0
        assert out["var_normal_95"] == 0.0


# ---------------------------------------------------------------------------
# D. Backward-compatibility: existing keys byte-identical at rf=0
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    EXISTING_KEYS = (
        "n_obs", "sharpe", "sortino", "calmar", "cagr", "volatility",
        "max_drawdown", "total_return", "win_rate", "ev", "ev_annual",
    )

    def test_existing_keys_present(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        for key in self.EXISTING_KEYS:
            assert key in out, f"Missing existing key: {key}"

    def test_existing_values_byte_identical_at_default_rf(self):
        """The old call compute_core_metrics(r) must produce the same values for
        all pre-existing keys — no numerical drift from new code paths."""
        r = _make_returns()
        out_new = m.compute_core_metrics(r)  # new signature, default rf=0
        # Reference values from standalone functions (unchanged functions)
        assert out_new["sharpe"] == pytest.approx(m.compute_sharpe(r), rel=1e-10)
        assert out_new["cagr"] == pytest.approx(m.compute_cagr(r), rel=1e-10)
        assert out_new["volatility"] == pytest.approx(m.compute_volatility(r), rel=1e-10)
        assert out_new["max_drawdown"] == pytest.approx(m.compute_max_drawdown(r),
                                                         rel=1e-10)
        assert out_new["sortino"] == pytest.approx(m.compute_sortino(r), rel=1e-10)
        assert out_new["calmar"] == pytest.approx(m.compute_calmar(r), rel=1e-10)
        assert out_new["win_rate"] == pytest.approx(m.compute_win_rate(r), rel=1e-10)
        ev_ref = m.compute_expected_value(r)
        assert out_new["ev"] == pytest.approx(ev_ref["ev"], rel=1e-10)
        assert out_new["ev_annual"] == pytest.approx(ev_ref["ev_annual"], rel=1e-10)

    def test_deterministic(self):
        """Two calls with same input must be identical (no RNG introduced)."""
        r = _make_returns()
        assert m.compute_core_metrics(r) == m.compute_core_metrics(r)

    def test_rf_param_does_not_affect_sharpe(self):
        """risk_free_annual only touches ev_excess_annual; Sharpe is unaffected
        (Sharpe has its own risk_free arg — these are separate knobs)."""
        r = _make_returns()
        out0 = m.compute_core_metrics(r, risk_free_annual=0.0)
        out4 = m.compute_core_metrics(r, risk_free_annual=0.04)
        assert out0["sharpe"] == out4["sharpe"]
        assert out0["cagr"] == out4["cagr"]
        assert out0["volatility"] == out4["volatility"]
