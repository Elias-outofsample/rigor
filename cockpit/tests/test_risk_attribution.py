"""Tests for Euler marginal risk attribution and ACF leakage profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_engine.risk import (
    autocorrelation_profile,
    marginal_risk_contributions,
    risk_concentration_hhi,
    risk_hhi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX = pd.bdate_range("2016-01-01", periods=1000)


def _matrix(vols: list[float], seed: int = 0) -> pd.DataFrame:
    """Build an uncorrelated returns_matrix with specified per-leg annualised vols."""
    rng = np.random.default_rng(seed)
    daily_vols = [v / np.sqrt(252) for v in vols]
    data = {
        f"leg{i}": rng.standard_normal(1000) * dv
        for i, dv in enumerate(daily_vols)
    }
    return pd.DataFrame(data, index=_IDX)


# ---------------------------------------------------------------------------
# 1. High-vol leg dominates risk
# ---------------------------------------------------------------------------

class TestMarginalRiskHighVol:
    """One high-vol leg should carry more risk than its weight implies."""

    def setup_method(self):
        # leg0=low vol (0.05), leg1=low vol (0.05), leg2=high vol (0.30)
        rm = _matrix([0.05, 0.05, 0.30], seed=42)
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        self.result = marginal_risk_contributions(rm, w)

    def test_pct_risk_sums_to_one(self):
        pct = self.result["pct_risk"]
        assert abs(sum(pct) - 1.0) < 1e-6, f"pct_risk sum = {sum(pct)}"

    def test_risk_contribution_sums_to_portfolio_vol(self):
        rc_sum = sum(self.result["risk_contribution"])
        pv = self.result["portfolio_vol"]
        assert abs(rc_sum - pv) < 1e-6, f"RC sum {rc_sum} != portfolio_vol {pv}"

    def test_high_vol_leg_carries_more_risk_than_weight(self):
        pct_risk = self.result["pct_risk"]
        pct_weight = self.result["pct_weight"]
        # leg2 (index 2) is the high-vol leg
        assert pct_risk[2] > pct_weight[2], (
            f"leg2 pct_risk {pct_risk[2]:.3f} not > pct_weight {pct_weight[2]:.3f}"
        )

    def test_concentration_flag_set_for_high_vol_leg(self):
        flags = self.result["concentration_flag"]
        assert flags[2] is True, f"concentration_flag[2] should be True, got {flags}"

    def test_concentration_flag_false_for_low_vol_legs(self):
        flags = self.result["concentration_flag"]
        assert flags[0] is False and flags[1] is False, (
            f"Low-vol legs should not be flagged: {flags}"
        )

    def test_portfolio_vol_positive(self):
        assert self.result["portfolio_vol"] > 0.0

    def test_assets_returned(self):
        assert self.result["assets"] == ["leg0", "leg1", "leg2"]


# ---------------------------------------------------------------------------
# 2. Equal-vol uncorrelated legs with equal weights -> equal risk shares
# ---------------------------------------------------------------------------

class TestEqualRiskEqualVol:
    """Equal weights + equal vol + uncorrelated -> pct_risk ≈ 1/n each."""

    def setup_method(self):
        n = 4
        rm = _matrix([0.15] * n, seed=7)
        w = np.ones(n) / n
        self.result = marginal_risk_contributions(rm, w)
        self.n = n

    def test_pct_risk_approximately_equal(self):
        pct = self.result["pct_risk"]
        expected = 1.0 / self.n
        for i, p in enumerate(pct):
            assert abs(p - expected) < 0.05, (
                f"leg{i} pct_risk {p:.4f} != {expected:.4f}"
            )

    def test_no_concentration_flags(self):
        flags = self.result["concentration_flag"]
        assert not any(flags), f"No flags expected for equal legs: {flags}"

    def test_pct_risk_sums_to_one(self):
        assert abs(sum(self.result["pct_risk"]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. risk_hhi and risk_concentration_hhi
# ---------------------------------------------------------------------------

class TestHHI:
    def test_risk_hhi_in_range(self):
        n = 3
        rm = _matrix([0.10, 0.10, 0.10], seed=1)
        w = np.ones(n) / n
        h = risk_hhi(rm, w)
        assert 1 / n - 0.05 <= h <= 1.0, f"risk_hhi {h} out of [1/n, 1]"

    def test_risk_hhi_approx_equal_for_equal_legs(self):
        n = 4
        rm = _matrix([0.12] * n, seed=2)
        w = np.ones(n) / n
        h = risk_hhi(rm, w)
        assert abs(h - 1 / n) < 0.05, f"risk_hhi {h} should be near 1/{n}={1/n:.3f}"

    def test_risk_concentration_hhi_equal_weights(self):
        n = 5
        w = np.ones(n) / n
        h = risk_concentration_hhi(w)
        assert abs(h - 1 / n) < 1e-9

    def test_risk_concentration_hhi_fully_concentrated(self):
        w = np.array([1.0, 0.0, 0.0])
        h = risk_concentration_hhi(w)
        assert abs(h - 1.0) < 1e-9

    def test_risk_hhi_greater_for_concentrated(self):
        n = 3
        rm = _matrix([0.05, 0.05, 0.30], seed=5)
        w_eq = np.ones(n) / n
        h_conc = risk_hhi(rm, w_eq)
        h_eq = risk_hhi(_matrix([0.15] * n, seed=6), w_eq)
        assert h_conc > h_eq - 0.01, (
            f"Concentrated HHI {h_conc:.4f} should exceed equal HHI {h_eq:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. autocorrelation_profile
# ---------------------------------------------------------------------------

class TestAutocorrelationProfile:
    def _ar1(self, phi: float = 0.5, n: int = 500, seed: int = 0) -> pd.Series:
        """Simulate AR(1) with given phi coefficient."""
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal(n) * 0.01
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        return pd.Series(x, index=_IDX[:n])

    def test_ar1_leakage_flag_true(self):
        series = self._ar1(phi=0.5)
        result = autocorrelation_profile(series)
        assert result["leakage_flag"] is True, (
            f"AR(1) phi=0.5 should trigger leakage_flag; acf[0]={result['acf'][0]:.3f}"
        )

    def test_ar1_lag1_acf_above_threshold(self):
        series = self._ar1(phi=0.5)
        result = autocorrelation_profile(series)
        assert result["acf"][0] > 0.20, (
            f"lag-1 acf {result['acf'][0]:.3f} should be > 0.20 for AR(1) phi=0.5"
        )

    def test_white_noise_leakage_flag_false(self):
        rng = np.random.default_rng(99)
        noise = pd.Series(rng.standard_normal(500) * 0.01, index=_IDX[:500])
        result = autocorrelation_profile(noise)
        assert result["leakage_flag"] is False, (
            f"White noise should not trigger leakage_flag; acf[0]={result['acf'][0]:.3f}"
        )

    def test_lags_length(self):
        series = self._ar1()
        result = autocorrelation_profile(series, max_lag=5)
        assert result["lags"] == [1, 2, 3, 4, 5]
        assert len(result["acf"]) == 5

    def test_ljung_box_p_present(self):
        series = self._ar1()
        result = autocorrelation_profile(series)
        # scipy is available in the rigor install
        assert result["ljung_box_p"] is not None
        assert 0.0 <= result["ljung_box_p"] <= 1.0

    def test_ljung_box_low_p_for_ar1(self):
        """AR(1) series should yield a very small Ljung-Box p (reject H0 of white noise)."""
        series = self._ar1(phi=0.6, n=500)
        result = autocorrelation_profile(series)
        if result["ljung_box_p"] is not None:
            assert result["ljung_box_p"] < 0.05, (
                f"Ljung-Box p={result['ljung_box_p']:.4f} should be < 0.05 for AR(1)"
            )

    def test_numpy_array_input(self):
        """Function should accept a plain numpy array."""
        arr = np.array([0.01, -0.02, 0.01, -0.01] * 50)
        result = autocorrelation_profile(arr)
        assert "lags" in result and "acf" in result


# ---------------------------------------------------------------------------
# 5. Degenerate inputs do not raise
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    def test_single_leg_no_raise(self):
        rng = np.random.default_rng(0)
        rm = pd.DataFrame({"only": rng.standard_normal(100) * 0.01}, index=_IDX[:100])
        w = np.array([1.0])
        result = marginal_risk_contributions(rm, w)
        assert isinstance(result, dict)
        assert "portfolio_vol" in result

    def test_few_observations_no_raise(self):
        """Less than 20 obs: marginal_risk_contributions should not raise."""
        rng = np.random.default_rng(1)
        rm = pd.DataFrame(
            {"a": rng.standard_normal(10) * 0.01, "b": rng.standard_normal(10) * 0.01},
            index=_IDX[:10],
        )
        w = np.array([0.5, 0.5])
        result = marginal_risk_contributions(rm, w)
        assert isinstance(result, dict)

    def test_short_series_acf_no_raise(self):
        """Series with < 20 obs: should return zeros and no leakage flag."""
        tiny = pd.Series([0.01, -0.01, 0.005, -0.005])
        result = autocorrelation_profile(tiny)
        assert result["leakage_flag"] is False
        assert result["ljung_box_p"] is None
        assert all(v == 0.0 for v in result["acf"])

    def test_wrong_weights_length_raises_value_error(self):
        rng = np.random.default_rng(0)
        rm = pd.DataFrame(
            {"a": rng.standard_normal(100) * 0.01, "b": rng.standard_normal(100) * 0.01},
            index=_IDX[:100],
        )
        w = np.array([0.5, 0.3, 0.2])  # wrong length
        with pytest.raises(ValueError, match="weights length"):
            marginal_risk_contributions(rm, w)
