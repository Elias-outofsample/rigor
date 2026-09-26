"""Tests for the EV (Expected Value) metric added to rigor.metrics and rigor.report.

EV definition:
  ev        = mean(returns)  per period (arithmetic expectation per bar)
  ev_annual = mean(returns) * periods_per_year  (annualised arithmetic mean)

This complements CAGR (geometric mean): EV is the honest "expected return per year
if periods were independent" — the standard trading expectancy generalised to a
continuous return series.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigor import metrics as m
from rigor.report import build_payload, render_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PPY = 252


def _make_returns(n: int = 500, seed: int = 42, mu: float = 0.0004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, 0.01, n)


# ---------------------------------------------------------------------------
# 1.  compute_expected_value directly
# ---------------------------------------------------------------------------

class TestComputeExpectedValue:
    def test_ev_equals_mean(self):
        r = np.array([0.01, -0.005, 0.02, 0.003, -0.007])
        result = m.compute_expected_value(r)
        assert result["ev"] == pytest.approx(r.mean(), rel=1e-10)

    def test_ev_annual_equals_mean_times_ppy(self):
        r = np.array([0.01, -0.005, 0.02, 0.003, -0.007])
        result = m.compute_expected_value(r)
        assert result["ev_annual"] == pytest.approx(r.mean() * _PPY, rel=1e-10)

    def test_custom_ppy(self):
        r = np.array([0.01, 0.02, 0.03])
        result = m.compute_expected_value(r, periods_per_year=52)
        assert result["ev_annual"] == pytest.approx(r.mean() * 52, rel=1e-10)

    def test_empty_returns_gives_zero(self):
        result = m.compute_expected_value(np.array([]))
        assert result["ev"] == 0.0
        assert result["ev_annual"] == 0.0

    def test_all_nan_gives_zero(self):
        result = m.compute_expected_value(np.array([float("nan"), float("nan")]))
        assert result["ev"] == 0.0
        assert result["ev_annual"] == 0.0

    def test_nan_ignored_in_mixed_series(self):
        r_clean = np.array([0.01, -0.005, 0.02])
        r_with_nan = np.array([0.01, float("nan"), -0.005, 0.02])
        result_clean = m.compute_expected_value(r_clean)
        result_nan = m.compute_expected_value(r_with_nan)
        assert result_clean["ev"] == pytest.approx(result_nan["ev"], rel=1e-10)

    def test_does_not_raise_on_empty(self):
        # Must not raise, regardless of input shape
        m.compute_expected_value([])
        m.compute_expected_value(np.array([float("nan")]))


# ---------------------------------------------------------------------------
# 2.  compute_core_metrics includes ev and ev_annual; existing keys unchanged
# ---------------------------------------------------------------------------

class TestCoreMetricsContainsEV:
    def test_ev_keys_present(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert "ev" in out
        assert "ev_annual" in out

    def test_ev_values_correct(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert out["ev"] == pytest.approx(r.mean(), rel=1e-10)
        assert out["ev_annual"] == pytest.approx(r.mean() * _PPY, rel=1e-10)

    def test_existing_keys_unchanged(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        for key in ("sharpe", "cagr", "max_drawdown", "volatility",
                    "sortino", "calmar", "total_return", "win_rate", "n_obs"):
            assert key in out, f"Missing existing key: {key}"

    def test_existing_values_match_standalone_functions(self):
        r = _make_returns()
        out = m.compute_core_metrics(r)
        assert out["sharpe"] == pytest.approx(m.compute_sharpe(r), rel=1e-10)
        assert out["cagr"] == pytest.approx(m.compute_cagr(r), rel=1e-10)
        assert out["volatility"] == pytest.approx(m.compute_volatility(r), rel=1e-10)
        assert out["max_drawdown"] == pytest.approx(m.compute_max_drawdown(r), rel=1e-10)

    def test_empty_returns_safe(self):
        out = m.compute_core_metrics(np.array([]))
        assert out["ev"] == 0.0
        assert out["ev_annual"] == 0.0
        assert out["n_obs"] == 0


# ---------------------------------------------------------------------------
# 3.  Sign semantics: positive mean → ev_annual > 0; negative → < 0
# ---------------------------------------------------------------------------

class TestEVSignSemantics:
    def test_positive_mean_gives_positive_ev_annual(self):
        r = np.full(500, 0.001)  # uniformly positive
        out = m.compute_core_metrics(r)
        assert out["ev_annual"] > 0

    def test_negative_mean_gives_negative_ev_annual(self):
        r = np.full(500, -0.001)  # uniformly negative
        out = m.compute_core_metrics(r)
        assert out["ev_annual"] < 0

    def test_zero_mean_gives_zero_ev_annual(self):
        # Perfectly symmetric ±equal array → mean == 0
        r = np.array([0.01, -0.01] * 50)
        out = m.compute_core_metrics(r)
        assert out["ev_annual"] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 4.  render_html: EV tile present when key available; backward-compat when missing
# ---------------------------------------------------------------------------

class TestRenderHTMLExpectedValue:
    def _make_payload(self, *, include_ev: bool = True) -> dict:
        """Build a minimal payload that render_html can consume."""
        import pandas as pd

        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-02", periods=252)
        r = pd.Series(rng.normal(0.0003, 0.01, 252), index=idx)

        from rigor.engine import BacktestResult
        equity = (1 + r).cumprod()
        metrics = m.compute_core_metrics(r.to_numpy())
        if not include_ev:
            metrics.pop("ev", None)
            metrics.pop("ev_annual", None)
        result = BacktestResult(returns=r, equity=equity, metrics=metrics)
        return build_payload(result, "TestStrategy")

    def test_ev_tile_present(self):
        payload = self._make_payload(include_ev=True)
        html = render_html(payload)
        assert "Expected Value" in html

    def test_ev_tile_shows_percent(self):
        payload = self._make_payload(include_ev=True)
        ev_annual = payload["metrics"]["ev_annual"]
        expected_pct = f"{ev_annual * 100:.2f}%"
        html = render_html(payload)
        assert expected_pct in html

    def test_backward_compat_missing_ev_annual(self):
        """render_html must not raise when ev_annual is absent (older summary.json)."""
        payload = self._make_payload(include_ev=False)
        # Must not raise
        html = render_html(payload)
        # Tile should still appear, showing fallback "—"
        assert "Expected Value" in html
        assert "—" in html

    def test_ev_tile_position_near_top(self):
        """Expected Value tile must appear before Max Drawdown in the HTML."""
        payload = self._make_payload(include_ev=True)
        html = render_html(payload)
        pos_ev = html.find("Expected Value")
        pos_mdd = html.find("Max Drawdown")
        assert pos_ev != -1 and pos_mdd != -1
        assert pos_ev < pos_mdd, "Expected Value tile should appear before Max Drawdown"
