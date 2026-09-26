"""Extra diagnostics tests: new blocks (regime/monte_carlo/persistence/concentration/
decay_split/corr_breakdown). Hermetic, offline, synthetic data only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rigor.analysis.diagnostics import (
    diagnostics_summary,
    diagnostics_to_html,
    strategy_diagnostics,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N = 1260  # ~5 years of daily data
_DATES = pd.bdate_range("2019-01-01", periods=_N)


def _make_returns(mu: float = 5e-4, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(mu + sigma * rng.standard_normal(_N), index=_DATES)


def _make_benchmark(seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(3e-4 + 0.012 * rng.standard_normal(_N), index=_DATES)


_RETURNS = _make_returns()
_BENCHMARK = _make_benchmark()


# ---------------------------------------------------------------------------
# Test 1: new blocks appear in "available" with and without benchmark
# ---------------------------------------------------------------------------

class TestNewBlocksPresent:
    def test_new_blocks_without_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        available = diag["available"]
        for name in ("regime", "monte_carlo", "persistence", "concentration", "decay_split"):
            assert name in available, f"expected {name!r} in available (no benchmark)"

    def test_corr_breakdown_absent_without_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        assert "corr_breakdown" not in diag["available"]
        assert diag["corr_breakdown"] is None

    def test_corr_breakdown_present_with_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)
        assert "corr_breakdown" in diag["available"], (
            "corr_breakdown must be present with benchmark"
        )
        cb = diag["corr_breakdown"]
        assert cb is not None
        assert "mean_rho" in cb
        assert "n_regime_shifts" in cb
        assert "breakdown_flag" in cb

    def test_existing_blocks_still_present(self) -> None:
        """Regression: original blocks must still be in available."""
        diag = strategy_diagnostics(_RETURNS)
        available = diag["available"]
        for name in ("tail", "risk", "crisis", "capacity"):
            assert name in available, f"original block {name!r} missing after extension"

    def test_regime_block_structure(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        reg = diag["regime"]
        assert reg is not None
        assert isinstance(reg, dict)
        # Must have at least one regime label with sharpe + pct_time
        found_regime = False
        for _k, v in reg.items():
            if isinstance(v, dict) and "sharpe" in v and "pct_time" in v:
                found_regime = True
        assert found_regime, "regime block must have per-regime dicts with sharpe + pct_time"

    def test_monte_carlo_block_structure(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        mc = diag["monte_carlo"]
        assert mc is not None
        for key in ("sharpe", "ci_low", "ci_high", "p_sharpe_gt_0"):
            assert key in mc, f"monte_carlo missing key {key!r}"

    def test_persistence_block_structure(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        pers = diag["persistence"]
        assert pers is not None
        assert "has_decay" in pers
        assert "slope_annual" in pers
        assert "decay_flag" in pers
        assert "acf_lag1" in pers

    def test_concentration_block_structure(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        conc = diag["concentration"]
        assert conc is not None
        for key in ("max_month_share", "max_month", "n_months", "level"):
            assert key in conc, f"concentration missing key {key!r}"
        assert conc["level"] in ("OK", "WARN", "FAIL")

    def test_decay_split_block_structure(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        ds = diag["decay_split"]
        assert ds is not None
        for key in ("sharpe_first_half", "sharpe_second_half", "decay", "level"):
            assert key in ds, f"decay_split missing key {key!r}"
        assert ds["level"] in ("OK", "WARN", "FAIL")


# ---------------------------------------------------------------------------
# Test 2: giant-month injection → concentration FAIL
# ---------------------------------------------------------------------------

class TestConcentrationFail:
    def _make_giant_month_returns(self) -> pd.Series:
        """Insert a +200% month against near-zero background to trigger FAIL (share > 0.30)."""
        rng = np.random.default_rng(0)
        # Very small background returns so the big month dominates absolutely
        r = pd.Series(0.0 + 0.0005 * rng.standard_normal(_N), index=_DATES)
        # Replace one full month (21 trading days) with ~+200% monthly compound
        # (1 + x)^21 = 3.0  =>  x = 3^(1/21) - 1 ≈ 5.3%/day
        mid = _N // 2
        target_daily = (3.0 ** (1 / 21)) - 1
        r.iloc[mid : mid + 21] = target_daily
        return r

    def test_concentration_level_fail(self) -> None:
        rets = self._make_giant_month_returns()
        diag = strategy_diagnostics(rets)
        conc = diag["concentration"]
        assert conc is not None, "concentration block must be present"
        assert conc["level"] == "FAIL", (
            f"expected FAIL for giant-month series, got {conc['level']!r}; "
            f"max_month_share={conc.get('max_month_share')}"
        )
        assert conc["max_month_share"] > 0.30, (
            f"max_month_share should exceed 0.30, got {conc['max_month_share']}"
        )


# ---------------------------------------------------------------------------
# Test 3: decaying series → decay_split WARN or FAIL
# ---------------------------------------------------------------------------

class TestDecaySplit:
    def _make_decaying_returns(self) -> pd.Series:
        """Strong positive first half, near-flat/negative second half."""
        rng = np.random.default_rng(10)
        n_half = _N // 2
        first_half = 8e-4 + 0.01 * rng.standard_normal(n_half)
        second_half = -1e-5 + 0.01 * rng.standard_normal(_N - n_half)
        combined = np.concatenate([first_half, second_half])
        return pd.Series(combined, index=_DATES)

    def test_decay_split_level_warn_or_fail(self) -> None:
        rets = self._make_decaying_returns()
        diag = strategy_diagnostics(rets)
        ds = diag["decay_split"]
        assert ds is not None, "decay_split block must be present"
        assert ds["level"] in ("WARN", "FAIL"), (
            f"expected WARN or FAIL for decaying series, got {ds['level']!r}; "
            f"decay={ds.get('decay')}, first={ds.get('sharpe_first_half')}, "
            f"second={ds.get('sharpe_second_half')}"
        )
        # First half should be materially better than second
        assert ds["sharpe_first_half"] > ds["sharpe_second_half"], (
            "First-half Sharpe should exceed second-half Sharpe for decaying series"
        )


# ---------------------------------------------------------------------------
# Test 4: HTML renders new sections
# ---------------------------------------------------------------------------

class TestHtmlNewSections:
    def test_html_non_empty(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        html = diagnostics_to_html(diag)
        assert html != ""
        assert '<div class="section">' in html

    def test_html_contains_new_headings(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        html = diagnostics_to_html(diag)
        assert "Vol-Regime Performance" in html, "regime section heading missing"
        assert "Monte Carlo" in html, "monte_carlo section heading missing"
        assert "Return Persistence" in html, "persistence section heading missing"
        assert "Concentration" in html, "concentration section heading missing"
        assert "Performance Decay" in html, "decay_split section heading missing"

    def test_html_contains_corr_breakdown_with_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)
        html = diagnostics_to_html(diag)
        assert "Rolling-Correlation Breakdown" in html, (
            "corr_breakdown section should render when benchmark is provided"
        )

    def test_html_corr_breakdown_absent_without_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        html = diagnostics_to_html(diag)
        assert "Rolling-Correlation Breakdown" not in html, (
            "corr_breakdown section must not render without benchmark"
        )

    def test_html_still_contains_original_headings(self) -> None:
        """Regression: original HTML sections must still render."""
        diag = strategy_diagnostics(_RETURNS)
        html = diagnostics_to_html(diag)
        assert "Tail Risk" in html
        assert "Risk Battery" in html


# ---------------------------------------------------------------------------
# Test 5: diagnostics_summary JSON-serialisable with new keys
# ---------------------------------------------------------------------------

class TestSummaryNewKeys:
    def test_json_serialisable(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        summ = diagnostics_summary(diag)
        json.dumps(summ)  # must not raise

    def test_json_serialisable_with_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)
        summ = diagnostics_summary(diag)
        json.dumps(summ)  # must not raise

    def test_new_summary_keys_present(self) -> None:
        diag = strategy_diagnostics(_RETURNS)
        summ = diagnostics_summary(diag)
        expected_keys = [
            "regime_best",
            "regime_worst",
            "mc_sharpe_ci_low",
            "mc_sharpe_ci_high",
            "edge_decay_slope_annual",
            "max_month_share",
            "concentration_level",
            "perf_decay",
            "perf_decay_level",
        ]
        for key in expected_keys:
            assert key in summ, f"summary missing key {key!r}"

    def test_corr_breakdown_summary_keys_with_benchmark(self) -> None:
        diag = strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)
        summ = diagnostics_summary(diag)
        assert "corr_breakdown_shifts" in summ
        assert "corr_breakdown_flag" in summ

    def test_no_non_serialisable_values(self) -> None:
        """All values must be str, int, float, bool, or None — no numpy types."""
        diag = strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)
        summ = diagnostics_summary(diag)
        allowed = (str, int, float, bool, type(None))
        for k, v in summ.items():
            assert isinstance(v, allowed), (
                f"summary[{k!r}] = {v!r} is type {type(v).__name__}, not JSON-safe"
            )


# ---------------------------------------------------------------------------
# Test 6: degenerate inputs don't raise; existing blocks still work
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    def test_single_observation_no_raise(self) -> None:
        tiny = pd.Series([0.001], index=pd.bdate_range("2020-01-01", periods=1))
        diag = strategy_diagnostics(tiny)
        assert isinstance(diag, dict)
        # No block should raise; new blocks may simply be absent (short data)

    def test_empty_series_no_raise(self) -> None:
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        diag = strategy_diagnostics(empty)
        assert isinstance(diag, dict)

    def test_all_zeros_no_raise(self) -> None:
        zeros = pd.Series(np.zeros(50), index=pd.bdate_range("2020-01-01", periods=50))
        diag = strategy_diagnostics(zeros)
        assert isinstance(diag, dict)
        json.dumps(diagnostics_summary(diag))

    def test_numpy_array_input_no_raise(self) -> None:
        arr = np.random.default_rng(1).standard_normal(200) * 0.01
        diag = strategy_diagnostics(arr)
        assert isinstance(diag, dict)
        assert isinstance(diagnostics_summary(diag), dict)
        html = diagnostics_to_html(diag)
        assert isinstance(html, str)

    def test_short_returns_with_benchmark_no_raise(self) -> None:
        tiny = pd.Series([0.001, -0.002, 0.003],
                         index=pd.bdate_range("2020-01-01", periods=3))
        bench = pd.Series([0.001, 0.001, 0.001],
                          index=pd.bdate_range("2020-01-01", periods=3))
        diag = strategy_diagnostics(tiny, benchmark=bench)
        assert isinstance(diag, dict)

    def test_nan_rich_series_no_raise(self) -> None:
        rng = np.random.default_rng(5)
        vals = rng.standard_normal(500) * 0.01
        vals[::3] = np.nan
        s = pd.Series(vals, index=pd.bdate_range("2018-01-01", periods=500))
        diag = strategy_diagnostics(s)
        assert isinstance(diag, dict)
        json.dumps(diagnostics_summary(diag))

    def test_original_blocks_still_work(self) -> None:
        """Regression: tail/risk/crisis/capacity must still produce results."""
        diag = strategy_diagnostics(_RETURNS)
        assert diag["tail"] is not None, "tail block regressed"
        assert diag["risk"] is not None, "risk block regressed"
        assert diag["crisis"] is not None, "crisis block regressed"
        assert diag["capacity"] is not None, "capacity block regressed"
        for name in ("tail", "risk", "crisis", "capacity"):
            assert name in diag["available"], f"{name!r} missing from available"

    def test_html_no_raise_on_degenerate(self) -> None:
        zeros = pd.Series(np.zeros(20), index=pd.bdate_range("2020-01-01", periods=20))
        diag = strategy_diagnostics(zeros)
        html = diagnostics_to_html(diag)
        assert isinstance(html, str)

    def test_summary_no_raise_on_degenerate(self) -> None:
        zeros = pd.Series(np.zeros(20), index=pd.bdate_range("2020-01-01", periods=20))
        diag = strategy_diagnostics(zeros)
        summ = diagnostics_summary(diag)
        json.dumps(summ)


# ---------------------------------------------------------------------------
# Test 7: concentration WARN threshold
# ---------------------------------------------------------------------------

class TestConcentrationWarn:
    def test_warn_threshold(self) -> None:
        """A ~25% single-month share triggers WARN."""
        rng = np.random.default_rng(77)
        r = pd.Series(1e-4 + 0.003 * rng.standard_normal(_N), index=_DATES)
        # Inject ~25% monthly return target: (1.25)^(1/21) - 1 per day
        target_daily = (1.25 ** (1 / 21)) - 1
        mid = _N // 2
        r.iloc[mid : mid + 21] = target_daily
        diag = strategy_diagnostics(r)
        conc = diag.get("concentration")
        assert conc is not None
        assert conc["level"] in ("WARN", "FAIL"), (
            f"expected WARN or FAIL for 25% month, got {conc['level']!r}; "
            f"share={conc.get('max_month_share')}"
        )
