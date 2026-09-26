"""Tests for rigor.analysis.diagnostics (hermetic, offline).

All data is synthetic — no network calls, no disk I/O beyond the module import.
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
# Synthetic data fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 1260  # ~5 years of daily data
_DATES = pd.bdate_range("2019-01-01", periods=_N)


def _make_returns(mu: float = 5e-4, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(mu + sigma * rng.standard_normal(_N), index=_DATES)


def _make_benchmark(seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(3e-4 + 0.012 * rng.standard_normal(_N), index=_DATES)


def _make_factors() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = _N
    return pd.DataFrame(
        {
            "Mkt-RF": rng.standard_normal(n) * 0.01,
            "SMB": rng.standard_normal(n) * 0.005,
            "HML": rng.standard_normal(n) * 0.004,
            "RF": np.full(n, 5e-5),
        },
        index=_DATES,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETURNS = _make_returns()
_BENCHMARK = _make_benchmark()
_FACTORS = _make_factors()


def _diag_returns_only() -> dict:
    return strategy_diagnostics(_RETURNS)


def _diag_with_benchmark() -> dict:
    return strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK)


def _diag_full() -> dict:
    return strategy_diagnostics(_RETURNS, benchmark=_BENCHMARK, factors=_FACTORS)


# ---------------------------------------------------------------------------
# Test 1: returns-only — tail/risk/crisis/capacity present; risk_dna/factor absent
# ---------------------------------------------------------------------------

class TestReturnsOnly:
    def test_present_blocks(self) -> None:
        diag = _diag_returns_only()
        available = diag["available"]
        for name in ("tail", "risk", "crisis", "capacity"):
            assert name in available, f"expected {name!r} in available"

    def test_absent_blocks(self) -> None:
        diag = _diag_returns_only()
        available = diag["available"]
        assert "risk_dna" not in available
        assert "factor" not in available
        assert diag["risk_dna"] is None
        assert diag["factor"] is None

    def test_tail_has_expected_keys(self) -> None:
        tail = _diag_returns_only()["tail"]
        assert tail is not None
        assert "hill_alpha" in tail
        assert "fat_tailed" in tail

    def test_risk_battery_keys(self) -> None:
        rb = _diag_returns_only()["risk"]
        assert rb is not None
        expected = {"var_95", "cvar_95", "omega", "ulcer_index", "sterling", "burke"}
        assert expected <= set(rb)

    def test_crisis_has_verdict(self) -> None:
        crisis = _diag_returns_only()["crisis"]
        assert crisis is not None
        assert "verdict" in crisis

    def test_capacity_headline_keys(self) -> None:
        cap = _diag_returns_only()["capacity"]
        assert cap is not None
        for key in ("capacity_low", "capacity_central", "capacity_high", "base_sharpe"):
            assert key in cap, f"missing {key!r}"

    def test_available_is_list(self) -> None:
        diag = _diag_returns_only()
        assert isinstance(diag["available"], list)


# ---------------------------------------------------------------------------
# Test 2: with benchmark → risk_dna present; with factors → factor present
# ---------------------------------------------------------------------------

class TestWithBenchmark:
    def test_risk_dna_present(self) -> None:
        diag = _diag_with_benchmark()
        assert "risk_dna" in diag["available"]
        rdna = diag["risk_dna"]
        assert rdna is not None
        assert "overall_profile" in rdna
        assert "payoff" in rdna

    def test_factor_absent_without_factors(self) -> None:
        diag = _diag_with_benchmark()
        assert "factor" not in diag["available"]


class TestWithFactors:
    def test_factor_present(self) -> None:
        diag = _diag_full()
        assert "factor" in diag["available"]
        fac = diag["factor"]
        assert fac is not None
        assert "alpha" in fac
        assert "alpha_t_stat" in fac

    def test_all_six_blocks_present(self) -> None:
        diag = _diag_full()
        available = set(diag["available"])
        for name in ("tail", "risk", "crisis", "capacity", "risk_dna", "factor"):
            assert name in available, f"{name!r} missing"


# ---------------------------------------------------------------------------
# Test 3: diagnostics_to_html
# ---------------------------------------------------------------------------

class TestToHtml:
    def test_non_empty_for_populated_diag(self) -> None:
        diag = _diag_returns_only()
        html = diagnostics_to_html(diag)
        assert html != ""
        assert '<div class="section">' in html

    def test_contains_block_headings(self) -> None:
        diag = _diag_returns_only()
        html = diagnostics_to_html(diag)
        assert "<h3>" in html

    def test_full_diag_contains_all_sections(self) -> None:
        diag = _diag_full()
        html = diagnostics_to_html(diag)
        assert "Tail Risk" in html
        assert "Risk Battery" in html
        assert "Crisis" in html
        assert "Capacity" in html
        assert "Risk-DNA" in html
        assert "Factor Regression" in html

    def test_empty_diag_returns_empty_string(self) -> None:
        empty_diag: dict = {
            "tail": None,
            "risk": None,
            "crisis": None,
            "capacity": None,
            "risk_dna": None,
            "factor": None,
            "available": [],
        }
        assert diagnostics_to_html(empty_diag) == ""

    def test_html_is_string(self) -> None:
        assert isinstance(diagnostics_to_html(_diag_returns_only()), str)


# ---------------------------------------------------------------------------
# Test 4: diagnostics_summary — JSON-serialisable
# ---------------------------------------------------------------------------

class TestSummary:
    def test_json_serialisable_returns_only(self) -> None:
        diag = _diag_returns_only()
        summ = diagnostics_summary(diag)
        json.dumps(summ)  # must not raise

    def test_json_serialisable_full(self) -> None:
        diag = _diag_full()
        summ = diagnostics_summary(diag)
        json.dumps(summ)  # must not raise

    def test_headline_keys_returns_only(self) -> None:
        diag = _diag_returns_only()
        summ = diagnostics_summary(diag)
        for key in ("var_95", "cvar_95", "kupiec_verdict", "crash_sharpe_ratio",
                    "capacity_central"):
            assert key in summ, f"missing {key!r}"

    def test_factor_keys_present_when_factors_supplied(self) -> None:
        diag = _diag_full()
        summ = diagnostics_summary(diag)
        assert "factor_alpha" in summ
        assert "factor_alpha_t" in summ

    def test_risk_dna_key_present_when_benchmark_supplied(self) -> None:
        diag = _diag_with_benchmark()
        summ = diagnostics_summary(diag)
        assert "risk_dna_payoff_type" in summ
        assert "risk_dna_profile" in summ

    def test_returns_dict(self) -> None:
        summ = diagnostics_summary(_diag_returns_only())
        assert isinstance(summ, dict)


# ---------------------------------------------------------------------------
# Test 5: degenerate / short returns don't raise
# ---------------------------------------------------------------------------

class TestDegenerate:
    def test_single_observation(self) -> None:
        tiny = pd.Series([0.001], index=pd.bdate_range("2020-01-01", periods=1))
        diag = strategy_diagnostics(tiny)
        assert isinstance(diag, dict)

    def test_empty_series(self) -> None:
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        diag = strategy_diagnostics(empty)
        assert isinstance(diag, dict)

    def test_all_zeros(self) -> None:
        zeros = pd.Series(np.zeros(50), index=pd.bdate_range("2020-01-01", periods=50))
        diag = strategy_diagnostics(zeros)
        assert isinstance(diag, dict)

    def test_numpy_array_input(self) -> None:
        arr = np.random.default_rng(1).standard_normal(200) * 0.01
        diag = strategy_diagnostics(arr)
        assert isinstance(diag, dict)
        assert isinstance(diagnostics_summary(diag), dict)
        # HTML must not raise even if blocks are sparse
        html = diagnostics_to_html(diag)
        assert isinstance(html, str)

    def test_short_returns_no_raise_with_benchmark(self) -> None:
        tiny = pd.Series([0.001, -0.002, 0.003],
                         index=pd.bdate_range("2020-01-01", periods=3))
        bench = pd.Series([0.001, 0.001, 0.001],
                          index=pd.bdate_range("2020-01-01", periods=3))
        diag = strategy_diagnostics(tiny, benchmark=bench)
        assert isinstance(diag, dict)

    def test_nan_rich_series(self) -> None:
        rng = np.random.default_rng(5)
        vals = rng.standard_normal(500) * 0.01
        vals[::3] = np.nan
        s = pd.Series(vals, index=pd.bdate_range("2018-01-01", periods=500))
        diag = strategy_diagnostics(s)
        assert isinstance(diag, dict)
        json.dumps(diagnostics_summary(diag))
