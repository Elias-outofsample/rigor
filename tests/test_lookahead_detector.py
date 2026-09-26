"""Tests for the future-data perturbation look-ahead detector (#9).

The detector (``rigor.analysis.lookahead.detect_lookahead``) reruns a strategy on a
dataset whose every bar *after* a cutoff is corrupted and asserts that all
positions *at or before* the cutoff are unchanged. A causal strategy passes; one
that reads the future fails.

These tests are hermetic (synthetic data only) and deterministic:

  * a deliberately LEAKY toy (``shift(-1)`` signal — tomorrow's return) is caught;
  * a CORRECT causal toy (1-bar lag) is not flagged;
  * shipped example strategies, built on the synthetic ``FakeDataLoader``, pass —
    confirming they are causal;
  * the preserved legacy Sharpe-shift entry point still returns its dict verdict.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fake_data import FakeDataLoader  # tests/ is on sys.path under pytest

from rigor.analysis import lookahead
from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig

_DATES = pd.bdate_range("2010-01-01", periods=400)
_SYMS = ("AAA", "BBB", "CCC")


# ===========================================================================
# A tiny deterministic price loader + toy strategies
# ===========================================================================

class _ToyLoader:
    """Minimal data loader exposing ``prices(symbol)`` like the real DataLoader."""

    def __init__(self) -> None:
        self._panels: dict[str, pd.DataFrame] = {}
        for k, sym in enumerate(_SYMS):
            rng = np.random.default_rng(100 + k)
            close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(_DATES))))
            self._panels[sym] = pd.DataFrame(
                {
                    "date": _DATES.strftime("%Y-%m-%d"),
                    "open": close, "high": close * 1.01, "low": close * 0.99,
                    "close": close, "adj_close": close,
                },
            )

    def prices(self, symbol: str, **_: Any) -> pd.DataFrame:
        return self._panels[symbol].copy()


class _ToyStrategy(StrategyBase):
    """Sign-of-return long/short basket; ``leaky`` flips it to read the future."""

    def __init__(self, config: StrategyConfig, data: Any, *, leaky: bool) -> None:
        super().__init__(config)
        self.data = data
        self.leaky = leaky

    def build_cache(self) -> dict[str, Any]:
        closes = {}
        for sym in _SYMS:
            df = self.data.prices(sym)
            df = df.set_index(pd.DatetimeIndex(df["date"]))
            closes[sym] = df["adj_close"]
        panel = pd.DataFrame(closes).reindex(_DATES)
        return {"dates": _DATES, "close": panel}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]):
        panel: pd.DataFrame = cache["close"]
        ret = panel.pct_change(fill_method=None)
        # leaky: today's weight is the sign of *tomorrow's* return (look-ahead).
        # causal: today's weight is the sign of *yesterday's* return (1-bar lag).
        signal = ret.shift(-1) if self.leaky else ret.shift(1)
        weights = np.sign(signal).fillna(0.0) / float(len(_SYMS))
        return simulate_weights(weights, panel, commission_bps=0.0)


def _toy_config() -> StrategyConfig:
    return StrategyConfig(name="Toy", start_date="2010-01-01", end_date=None,
                          commission_bps=0.0, rebalance_freq="daily")


# ===========================================================================
# 1 + 2: leaky toy is caught, causal toy is clean
# ===========================================================================

def test_leaky_toy_is_detected() -> None:
    data = _ToyLoader()
    strat = _ToyStrategy(_toy_config(), data, leaky=True)
    result = lookahead.detect_lookahead(
        strat, data, rebuild=lambda c, d: _ToyStrategy(c, d, leaky=True))
    assert result.leak_detected is True
    assert any(c.leak for c in result.per_cutoff)
    assert "LOOK-AHEAD DETECTED" in result.summary


def test_causal_toy_is_not_flagged() -> None:
    data = _ToyLoader()
    strat = _ToyStrategy(_toy_config(), data, leaky=False)
    result = lookahead.detect_lookahead(
        strat, data, rebuild=lambda c, d: _ToyStrategy(c, d, leaky=False))
    assert result.leak_detected is False
    # Every evaluated cutoff compared real pre-T positions and found no delta.
    tested = [c for c in result.per_cutoff if c.n_compared > 0]
    assert tested, "expected at least one cutoff with overlapping pre-T positions"
    assert all(c.max_abs_diff == 0.0 for c in tested)
    assert "no look-ahead" in result.summary


def test_scramble_mode_also_catches_the_leak() -> None:
    data = _ToyLoader()
    strat = _ToyStrategy(_toy_config(), data, leaky=True)
    result = lookahead.detect_lookahead(
        strat, data, mode="scramble",
        rebuild=lambda c, d: _ToyStrategy(c, d, leaky=True))
    assert result.leak_detected is True


def test_explicit_cutoffs_and_fractions_agree() -> None:
    data = _ToyLoader()
    strat = _ToyStrategy(_toy_config(), data, leaky=False)
    mid = _DATES[len(_DATES) // 2]
    res_frac = lookahead.detect_lookahead(
        strat, data, cutoffs=[0.5],
        rebuild=lambda c, d: _ToyStrategy(c, d, leaky=False))
    res_ts = lookahead.detect_lookahead(
        strat, data, cutoffs=[mid],
        rebuild=lambda c, d: _ToyStrategy(c, d, leaky=False))
    assert not res_frac.leak_detected and not res_ts.leak_detected
    assert res_frac.per_cutoff[0].cutoff == res_ts.per_cutoff[0].cutoff


# ===========================================================================
# 3: shipped example strategies are causal
# ===========================================================================

_REPO = Path(__file__).resolve().parents[1]


def _example_strategy(sid: str) -> tuple[Any, FakeDataLoader]:
    """Build one of the committed example strategies on the synthetic loader."""
    from rigor.project.run import load_config, load_strategy_module

    sdir = _REPO / "strategies" / sid
    config, raw = load_config(sdir)
    data = FakeDataLoader()
    return load_strategy_module(sdir, raw.get("slug") or sdir.name).build(config, data), data


def _assert_substantive_pass(result: lookahead.LookaheadResult) -> None:
    """A real causal engine must pass *and* have actually compared pre-T cells.

    This guards against the original flaw — a detector that "passes" only because
    it never compared anything (a vacuous all-clear).
    """
    assert result.leak_detected is False, result.summary
    assert result.error is None
    tested = [c for c in result.per_cutoff if c.n_compared > 0]
    assert tested, f"vacuous pass — no pre-T cells compared: {result.summary}"
    assert all(c.max_abs_diff == 0.0 for c in tested)


@pytest.mark.parametrize("sid", [
    "trend_following/sma_trend",
    "breakout/bollinger_squeeze",
    "mean_reversion/rsi2_mr",
    "carry/credit_carry_rotation",
])
def test_example_strategies_are_causal(sid: str) -> None:
    strat, data = _example_strategy(sid)
    _assert_substantive_pass(lookahead.detect_lookahead(strat, data))


def test_detector_defaults_data_to_strategy_attr() -> None:
    # When ``data`` is omitted the detector reads ``strategy.data``.
    strat, _ = _example_strategy("trend_following/sma_trend")
    _assert_substantive_pass(lookahead.detect_lookahead(strat))


# ===========================================================================
# Structured-result + legacy-compat surface
# ===========================================================================

def test_result_as_dict_and_bool() -> None:
    data = _ToyLoader()
    strat = _ToyStrategy(_toy_config(), data, leaky=True)
    result = lookahead.detect_lookahead(
        strat, data, rebuild=lambda c, d: _ToyStrategy(c, d, leaky=True))
    d = result.as_dict()
    assert d["leak_detected"] is True
    assert isinstance(d["per_cutoff"], list) and d["per_cutoff"]
    assert bool(result) is True  # truthy == leak


def test_missing_data_is_reported_not_crashed() -> None:
    strat = SimpleNamespace(backtest=lambda: SimpleNamespace(
        weights=None, returns=pd.Series([0.0, 0.1])))
    result = lookahead.detect_lookahead(strat, None)
    assert result.leak_detected is False
    assert result.error is not None


def test_legacy_sharpe_shift_entrypoint_still_works() -> None:
    # The original #9 callable signature is preserved and dispatched to.
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(600)

    def bt(cache: dict, params: dict) -> np.ndarray:
        shift = cache.get("_shift", 0)
        sig = np.roll(signal, shift) if shift else signal
        return 0.01 * sig * signal

    out = lookahead.detect_lookahead(bt, {"signal": signal}, {}, n_random_trials=3)
    assert isinstance(out, dict)
    assert out["shift_supported"] and out["suspect"] and out["confidence"] == "high"
