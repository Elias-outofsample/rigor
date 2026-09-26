"""Hermetic tests for the TradeLog -> position-ledger path (event-driven strategies).

Verifies that a strategy which records trades into a ``TradeLog`` (instead of
returning a weights matrix) produces a ledger with correct per-trade MAE/MFE.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rigor.engine import TradeLog, result_from_returns
from rigor.project.trades import build_position_ledger


def _result_with_trades(log: TradeLog):
    r = pd.Series([0.0, 0.01, -0.005], index=pd.date_range("2020-01-02", periods=3))
    return result_from_returns(r, trade_log=log)


def test_long_trade_mae_mfe() -> None:
    """A long trade dipping to -8% then peaking +12% before a +6% exit."""
    log = TradeLog()
    log.record(
        symbol="AAA", side="long",
        entry_date=pd.Timestamp("2020-01-02"), exit_date=pd.Timestamp("2020-01-08"),
        entry_price=100.0, exit_price=106.0,
        bar_prices=pd.Series([100.0, 92.0, 112.0, 106.0]),
    )
    led = build_position_ledger(_result_with_trades(log))
    assert len(led) == 1
    row = led.iloc[0]
    assert row["side"] == "long"
    assert row["max_adverse_excursion"] == pytest.approx(-0.08)
    assert row["max_favorable_excursion"] == pytest.approx(0.12)
    assert row["return_contribution"] == pytest.approx(0.06)


def test_short_trade_sign_convention() -> None:
    """A short trade: price rising is adverse (MAE), falling is favorable (MFE)."""
    log = TradeLog()
    log.record(
        symbol="BBB", side="short",
        entry_date=pd.Timestamp("2020-02-03"), exit_date=pd.Timestamp("2020-02-05"),
        entry_price=50.0, exit_price=48.0,
        bar_prices=pd.Series([50.0, 55.0, 48.0]),
    )
    led = build_position_ledger(_result_with_trades(log))
    row = led.iloc[0]
    assert row["side"] == "short"
    assert row["max_adverse_excursion"] == pytest.approx(-0.10)   # price rose 10% against the short
    assert row["max_favorable_excursion"] == pytest.approx(0.04)  # exit 4% in favour
    assert row["return_contribution"] == pytest.approx(0.04)


def test_no_trades_is_empty_ledger() -> None:
    """A returns-only result with no recorded trades yields an empty ledger."""
    led = build_position_ledger(_result_with_trades(TradeLog()))
    assert led.empty


def test_single_bar_intraday_trade() -> None:
    """Entry and exit on the same bar (no held marks) still computes from prices."""
    log = TradeLog()
    log.record(
        symbol="CCC", side="long",
        entry_date=pd.Timestamp("2020-03-02"), exit_date=pd.Timestamp("2020-03-02"),
        entry_price=10.0, exit_price=10.5, bar_prices=None,
    )
    led = build_position_ledger(_result_with_trades(log))
    row = led.iloc[0]
    assert row["return_contribution"] == pytest.approx(0.05)
    assert row["max_favorable_excursion"] == pytest.approx(0.05)
    assert row["max_adverse_excursion"] == pytest.approx(0.0)
    assert int(row["holding_days"]) == 1
