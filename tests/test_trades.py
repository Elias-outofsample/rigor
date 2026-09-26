"""Hermetic tests for rigor.project.trades.

Uses only synthetic data and tmp_path — no network, no real strategies/ dirs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rigor.project.trades import (
    build_position_ledger,
    summarize_ledger,
    write_position_ledger,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_DATES = pd.date_range("2024-01-01", periods=30, freq="B")


def _make_weights() -> pd.DataFrame:
    """Build a synthetic weights frame with well-defined spells.

    Symbol A: two spells (bars 0-2, then bars 4-5).
    Symbol B: one spell (bars 5-9).
    Symbol C: all zeros (never traded).
    """
    n = len(_DATES)
    a = np.zeros(n)
    a[0:3] = 0.5   # spell 1: 3 bars at 0.5
    a[4:6] = 0.3   # spell 2: 2 bars at 0.3

    b = np.zeros(n)
    b[5:10] = -0.2  # 5 bars short at -0.2

    c = np.zeros(n)  # never active

    return pd.DataFrame({"A": a, "B": b, "C": c}, index=_DATES)


def _make_result(weights: pd.DataFrame | None):
    """Minimal duck-typed BacktestResult stand-in."""
    return SimpleNamespace(weights=weights)


def _make_asset_returns(weights: pd.DataFrame) -> pd.DataFrame:
    """Constant +1% daily return for every symbol in *weights*."""
    return pd.DataFrame(0.01, index=weights.index, columns=weights.columns)


# ── test 1: spell counting and attributes ────────────────────────────────────

def test_spell_count_and_attributes():
    """build_position_ledger extracts the correct spells from synthetic weights."""
    weights = _make_weights()
    result = _make_result(weights)
    ledger = build_position_ledger(result)

    # Symbol A has 2 spells; Symbol B has 1 spell; Symbol C has 0 — total 3.
    assert len(ledger) == 3, f"Expected 3 spells, got {len(ledger)}"

    a_spells = ledger[ledger["symbol"] == "A"].reset_index(drop=True)
    b_spells = ledger[ledger["symbol"] == "B"].reset_index(drop=True)
    c_spells = ledger[ledger["symbol"] == "C"]

    assert len(a_spells) == 2, "Symbol A must have exactly 2 spells"
    assert len(b_spells) == 1, "Symbol B must have exactly 1 spell"
    assert len(c_spells) == 0, "Symbol C (all-zero) must have no spells"

    # First A spell: bars 0-2 → 3 holding days, long, entry = _DATES[0]
    s0 = a_spells.iloc[0]
    assert s0["holding_days"] == 3
    assert s0["side"] == "long"
    assert s0["entry_date"] == _DATES[0]
    assert s0["exit_date"] == _DATES[2]
    assert abs(s0["avg_weight"] - 0.5) < 1e-9

    # Second A spell: bars 4-5 → 2 holding days
    s1 = a_spells.iloc[1]
    assert s1["holding_days"] == 2
    assert s1["side"] == "long"
    assert s1["entry_date"] == _DATES[4]
    assert s1["exit_date"] == _DATES[5]

    # B spell: bars 5-9 → 5 holding days, short
    bs = b_spells.iloc[0]
    assert bs["holding_days"] == 5
    assert bs["side"] == "short"
    assert abs(bs["avg_weight"] - (-0.2)) < 1e-9


# ── test 2: returns-only result (weights=None) → empty ledger, no file ───────

def test_returns_only_result_gives_empty_ledger_and_no_file(tmp_path: Path):
    """weights=None → empty ledger; write_position_ledger returns None."""
    result = _make_result(None)
    ledger = build_position_ledger(result)

    # Empty but correct columns.
    assert ledger.empty
    assert list(ledger.columns) == [
        "symbol", "side", "entry_date", "exit_date", "holding_days",
        "avg_weight", "max_weight", "n_rebalances", "return_contribution",
        "max_adverse_excursion", "max_favorable_excursion",
    ]

    # write_position_ledger returns None and does NOT create a file.
    out = write_position_ledger(tmp_path / "strat", "demo", result)
    assert out is None
    positions_csv = tmp_path / "strat" / "artifacts" / "demo_positions.csv"
    assert not positions_csv.exists()


# ── test 3: return_contribution populated when asset_returns supplied ─────────

def test_return_contribution_populated_with_asset_returns():
    """Providing asset_returns populates finite return_contribution values."""
    weights = _make_weights()
    result = _make_result(weights)
    ar = _make_asset_returns(weights)

    ledger = build_position_ledger(result, asset_returns=ar)

    assert not ledger.empty
    # All spells that have at least 2 bars (so shift(1) catches at least 1 bar)
    # should produce a finite (non-NaN) contribution.
    multi_bar = ledger[ledger["holding_days"] > 1]
    assert multi_bar["return_contribution"].notna().all(), (
        "Multi-bar spells must have finite return_contribution"
    )
    # Single-bar spells may be NaN (shifted weight is from outside the spell).
    # Just confirm the column is present.
    assert "return_contribution" in ledger.columns


# ── test 4: summarize_ledger keys and NaN-safety on empty ────────────────────

def test_summarize_ledger_keys_present():
    """summarize_ledger returns the expected keys for a non-empty ledger."""
    weights = _make_weights()
    ar = _make_asset_returns(weights)
    ledger = build_position_ledger(_make_result(weights), asset_returns=ar)

    summary = summarize_ledger(ledger)

    required_keys = {
        "n_trades", "n_symbols", "pct_long", "avg_holding_days",
        "median_holding_days", "max_holding_days", "turnover_proxy",
    }
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - summary.keys()}"
    )
    assert summary["n_trades"] == 3
    assert summary["n_symbols"] == 2  # A and B (C never traded)
    assert 0.0 <= summary["pct_long"] <= 1.0
    # Return-contribution keys should be present since we provided asset_returns.
    for key in ("win_rate", "avg_win", "avg_loss", "profit_factor"):
        assert key in summary, f"Expected return-contribution key '{key}'"


def test_summarize_ledger_empty_is_nan_safe():
    """summarize_ledger on an empty ledger must not raise and returns correct keys."""
    summary = summarize_ledger(pd.DataFrame())

    assert summary["n_trades"] == 0
    assert summary["n_symbols"] == 0
    assert np.isnan(summary["avg_holding_days"])
    assert np.isnan(summary["pct_long"])
    # Return-contribution stats should NOT be present on an empty ledger.
    assert "win_rate" not in summary


# ── test 5: write_position_ledger writes a round-trippable CSV ───────────────

def test_write_position_ledger_roundtrip(tmp_path: Path):
    """write_position_ledger writes a CSV that re-reads to the same shape."""
    weights = _make_weights()
    result = _make_result(weights)
    strategy_dir = tmp_path / "strategies" / "momentum" / "test_strat"

    out_path = write_position_ledger(strategy_dir, "test_strat", result)

    assert out_path is not None
    assert out_path.exists()
    assert out_path.name == "test_strat_positions.csv"

    # Confirm it sits in artifacts/.
    assert out_path.parent == strategy_dir / "artifacts"

    # Round-trip check: re-read and compare shape.
    ledger = build_position_ledger(result)
    reread = pd.read_csv(out_path)
    assert reread.shape == ledger.shape
    assert list(reread.columns) == list(ledger.columns)


# ── test 6: raw weights DataFrame accepted directly ──────────────────────────

def test_raw_weights_dataframe_accepted():
    """build_position_ledger also accepts a raw pd.DataFrame (not a result object)."""
    weights = _make_weights()
    ledger = build_position_ledger(weights)

    assert len(ledger) == 3
    assert set(ledger["symbol"].unique()) == {"A", "B"}


# ── test 7: n_rebalances counts correctly ────────────────────────────────────

def test_n_rebalances_counted():
    """n_rebalances counts weight changes within a spell."""
    # Spell with three different weight levels: 0.5, 0.3, 0.3 → 1 change.
    n = 5
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    w = pd.DataFrame({"X": [0.5, 0.3, 0.3, 0.7, 0.0]}, index=dates)
    ledger = build_position_ledger(w)

    # One spell: bars 0-3; bars 0→1 change (+1), 1→2 no change, 2→3 change (+1) = 2 changes.
    assert len(ledger) == 1
    assert ledger.iloc[0]["n_rebalances"] == 2
    assert ledger.iloc[0]["holding_days"] == 4


# ── test 8: MAE/MFE — no asset_returns → NaN ────────────────────────────────

def test_mae_mfe_nan_without_asset_returns():
    """Without asset_returns, MAE and MFE columns exist but are NaN."""
    weights = _make_weights()
    ledger = build_position_ledger(_make_result(weights))

    assert "max_adverse_excursion" in ledger.columns
    assert "max_favorable_excursion" in ledger.columns
    assert ledger["max_adverse_excursion"].isna().all()
    assert ledger["max_favorable_excursion"].isna().all()


# ── test 9: MAE — losing trade drops to -8% mid-way, exits at -2% ───────────

def test_mae_losing_trade():
    """A position that dips to -8% cumulative P&L then exits at -2%: MAE = -0.08.

    Setup (weight=1.0 constant, all bars except the first generate the shifted P&L):
      Bar 0  weight=1.0  return=NaN (shifted from bar -1, which doesn't exist) → pnl=0
      Bar 1  weight=1.0  return=-0.05   → cum_pnl = -0.05
      Bar 2  weight=1.0  return=-0.03   → cum_pnl = -0.08   ← MAE here
      Bar 3  weight=1.0  return=+0.06   → cum_pnl = -0.02   ← exit
    Expected MAE = -0.08, return_contribution = -0.02, MFE = 0.0
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    # Spell covers all 4 bars (weight=1.0 throughout).
    weights = pd.DataFrame({"S": [1.0, 1.0, 1.0, 1.0]}, index=dates)
    # Bar 0 return is irrelevant (shifted weight from before the spell = NaN → 0).
    asset_returns = pd.DataFrame({"S": [0.0, -0.05, -0.03, 0.06]}, index=dates)

    ledger = build_position_ledger(weights, asset_returns=asset_returns)

    assert len(ledger) == 1
    row = ledger.iloc[0]

    assert abs(row["max_adverse_excursion"] - (-0.08)) < 1e-9, (
        f"Expected MAE=-0.08, got {row['max_adverse_excursion']}"
    )
    assert abs(row["max_favorable_excursion"] - 0.0) < 1e-9, (
        f"Expected MFE=0.0, got {row['max_favorable_excursion']}"
    )
    # return_contribution should reflect exit at -0.02 cumulative.
    assert abs(row["return_contribution"] - (-0.02)) < 1e-9, (
        f"Expected rc=-0.02, got {row['return_contribution']}"
    )


# ── test 10: MFE — winning trade peaks at +12%, exits at +6% ────────────────

def test_mfe_winning_trade():
    """A position that peaks at +12% cumulative P&L then exits at +6%: MFE = +0.12.

    Setup (weight=1.0 constant):
      Bar 0  weight=1.0  return=NaN (shifted) → pnl=0
      Bar 1  weight=1.0  return=+0.07   → cum_pnl = +0.07
      Bar 2  weight=1.0  return=+0.05   → cum_pnl = +0.12   ← MFE here
      Bar 3  weight=1.0  return=-0.06   → cum_pnl = +0.06   ← exit
    Expected MFE = +0.12, return_contribution = +0.06, MAE = 0.0
    """
    dates = pd.date_range("2024-02-01", periods=4, freq="B")
    weights = pd.DataFrame({"S": [1.0, 1.0, 1.0, 1.0]}, index=dates)
    asset_returns = pd.DataFrame({"S": [0.0, 0.07, 0.05, -0.06]}, index=dates)

    ledger = build_position_ledger(weights, asset_returns=asset_returns)

    assert len(ledger) == 1
    row = ledger.iloc[0]

    assert abs(row["max_favorable_excursion"] - 0.12) < 1e-9, (
        f"Expected MFE=0.12, got {row['max_favorable_excursion']}"
    )
    assert abs(row["max_adverse_excursion"] - 0.0) < 1e-9, (
        f"Expected MAE=0.0, got {row['max_adverse_excursion']}"
    )
    assert abs(row["return_contribution"] - 0.06) < 1e-9, (
        f"Expected rc=+0.06, got {row['return_contribution']}"
    )


# ── test 11: MAE/MFE with weight < 1 ────────────────────────────────────────

def test_mae_mfe_partial_weight():
    """MAE/MFE scale with position weight, not raw price moves.

    weight=0.5, returns: [0, -0.10, +0.04] → bar_pnl = [0, -0.05, +0.02]
    cum_pnl: [0, -0.05, -0.03]
    MAE = -0.05, MFE = 0.0
    """
    dates = pd.date_range("2024-03-01", periods=3, freq="B")
    weights = pd.DataFrame({"S": [0.5, 0.5, 0.5]}, index=dates)
    asset_returns = pd.DataFrame({"S": [0.0, -0.10, 0.04]}, index=dates)

    ledger = build_position_ledger(weights, asset_returns=asset_returns)

    row = ledger.iloc[0]
    assert abs(row["max_adverse_excursion"] - (-0.05)) < 1e-9, (
        f"Expected MAE=-0.05, got {row['max_adverse_excursion']}"
    )
    assert abs(row["max_favorable_excursion"] - 0.0) < 1e-9, (
        f"Expected MFE=0.0, got {row['max_favorable_excursion']}"
    )


# ── test 12: MAE/MFE both non-zero in a round-trip spell ────────────────────

def test_mae_and_mfe_both_nonzero():
    """A spell that goes negative then recovers to positive has both MAE < 0 and MFE > 0.

    weight=1.0, returns: [0, -0.04, +0.10, -0.03]
    bar_pnl: [0, -0.04, +0.10, -0.03]
    cum_pnl: [0, -0.04, +0.06, +0.03]
    MAE = -0.04, MFE = +0.06
    """
    dates = pd.date_range("2024-04-01", periods=4, freq="B")
    weights = pd.DataFrame({"S": [1.0, 1.0, 1.0, 1.0]}, index=dates)
    asset_returns = pd.DataFrame({"S": [0.0, -0.04, 0.10, -0.03]}, index=dates)

    ledger = build_position_ledger(weights, asset_returns=asset_returns)

    row = ledger.iloc[0]
    assert abs(row["max_adverse_excursion"] - (-0.04)) < 1e-9, (
        f"Expected MAE=-0.04, got {row['max_adverse_excursion']}"
    )
    assert abs(row["max_favorable_excursion"] - 0.06) < 1e-9, (
        f"Expected MFE=0.06, got {row['max_favorable_excursion']}"
    )


# ── test 13: summarize_ledger MAE/MFE aggregates ────────────────────────────

def test_summarize_ledger_mae_mfe_aggregates():
    """summarize_ledger produces correct MAE/MFE aggregates for a mixed ledger.

    We build a two-spell ledger manually (skipping build_position_ledger) so the
    expected values are exact:
      Spell 1 (winner): rc=+0.06, MAE=-0.04, MFE=+0.06
      Spell 2 (loser):  rc=-0.02, MAE=-0.08, MFE=0.0
    """
    ledger = pd.DataFrame(
        [
            {
                "symbol": "A",
                "side": "long",
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "holding_days": 4,
                "avg_weight": 1.0,
                "max_weight": 1.0,
                "n_rebalances": 0,
                "return_contribution": 0.06,
                "max_adverse_excursion": -0.04,
                "max_favorable_excursion": 0.06,
            },
            {
                "symbol": "A",
                "side": "long",
                "entry_date": pd.Timestamp("2024-01-08"),
                "exit_date": pd.Timestamp("2024-01-11"),
                "holding_days": 4,
                "avg_weight": 1.0,
                "max_weight": 1.0,
                "n_rebalances": 0,
                "return_contribution": -0.02,
                "max_adverse_excursion": -0.08,
                "max_favorable_excursion": 0.0,
            },
        ]
    )

    summary = summarize_ledger(ledger)

    # Winning spell (rc > 0): MAE=-0.04, MFE=+0.06
    assert abs(summary["mae_mean_winning"] - (-0.04)) < 1e-9
    assert abs(summary["mae_median_winning"] - (-0.04)) < 1e-9
    assert abs(summary["mfe_mean_winning"] - 0.06) < 1e-9
    assert abs(summary["mfe_median_winning"] - 0.06) < 1e-9

    # Losing spell (rc <= 0): MAE=-0.08, MFE=0.0
    assert abs(summary["mae_mean_losing"] - (-0.08)) < 1e-9
    assert abs(summary["mae_median_losing"] - (-0.08)) < 1e-9
    assert abs(summary["mae_worst"] - (-0.08)) < 1e-9
    assert abs(summary["mfe_mean_losing"] - 0.0) < 1e-9
    assert abs(summary["mfe_median_losing"] - 0.0) < 1e-9
