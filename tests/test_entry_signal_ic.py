"""Tests for rigor.analysis.entry_signal_ic — forensic entry-signal reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.analysis import entry_signal_ic as esi


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ---------------------------------------------------------------------------
# quintile_monotonicity
# ---------------------------------------------------------------------------


def test_monotone_signal_flags_monotonic_with_positive_spread(rng):
    n = 600
    fwd = rng.normal(0.0, 0.02, n)
    signal = fwd + rng.normal(0.0, 0.003, n)  # strongly aligned with fwd
    res = esi.quintile_monotonicity(signal, fwd)
    assert res["monotonic"] is True
    assert res["spread"] > 0
    assert res["buckets"][-1] > res["buckets"][0]


def test_negative_signal_monotonic_negative_spread(rng):
    n = 600
    fwd = rng.normal(0.0, 0.02, n)
    signal = -fwd + rng.normal(0.0, 0.003, n)
    res = esi.quintile_monotonicity(signal, fwd)
    assert res["monotonic"] is True
    assert res["spread"] < 0


def test_monotonicity_short_input_returns_none():
    res = esi.quintile_monotonicity(np.arange(3.0), np.arange(3.0))
    assert res["buckets"] is None
    assert res["monotonic"] is None
    assert res["spread"] is None


def test_monotonicity_drops_nan_pairs(rng):
    n = 400
    fwd = rng.normal(0.0, 0.02, n)
    signal = fwd.copy()
    signal[:50] = np.nan
    res = esi.quintile_monotonicity(signal, fwd)
    assert res["n"] == n - 50


# ---------------------------------------------------------------------------
# compute_entry_signal_ic
# ---------------------------------------------------------------------------


def test_dominant_signal_is_strongest_ic(rng):
    n = 500
    fwd = rng.normal(0.0, 0.02, n)
    signals = {
        "good": fwd + rng.normal(0.0, 0.004, n),
        "noise": rng.normal(0.0, 1.0, n),
        "weak": 0.2 * fwd + rng.normal(0.0, 0.02, n),
    }
    res = esi.compute_entry_signal_ic(signals, fwd)
    assert res["dominant"] == "good"
    assert res["n_trades"] == n
    # rows sorted by |IC| descending
    abs_ics = [abs(r["ic"]) for r in res["signals"]]
    assert abs_ics == sorted(abs_ics, reverse=True)
    good_row = next(r for r in res["signals"] if r["signal"] == "good")
    assert good_row["ic"] > 0.5
    assert good_row["monotonic"] is True
    assert good_row["t_stat"] > 1.96
    # Every row now carries Chatterjee's ξ as a float.
    assert all(isinstance(r["xi"], float) for r in res["signals"])


def test_anti_signal_picked_when_strongest(rng):
    n = 500
    fwd = rng.normal(0.0, 0.02, n)
    res = esi.compute_entry_signal_ic(
        {"anti": -fwd + rng.normal(0.0, 0.002, n),
         "noise": rng.normal(0.0, 1.0, n)},
        fwd,
    )
    assert res["dominant"] == "anti"
    assert res["dominant_ic"] < 0


def test_mismatched_shape_signal_skipped(rng):
    fwd = rng.normal(0.0, 0.02, 100)
    res = esi.compute_entry_signal_ic({"bad": np.zeros(50), "ok": fwd}, fwd)
    names = [r["signal"] for r in res["signals"]]
    assert "bad" not in names
    assert "ok" in names


def test_empty_inputs_no_raise():
    res = esi.compute_entry_signal_ic({}, np.array([]))
    assert res["signals"] == []
    assert res["dominant"] is None
    assert res["dominant_ic"] is None
    assert res["n_trades"] == 0


# ---------------------------------------------------------------------------
# reconstruct_candidate_signals
# ---------------------------------------------------------------------------


def _price_series(rng, n=900):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    return pd.Series(close, index=idx)


def test_reconstruct_label_indexed(rng):
    close = _price_series(rng)
    entries = close.index[400::5]
    sigs = esi.reconstruct_candidate_signals(close, entries)
    assert set(sigs) == {"rsi2", "momentum", "reversal", "ibs",
                         "ma_distance", "low_vol"}
    assert all(len(v) == len(entries) for v in sigs.values())
    # late entries (past all warm-ups) should be finite for every signal
    assert all(np.isfinite(v[-5:]).all() for v in sigs.values())


def test_reconstruct_positional_indexed(rng):
    close = _price_series(rng)
    pos = np.arange(400, 900, 5)
    sigs = esi.reconstruct_candidate_signals(close, pos)
    assert all(len(v) == len(pos) for v in sigs.values())


def test_reconstruct_feeds_ic_end_to_end(rng):
    """A signal engineered to drive forward returns is recovered as dominant."""
    close = _price_series(rng, 1000)
    entries = close.index[300::3]
    sigs = esi.reconstruct_candidate_signals(close, entries)
    # Build forward returns that depend on the reconstructed momentum signal.
    mom = np.nan_to_num(sigs["momentum"], nan=0.0)
    fwd = 0.5 * (mom - mom.mean()) + rng.normal(0.0, 0.002, len(entries))
    res = esi.compute_entry_signal_ic(sigs, fwd)
    assert res["dominant"] == "momentum"
