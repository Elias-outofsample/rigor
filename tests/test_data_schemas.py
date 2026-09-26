"""Tests for the OHLCV data-contract layer (``rigor.data.schemas``).

Two halves:

* **Good data passes** — a clean frame, NaN legs (vendor gaps), a 0 low and a
  NaN close (real thin/halted-session artifacts the engine neutralises), zero/NaN
  volume, and the *real* committed SPY fixture all validate.
* **Bad data fails loudly** — only the *structurally impossible* (negative price,
  high<low, duplicate/out-of-order date) raises ``DataContractError``, with a
  message naming the ticker, date, and rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rigor.data.loader import _to_frame
from rigor.data.schemas import DataContractError, validate_ohlcv, validate_returns

_REPRO = Path(__file__).resolve().parent / "fixtures" / "repro" / "2026-06-01"
FIXTURE_SPY = _REPRO / "eod" / "SPY.json"


def _clean_frame() -> pd.DataFrame:
    """A small, well-formed OHLCV frame that must always pass."""
    return pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "open": [100.0, 101.0, 102.0],
            "high": [103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1_000, 2_000, 3_000],
        }
    )


# ---------------------------------------------------------------------------
# Good data passes
# ---------------------------------------------------------------------------

def test_clean_frame_passes_and_is_returned_unchanged():
    df = _clean_frame()
    out = validate_ohlcv(df, ticker="TEST")
    assert out is df  # returned inline, untouched


def test_empty_frame_is_valid():
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    pd.testing.assert_frame_equal(validate_ohlcv(empty, ticker="TEST"), empty)


def test_nan_in_non_close_legs_is_tolerated():
    # A vendor gap that NaNs open/high/low (but keeps close) must NOT fail:
    # OHLC consistency is only enforced on fully-populated bars.
    df = _clean_frame()
    df.loc[1, ["open", "high", "low"]] = np.nan
    validate_ohlcv(df, ticker="GAP")  # no raise


def test_zero_and_nan_volume_is_tolerated():
    df = _clean_frame()
    df.loc[0, "volume"] = 0
    df.loc[1, "volume"] = np.nan
    validate_ohlcv(df, ticker="THIN")  # volume is unconstrained


def test_frame_without_volume_column_passes():
    df = _clean_frame().drop(columns=["volume"])
    validate_ohlcv(df, ticker="NOVOL")  # volume is optional


def test_timestamp_dates_pass():
    df = _clean_frame()
    df["date"] = pd.to_datetime(df["date"])
    validate_ohlcv(df, ticker="TS")


def test_real_spy_fixture_passes():
    """The committed SPY snapshot (8000+ real bars) must validate as-is."""
    raw = json.loads(FIXTURE_SPY.read_text(encoding="utf-8"))
    df = _to_frame(raw)
    assert not df.empty
    out = validate_ohlcv(df, ticker="SPY")
    assert len(out) == len(df)


# ---------------------------------------------------------------------------
# Bad data fails loudly — one case per contract rule
# ---------------------------------------------------------------------------

def test_negative_price_raises():
    df = _clean_frame()
    df.loc[1, "open"] = -5.0
    with pytest.raises(DataContractError, match="price violation: BAD 2020-01-03"):
        validate_ohlcv(df, ticker="BAD")


def test_zero_low_is_tolerated():
    # EODHD ships `low=0` on a thin / halted / no-trade session; the engine
    # neutralises it, so the contract tolerates a 0 price (only a *negative* one
    # is a violation). This is the real case that broke 89 strategies on re-run.
    df = _clean_frame()
    df.loc[0, "low"] = 0.0
    validate_ohlcv(df, ticker="THIN")  # no raise


def test_high_below_low_raises_with_ticker_and_date():
    df = _clean_frame()
    df.loc[1, "high"] = 50.0  # high now below low (100)
    with pytest.raises(DataContractError, match=r"OHLC violation: SPY 2020-01-03"):
        validate_ohlcv(df, ticker="SPY")


def test_low_above_close_raises():
    df = _clean_frame()
    df.loc[0, "low"] = 200.0  # low above both open and close
    with pytest.raises(DataContractError, match="OHLC violation: BAD 2020-01-02 low"):
        validate_ohlcv(df, ticker="BAD")


def test_high_below_close_raises():
    df = _clean_frame()
    df.loc[2, "high"] = 50.0  # high below close (104) but not below low (101)
    with pytest.raises(DataContractError, match="OHLC violation: BAD 2020-01-06 high"):
        validate_ohlcv(df, ticker="BAD")


def test_nan_close_is_tolerated():
    # A NaN bar (no-trade session) is a real vendor gap the loader neutralises;
    # the contract tolerates it rather than failing the whole symbol's load.
    df = _clean_frame()
    df.loc[1, "close"] = np.nan
    validate_ohlcv(df, ticker="GAP")  # no raise


def test_duplicate_date_raises():
    df = _clean_frame()
    df.loc[2, "date"] = "2020-01-03"  # duplicate of row 1
    with pytest.raises(DataContractError, match="date violation: BAD duplicate date"):
        validate_ohlcv(df, ticker="BAD")


def test_non_monotonic_dates_raise():
    df = _clean_frame()
    df["date"] = ["2020-01-02", "2020-01-06", "2020-01-03"]  # out of order
    with pytest.raises(DataContractError, match="date violation: BAD .* not after"):
        validate_ohlcv(df, ticker="BAD")


def test_missing_column_raises():
    df = _clean_frame().drop(columns=["high"])
    with pytest.raises(DataContractError, match="schema violation: BAD .* missing"):
        validate_ohlcv(df, ticker="BAD")


# ---------------------------------------------------------------------------
# Returns contract (optional, lighter)
# ---------------------------------------------------------------------------

def test_validate_returns_passes_clean_series():
    s = pd.Series([0.01, -0.02, 0.005])
    assert validate_returns(s, name="R") is s


def test_validate_returns_rejects_nan():
    s = pd.Series([0.01, np.nan, 0.005])
    with pytest.raises(DataContractError, match="returns violation"):
        validate_returns(s, name="R")


def test_validate_returns_rejects_inf():
    s = pd.Series([0.01, np.inf, 0.005])
    with pytest.raises(DataContractError, match="returns violation"):
        validate_returns(s, name="R")


def test_validate_returns_empty_is_valid():
    s = pd.Series(dtype="float64")
    assert validate_returns(s, name="R") is s


# ---------------------------------------------------------------------------
# Integration: the loader seam rejects corrupt vendor data
# ---------------------------------------------------------------------------

def test_loader_prices_rejects_corrupt_vendor_data(tmp_path):
    """A bad bar from the source surfaces as a clear error through DataLoader.prices."""
    from rigor.data.config import DataConfig
    from rigor.data.loader import DataLoader

    class _BadSource:
        def eod(self, symbol):
            return [
                {"date": "2020-03-16", "open": 100.0, "high": 90.0, "low": 95.0,
                 "close": 98.0, "adjusted_close": 98.0, "volume": 1000},  # high<low
            ]

        def splits(self, symbol):
            return []

        def dividends(self, symbol):
            return []

    cfg = DataConfig(api_key="OFFLINE", cache_dir=tmp_path, as_of="2026-06-01")
    dl = DataLoader(cfg, source=_BadSource())
    with pytest.raises(DataContractError, match=r"OHLC violation: SPY 2020-03-16"):
        dl.prices("SPY")
