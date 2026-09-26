"""Tests for the file-based futures ``DataSource`` (``rigor.data.futures_source``).

Hermetic and deterministic: a tiny committed fixture of three fake continuous
contracts (GC/CL parquet, NG csv with vendor-style headers) under
``tests/fixtures/futures/`` exercises:

* loading parquet + CSV (with header aliases) into raw ``eod`` records;
* protocol conformance — ``FuturesDataSource`` is a ``DataSource`` (runtime
  ``isinstance`` against the ``runtime_checkable`` Protocol) and slots into
  ``DataLoader`` unchanged;
* the as-of / start / end point-in-time cutoffs;
* back-adjusted ``close == adjusted_close`` so ``DataLoader.prices`` is a no-op
  adjustment over the continuous series;
* the OHLCV contract (``validate_ohlcv``) firing on a corrupt file;
* non-applicable methods (splits/dividends/…) returning sane empties;
* the commodity-futures universe helper.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rigor.data import (
    COMMODITY_FUTURES_SYMBOLS,
    DataConfig,
    DataLoader,
    DataSource,
    FuturesDataError,
    FuturesDataSource,
    commodity_futures_universe,
)
from rigor.data.futures_source import ETF_PROXY_TO_FUTURES
from rigor.data.schemas import DataContractError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "futures"


@pytest.fixture
def source() -> FuturesDataSource:
    """A source over the committed three-contract fixture directory."""
    return FuturesDataSource(FIXTURES)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_discovers_committed_contracts(source: FuturesDataSource):
    assert source.available_symbols() == ["CL.COMM", "GC.COMM", "NG.COMM"]


def test_eod_loads_parquet_contract(source: FuturesDataSource):
    bars = source.eod("GC.COMM")
    assert len(bars) == 5
    first = bars[0]
    assert first["date"] == "2020-01-02"
    assert first["close"] == 1528.0
    assert first["open_interest"] == 450000  # optional column preserved


def test_eod_loads_csv_with_vendor_headers(source: FuturesDataSource):
    # NG.csv uses Date/Open/High/Low/Settle/Vol/OI — the alias map normalises them.
    bars = source.eod("NG.COMM")
    assert len(bars) == 5
    assert bars[0]["close"] == 2.12  # Settle -> close
    assert bars[0]["volume"] == 120000  # Vol -> volume
    assert bars[0]["open_interest"] == 1100000  # OI -> open_interest


def test_symbol_suffix_optional(source: FuturesDataSource):
    # CL and CL.COMM resolve to the same CL.parquet file.
    assert source.eod("CL") == source.eod("CL.COMM")


def test_unknown_symbol_returns_empty(source: FuturesDataSource):
    assert source.eod("ZZZ.COMM") == []


def test_absent_directory_is_inert(tmp_path: Path):
    src = FuturesDataSource(tmp_path / "does_not_exist")
    assert src.available_symbols() == []
    assert src.eod("GC.COMM") == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_satisfies_datasource_protocol(source: FuturesDataSource):
    # DataSource is @runtime_checkable, so isinstance verifies the method set.
    assert isinstance(source, DataSource)


def test_has_every_protocol_method(source: FuturesDataSource):
    for name in (
        "eod", "splits", "dividends", "earnings", "fundamentals", "funding",
        "fred", "intraday", "historical_constituents", "exchange_symbols",
    ):
        assert callable(getattr(source, name))


def test_drops_into_dataloader(source: FuturesDataSource, tmp_path: Path):
    cfg = DataConfig(api_key="OFFLINE", cache_dir=tmp_path, as_of="2026-06-01")
    dl = DataLoader(cfg, source=source)
    px = dl.prices("GC.COMM")
    assert not px.empty
    assert len(px) == 5
    # Back-adjusted: adjusted close equals the file's close (no split/div adj).
    assert px["adj_close"].iloc[-1] == pytest.approx(1608.0)
    assert px["close"].iloc[-1] == pytest.approx(1608.0)


# ---------------------------------------------------------------------------
# Point-in-time safety
# ---------------------------------------------------------------------------

def test_as_of_cutoff_drops_future_bars():
    src = FuturesDataSource(FIXTURES, as_of="2020-01-06")
    dates = [b["date"] for b in src.eod("GC.COMM")]
    assert dates == ["2020-01-02", "2020-01-03", "2020-01-06"]  # 01-07/01-08 dropped


def test_end_argument_overrides_as_of():
    src = FuturesDataSource(FIXTURES, as_of="2020-01-08")
    dates = [b["date"] for b in src.eod("GC.COMM", end="2020-01-03")]
    assert dates == ["2020-01-02", "2020-01-03"]


def test_start_argument_trims_lower_bound(source: FuturesDataSource):
    dates = [b["date"] for b in source.eod("GC.COMM", start="2020-01-07")]
    assert dates == ["2020-01-07", "2020-01-08"]


def test_no_cutoff_returns_full_history(source: FuturesDataSource):
    # as_of=None and no end -> every bar (PIT bounding deferred to DataLoader).
    assert len(source.eod("GC.COMM")) == 5


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

def test_corrupt_file_fails_validation(tmp_path: Path):
    bad = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03"],
        "open": [100.0, 101.0],
        "high": [90.0, 104.0],   # high < low on row 0
        "low": [95.0, 100.0],
        "close": [98.0, 103.0],
    })
    bad.to_csv(tmp_path / "BAD.csv", index=False)
    src = FuturesDataSource(tmp_path)
    with pytest.raises(DataContractError, match=r"OHLC violation: BAD.COMM 2020-01-02"):
        src.eod("BAD.COMM")


def test_validation_can_be_disabled(tmp_path: Path):
    bad = pd.DataFrame({
        "date": ["2020-01-02"],
        "open": [100.0], "high": [90.0], "low": [95.0], "close": [98.0],
    })
    bad.to_csv(tmp_path / "BAD.csv", index=False)
    src = FuturesDataSource(tmp_path, validate=False)
    assert len(src.eod("BAD.COMM")) == 1  # no contract enforced


def test_missing_required_column_raises(tmp_path: Path):
    # No close/Settle column -> structural error before the contract runs.
    incomplete = pd.DataFrame({
        "date": ["2020-01-02"], "open": [1.0], "high": [2.0], "low": [0.5],
    })
    incomplete.to_csv(tmp_path / "NOCLOSE.csv", index=False)
    src = FuturesDataSource(tmp_path)
    with pytest.raises(FuturesDataError, match="missing required column"):
        src.eod("NOCLOSE.COMM")


# ---------------------------------------------------------------------------
# Non-applicable methods return sane empties
# ---------------------------------------------------------------------------

def test_inapplicable_methods_return_empty(source: FuturesDataSource):
    assert source.splits("GC.COMM") == []
    assert source.dividends("GC.COMM") == []
    assert source.earnings("GC.COMM") == {}
    assert source.fundamentals("GC.COMM") == {}
    assert source.funding("GC.COMM") == []
    assert source.fred("DGS10") == []
    assert source.intraday("GC.COMM") == []
    assert source.historical_constituents("GSPC.INDX") == {}
    assert source.exchange_symbols(delisted=True) == []


def test_exchange_symbols_enumerates_present_contracts(source: FuturesDataSource):
    codes = {row["Code"] for row in source.exchange_symbols()}
    assert codes == {"GC", "CL", "NG"}


# ---------------------------------------------------------------------------
# Universe helper
# ---------------------------------------------------------------------------

def test_universe_symbols_are_comm_suffixed():
    universe = commodity_futures_universe()
    assert all(s.endswith(".COMM") for s in universe)
    assert "GC.COMM" in universe and "CL.COMM" in universe and "NG.COMM" in universe
    assert tuple(universe) == COMMODITY_FUTURES_SYMBOLS


def test_universe_meta_documents_proxies():
    meta = commodity_futures_universe(with_meta=True)
    gold = next(r for r in meta if r["symbol"] == "GC.COMM")
    assert gold["root"] == "GC"
    assert gold["etf_proxy"] == "GLD"
    assert "Gold" in gold["description"]


def test_etf_proxy_map_round_trips():
    # Each ETF proxy points at exactly one continuous-futures symbol.
    assert ETF_PROXY_TO_FUTURES["USO"] == "CL.COMM"
    assert ETF_PROXY_TO_FUTURES["GLD"] == "GC.COMM"
    assert set(ETF_PROXY_TO_FUTURES.values()) == set(COMMODITY_FUTURES_SYMBOLS)
