"""Local Databento futures/crypto intraday — the rigor data layer's best-available-data seam.

``LocalFuturesLoader`` subclasses :class:`~rigor.data.loader.DataLoader` and overrides ONLY
``intraday()`` so that, for the CME futures / crypto instruments in :data:`CATALOG`, 1-minute bars
are read from a local Databento parquet (full history) instead of EODHD. Every other method and
every non-catalogued symbol is delegated unchanged to the EODHD path via ``super()`` — so it is a
drop-in **superset** of ``DataLoader``: an equity strategy behaves identically, while a futures
strategy automatically gets the deepest, highest-resolution series available.

Why a subclass (not the advertised-but-unimplemented ``DataLoader(source=...)`` hook): the official
``parse_intraday`` hard-codes the US-equity cash window (09:30-16:00 ET) for ``session="rth"``. CME
products settle on different clocks (silver 08:25-13:25, etc.). Baking the correct per-instrument
session window into the loader makes the "wrong session window" bug structurally impossible.

This does NOT add a second *equity* price provider (see ``docs/data.md``): Databento serves
only CME futures/crypto that EODHD does not carry; equities remain single-source EODHD.

Output is byte-for-byte the framework's ``parse_intraday`` contract::

    columns = ["dt", "session", "open", "high", "low", "close", "volume"]
    dt         : tz-naive datetime64[ns]. rth -> ET wall-clock; all -> UTC. (bar timestamp = OPEN)
    session    : midnight-normalised trading date.  prices/vol : float64.

Data is one ``<SYM>_ohlcv1m.parquet`` per instrument, resolved in priority order:
(1) ``$RIGOR_FUTURES_DATA_DIR`` if set, (2) the in-repo ``data_share/databento/`` (committed on
purpose — see ``docs/futures-data.md``), (3) the legacy ``~/databento_data``. On a checkout
missing them, only *futures* requests raise ``FileNotFoundError``; equity strategies are unaffected.

Roll integrity: the continuous series is RAW-STITCHED (``instrument_id`` changes at each roll). Pure
same-day intraday strategies (flat overnight) are roll-IMMUNE; strategies reading the OVERNIGHT
return must drop roll days via :meth:`LocalFuturesLoader.roll_dates`.

Usage::

    from rigor.data import DataConfig, build_loader
    loader = build_loader(DataConfig.from_env(as_of="2026-06-01", offline=True))
    run_strategy("strategies/intraday/orb_nq", data=loader, as_of="2026-06-01")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rigor.data.config import DataConfig
from rigor.data.loader import DataLoader, clean_nonpositive_prices

# Databento parquet location, in priority order: (1) $RIGOR_FUTURES_DATA_DIR override,
# (2) data/databento/ at the repo root (gitignored: licensed data, bring your own),
# (3) ~/databento_data.
_REPO_DATABENTO = Path(__file__).resolve().parents[3] / "data" / "databento"
if os.environ.get("RIGOR_FUTURES_DATA_DIR"):
    DATA_DIR = Path(os.environ["RIGOR_FUTURES_DATA_DIR"])
elif _REPO_DATABENTO.is_dir():
    DATA_DIR = _REPO_DATABENTO
else:
    DATA_DIR = Path(os.path.expanduser("~/databento_data"))


@dataclass(frozen=True)
class FuturesSpec:
    """One catalogued instrument: best source (Databento 1m) + ET session + FTMO target."""

    session: tuple[str, str]   # (start_inclusive, end_exclusive) in ET
    ftmo_symbol: str           # matching FTMO CFD symbol (deployment target)
    coverage: str              # best-available coverage, human-readable
    note: str = ""


# Registry: symbol -> best-available source spec. End of session is EXCLUSIVE (matches `tod < end`).
CATALOG: dict[str, FuturesSpec] = {
    "ES":  FuturesSpec(("09:30", "16:00"), "US500",  "1m 2010-06+", "CME e-mini S&P 500"),
    "NQ":  FuturesSpec(("09:30", "16:00"), "US100",  "1m 2010-06+", "CME e-mini Nasdaq-100"),
    "RTY": FuturesSpec(("09:30", "16:00"), "US2000", "1m 2017-07+", "CME e-mini Russell 2000"),
    "CL":  FuturesSpec(("09:00", "14:30"), "USOIL",  "1m 2010-06+", "NYMEX WTI crude"),
    "GC":  FuturesSpec(("08:20", "13:30"), "XAUUSD", "1m 2010-06+", "COMEX gold"),
    "SI":  FuturesSpec(("08:25", "13:25"), "XAGUSD", "1m 2010-06+", "COMEX silver (floor ET)"),
    "BTC": FuturesSpec(("09:30", "16:00"), "BTCUSD", "1m 2017-12+", "CME Bitcoin futures"),
    "ETH": FuturesSpec(("09:30", "16:00"), "ETHUSD", "1m 2021-02+", "CME Ether futures"),
}
SESSIONS: dict[str, tuple[str, str]] = {sym: spec.session for sym, spec in CATALOG.items()}
FUTURES_SYMBOLS = set(CATALOG)

_NY = "America/New_York"
_RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
_OUT_COLS = ["dt", "session", "open", "high", "low", "close", "volume"]


def _parquet_path(sym: str) -> Path:
    return DATA_DIR / f"{sym}_ohlcv1m.parquet"


def _interval_to_rule(interval: str) -> str:
    """'1m'->'1min', '5m'->'5min', '30m'->'30min', '1h'->'60min'."""
    interval = interval.strip().lower()
    if interval.endswith("min"):
        return f"{int(interval[:-3])}min"
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1]) * 60}min"
    raise ValueError(f"unsupported interval {interval!r}")


class LocalFuturesLoader(DataLoader):
    """``DataLoader`` superset: catalogued CME futures/crypto from local parquet, else EODHD."""

    def __init__(self, config: DataConfig | None = None) -> None:
        super().__init__(config)
        self._raw: dict[str, pd.DataFrame] = {}
        self._intraday_cache: dict[tuple, pd.DataFrame] = {}
        self._roll_cache: dict[str, pd.DatetimeIndex] = {}

    def _raw_1m(self, sym: str) -> pd.DataFrame:
        if sym not in self._raw:
            path = _parquet_path(sym)
            if not path.exists():
                raise FileNotFoundError(
                    f"Databento parquet not found: {path} (set $RIGOR_FUTURES_DATA_DIR or obtain "
                    "the data — see docs/data.md)"
                )
            df = pd.read_parquet(
                path, columns=["open", "high", "low", "close", "volume", "instrument_id"]
            )
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self._raw[sym] = df
        return self._raw[sym]

    def intraday(self, symbol: str, *, interval: str = "1m", session: str = "rth") -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in FUTURES_SYMBOLS:
            return super().intraday(symbol, interval=interval, session=session)

        key = (sym, interval, session)
        if key in self._intraday_cache:
            return self._intraday_cache[key].copy()

        raw = self._raw_1m(sym)[["open", "high", "low", "close", "volume"]]
        if _interval_to_rule(interval) != "1min":
            raw = (
                raw.resample(_interval_to_rule(interval), label="left", closed="left")
                .agg(_RESAMPLE_AGG)
                .dropna(subset=["open"])
            )

        if session == "all":
            out = raw.copy()
            out.index = raw.index.tz_localize(None)
        else:
            et = raw.index.tz_convert(_NY)
            start_s, end_s = SESSIONS[sym]
            tod = et.time
            mask = (tod >= pd.Timestamp(start_s).time()) & (tod < pd.Timestamp(end_s).time())
            out = raw[mask].copy()
            out.index = et[mask].tz_localize(None)

        out = out.reset_index()
        if out.columns[0] != "dt":
            out = out.rename(columns={out.columns[0]: "dt"})

        cutoff = pd.Timestamp(self.cfg.as_of_date) + pd.Timedelta(days=1)
        out = out[out["dt"] <= cutoff]
        out["session"] = out["dt"].dt.normalize()
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        # Non-positive prices are real in this series (CL settled NEGATIVE on
        # 2020-04-20: 358 raw bars, low -40.x). Left in, every `exit/entry - 1`
        # across such a bar is nonsense, so apply the same guard as the EOD path.
        out = clean_nonpositive_prices(out)
        out = out.dropna(subset=["dt", "close"]).sort_values("dt").reset_index(drop=True)
        out = out[_OUT_COLS]

        self._intraday_cache[key] = out
        return out.copy()

    def roll_dates(self, symbol: str) -> pd.DatetimeIndex:
        """ET session dates on which the underlying contract (``instrument_id``) rolled.

        On these days the overnight move is mostly contract basis, not a real return — overnight
        strategies should exclude them.
        """
        sym = symbol.upper()
        if sym not in FUTURES_SYMBOLS:
            return pd.DatetimeIndex([])
        if sym in self._roll_cache:
            return self._roll_cache[sym]
        raw = self._raw_1m(sym)
        et = raw.index.tz_convert(_NY)
        start_s, end_s = SESSIONS[sym]
        tod = et.time
        mask = (tod >= pd.Timestamp(start_s).time()) & (tod < pd.Timestamp(end_s).time())
        sess_date = pd.DatetimeIndex(et[mask].tz_localize(None)).normalize()
        iid = pd.Series(raw["instrument_id"].to_numpy()[mask], index=sess_date)
        day_iid = iid.groupby(level=0).last()
        rolled = day_iid.ne(day_iid.shift())
        rolled.iloc[0] = False
        out = pd.DatetimeIndex(day_iid.index[rolled.to_numpy()])
        self._roll_cache[sym] = out
        return out


def build_loader(config: DataConfig | None = None) -> DataLoader:
    """The framework's default intraday-aware loader.

    Returns a :class:`LocalFuturesLoader` — a drop-in superset of :class:`DataLoader` serving the
    best-available source per symbol (local Databento for catalogued CME futures/crypto; EODHD,
    delegated, for everything else). ``run_strategy`` builds this by default, so strategy
    development always gets the deepest, most complete series without per-strategy wiring.
    """
    return LocalFuturesLoader(config)


__all__ = ["CATALOG", "DATA_DIR", "FUTURES_SYMBOLS", "FuturesSpec", "LocalFuturesLoader",
           "SESSIONS", "build_loader"]
