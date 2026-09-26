"""File-based futures ``DataSource`` — the on-ramp for continuous-futures data.

The framework's default vendor (EODHD) has **no futures** of any kind: every
continuous-futures ticker (ES/NQ/CL/GC/ZN/6E/``.COMM``…) returns HTTP 404
(verified 2026-06-05). That blocks commodity- and multi-asset-futures strategies,
which need another source.

This module supplies the *plumbing*, not the data. :class:`FuturesDataSource`
is a concrete :class:`~rigor.data.datasource.DataSource` that reads continuous /
back-adjusted futures OHLCV from a **local directory you populate** — one file
per continuous contract, in a documented column schema (CSV or parquet). Drop it
in behind ``DataLoader``::

    from rigor.data import DataConfig, DataLoader
    from rigor.data.futures_source import FuturesDataSource

    src = FuturesDataSource("/path/to/futures_data")   # CSV/parquet you exported
    dl = DataLoader(DataConfig.from_env(as_of="2026-06-01"), source=src)
    px = dl.prices("CL.COMM")        # continuous WTI crude, back-adjusted

Honest scope
------------
**No vendor is bundled.** Futures price history is yours to provide — export
it from Norgate / CSI / Interactive Brokers into the layout documented in
``docs/futures-data.md`` and point this source at the directory. With an empty /
absent directory the source is inert (every symbol returns no bars), so wiring it
up never breaks the default EODHD path.

Design
------
* **Point-in-time safe.** ``eod`` drops every bar after the as-of cutoff
  (``end``, defaulting to the config date), exactly like the EODHD client, so a
  frozen snapshot never leaks the future.
* **Contract-checked.** Each file is validated once with
  :func:`~rigor.data.schemas.validate_ohlcv` on load, so a corrupt continuous
  series fails loudly (ticker + date + rule) instead of poisoning a backtest.
* **Back-adjusted close is the adjusted close.** Continuous-futures files are
  already roll-adjusted, so the file's ``close`` is emitted as both ``close`` and
  ``adjusted_close``; the framework's split/dividend machinery is a no-op for
  futures (``splits``/``dividends`` return ``[]``), leaving the back-adjusted
  series untouched through ``DataLoader.prices``.
* **Only OHLCV applies.** Methods that have no meaning for futures
  (``splits``/``dividends``/``earnings``/``fundamentals``/``funding``/``perp``/
  ``fred``/``intraday``/``historical_constituents``/``exchange_symbols``) return
  empty/neutral defaults so the source still satisfies the full protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import validate_ohlcv

# Canonical OHLCV columns a futures file may carry. ``open_interest`` is an
# optional futures-specific extra (kept on the frame, ignored by the OHLCV
# contract, which constrains only open/high/low/close + clean dates).
_REQUIRED_COLS = ("date", "open", "high", "low", "close")
_OPTIONAL_COLS = ("volume", "open_interest")
# Common header spellings mapped onto the canonical schema, so a file exported
# from Norgate / CSI / IB needs no hand-editing of its header row.
_COLUMN_ALIASES = {
    "Date": "date",
    "DATE": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Settle": "close",          # CSI/Norgate settlement price -> close
    "settle": "close",
    "Volume": "volume",
    "Vol": "volume",
    "vol": "volume",
    "OpenInterest": "open_interest",
    "Open Interest": "open_interest",
    "OI": "open_interest",
    "oi": "open_interest",
}

# File extensions probed for a symbol, in priority order (parquet first: typed,
# compact, fast; CSV second: portable, human-readable).
_EXTENSIONS = (".parquet", ".csv")

# ---------------------------------------------------------------------------
# Commodity-futures universe
# ---------------------------------------------------------------------------
# The continuous-futures contracts the commodity-momentum strategies
# trades. Those strategies currently run on liquid ETF *proxies* (GLD, USO, …)
# because of the EODHD futures wall; this is the real underlying universe they
# map onto once continuous-futures files exist. ``.COMM`` is the framework's
# pseudo-exchange suffix flagging a continuous-futures symbol (so it never
# collides with an EODHD equity code). Roots follow CME/CBOT/COMEX/NYMEX/ICE
# convention. ``proxy`` is the ETF used as a stand-in when futures are unavailable.
COMMODITY_FUTURES = (
    # (continuous symbol, root, description, ETF proxy)
    ("GC.COMM", "GC", "Gold (COMEX)", "GLD"),
    ("SI.COMM", "SI", "Silver (COMEX)", "SLV"),
    ("PL.COMM", "PL", "Platinum (NYMEX)", "PPLT"),
    ("PA.COMM", "PA", "Palladium (NYMEX)", "PALL"),
    ("HG.COMM", "HG", "Copper (COMEX)", "CPER"),
    ("CL.COMM", "CL", "WTI Crude Oil (NYMEX)", "USO"),
    ("BZ.COMM", "BZ", "Brent Crude Oil (ICE)", "BNO"),
    ("NG.COMM", "NG", "Henry Hub Natural Gas (NYMEX)", "UNG"),
    ("ZC.COMM", "ZC", "Corn (CBOT)", "CORN"),
    ("ZW.COMM", "ZW", "Wheat (CBOT)", "WEAT"),
    ("ZS.COMM", "ZS", "Soybeans (CBOT)", "SOYB"),
    ("SB.COMM", "SB", "Sugar No. 11 (ICE)", "CANE"),
)

# Just the continuous-futures symbols (the universe a strategy iterates over).
COMMODITY_FUTURES_SYMBOLS = tuple(sym for sym, _root, _desc, _proxy in COMMODITY_FUTURES)

# ETF proxy -> continuous-futures symbol, to re-point an ETF-based commodity strategy
# (which loads GLD/USO/… today) at this source once futures files are supplied.
ETF_PROXY_TO_FUTURES = {
    proxy: sym for sym, _root, _desc, proxy in COMMODITY_FUTURES
}


def commodity_futures_universe(*, with_meta: bool = False) -> list[str] | list[dict]:
    """The commodity continuous-futures universe.

    Returns the ``ROOT.COMM`` symbol list (default), or — with ``with_meta`` —
    a list of ``{"symbol", "root", "description", "etf_proxy"}`` records that
    document each contract and the ETF used as a stand-in.
    """
    if not with_meta:
        return list(COMMODITY_FUTURES_SYMBOLS)
    return [
        {"symbol": sym, "root": root, "description": desc, "etf_proxy": proxy}
        for sym, root, desc, proxy in COMMODITY_FUTURES
    ]


class FuturesDataError(RuntimeError):
    """A futures data file was malformed in a way the OHLCV contract can't name.

    Used for structural problems *before* the contract runs (an unreadable file,
    or one with none of the required price columns). Bar-level contract breaches
    raise :class:`~rigor.data.schemas.DataContractError` from ``validate_ohlcv``.
    """


def _symbol_to_stem(symbol: str) -> str:
    """Map a continuous-futures symbol to its on-disk file stem.

    ``CL.COMM`` -> ``CL`` (the ``.COMM`` pseudo-exchange suffix is a framework
    convention to flag a continuous-futures symbol; the file is named for the
    root). A symbol with no suffix is used verbatim, so ``CL`` and ``CL.COMM``
    resolve to the same ``CL.{parquet,csv}`` file.
    """
    return symbol.split(".", 1)[0] if "." in symbol else symbol


class FuturesDataSource:
    """Read continuous / back-adjusted futures OHLCV from a local directory.

    One file per continuous contract, named for the root symbol (``CL.parquet``
    or ``CL.csv``), in the schema documented in ``docs/futures-data.md``. Satisfies
    the :class:`~rigor.data.datasource.DataSource` protocol so it drops straight into
    ``DataLoader`` in place of the EODHD client.

    Parameters
    ----------
    data_dir:
        Directory holding the per-contract files. May be absent or empty — the
        source is then simply inert (every symbol returns no bars), which keeps
        wiring it up from ever breaking a run that has no futures data yet.
    as_of:
        Inclusive point-in-time cutoff (``YYYY-MM-DD``). Bars dated after it are
        dropped from every ``eod`` result. ``None`` (the default) applies no
        cutoff here and defers point-in-time bounding to ``DataLoader``'s as-of
        date — pass it explicitly to bound a bare source used without a loader.
    validate:
        When ``True`` (default), each file is checked once against the OHLCV
        contract on load; a corrupt series fails loudly rather than silently.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        as_of: str | None = None,
        validate: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.as_of = as_of
        self.validate = validate
        # Per-symbol frame cache: a continuous file is parsed/validated once.
        self._frames: dict[str, pd.DataFrame] = {}

    # -- discovery ---------------------------------------------------------
    def _path_for(self, symbol: str) -> Path | None:
        """Locate the file backing ``symbol`` (parquet preferred), or ``None``."""
        stem = _symbol_to_stem(symbol)
        for ext in _EXTENSIONS:
            candidate = self.data_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def available_symbols(self) -> list[str]:
        """Sorted ``ROOT.COMM`` symbols backed by a file in ``data_dir``.

        Empty when the directory is absent — useful to confirm wiring before a
        run, and to enumerate which continuous contracts are actually present.
        """
        if not self.data_dir.is_dir():
            return []
        stems: set[str] = set()
        for ext in _EXTENSIONS:
            stems.update(p.stem for p in self.data_dir.glob(f"*{ext}"))
        return sorted(f"{stem}.COMM" for stem in stems)

    # -- parsing -----------------------------------------------------------
    def _read_file(self, path: Path) -> pd.DataFrame:
        """Read one continuous-futures file into a raw, unbounded frame."""
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return df.rename(columns=_COLUMN_ALIASES)

    def _frame_for(self, symbol: str) -> pd.DataFrame:
        """Tidy, validated, date-ascending OHLCV frame for ``symbol`` (cached).

        Returns an empty frame (with the canonical columns) when the symbol has
        no file, so a missing contract is a no-data result, never an error.
        """
        stem = _symbol_to_stem(symbol)
        cached = self._frames.get(stem)
        if cached is not None:
            return cached

        path = self._path_for(symbol)
        if path is None:
            empty = pd.DataFrame(columns=[*_REQUIRED_COLS, *_OPTIONAL_COLS])
            self._frames[stem] = empty
            return empty

        df = self._read_file(path)
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise FuturesDataError(
                f"futures file {path.name} for {symbol} is missing required "
                f"column(s) {missing}; expected {list(_REQUIRED_COLS)} "
                "(see docs/futures-data.md)"
            )

        # Normalise dtypes: dates to plain YYYY-MM-DD strings (matching the
        # EODHD client's eod output), prices/volume to numeric.
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close", *_OPTIONAL_COLS):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        keep = [c for c in (*_REQUIRED_COLS, *_OPTIONAL_COLS) if c in df.columns]
        df = df[keep]

        if self.validate:
            validate_ohlcv(df, ticker=symbol)

        self._frames[stem] = df
        return df

    # -- DataSource protocol: the one method that applies to futures -------
    def eod(
        self, symbol: str, *, start: str | None = None, end: str | None = None
    ) -> list[dict]:
        """Continuous back-adjusted daily OHLCV for ``symbol`` as raw records.

        Each record is ``{date, open, high, low, close, adjusted_close, volume
        [, open_interest]}``. ``adjusted_close`` mirrors the file's (already
        back-adjusted) ``close`` so ``DataLoader``'s adjustment layer — driven by
        the empty ``splits``/``dividends`` below — leaves the series untouched.

        Point-in-time: bars after ``end`` (defaulting to ``self.as_of``) are
        dropped; ``start`` trims the lower bound. An unknown symbol yields ``[]``.
        """
        df = self._frame_for(symbol)
        if df.empty:
            return []

        cutoff = end or self.as_of
        if cutoff is not None:
            df = df[df["date"] <= cutoff]
        if start is not None:
            df = df[df["date"] >= start]

        records: list[dict] = []
        for row in df.itertuples(index=False):
            data = row._asdict()
            close = data.get("close")
            rec: dict[str, Any] = {
                "date": data["date"],
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "close": close,
                # Back-adjusted series: adjusted_close == close (no further adj).
                "adjusted_close": close,
                "volume": data.get("volume"),
            }
            if "open_interest" in data:
                rec["open_interest"] = data["open_interest"]
            records.append(rec)
        return records

    # -- DataSource protocol: methods that do not apply to futures ---------
    # Each returns the neutral default its consumer treats as "no data", so the
    # class satisfies the full protocol and DataLoader's adjustment/PIT layers
    # become no-ops over the (already back-adjusted) continuous series.
    def splits(self, symbol: str) -> list[dict]:
        """Not applicable to continuous futures — always ``[]`` (no split events)."""
        return []

    def dividends(self, symbol: str) -> list[dict]:
        """Not applicable to futures — always ``[]`` (futures pay no dividends)."""
        return []

    def earnings(self, symbol: str) -> dict:
        """Not applicable to futures — always ``{}`` (no issuer / earnings)."""
        return {}

    def fundamentals(self, symbol: str) -> dict:
        """Not applicable to futures — always ``{}`` (no financial statements)."""
        return {}

    def funding(self, symbol: str, *, start: str = "2019-01-01") -> list[dict]:
        """Crypto-perpetual funding only — always ``[]`` for exchange futures."""
        return []

    def perp(self, symbol: str, *, interval: str = "1d", venue: str = "binance",
             start: str = "2019-09-01", end: str | None = None) -> list[dict]:
        """Crypto-perpetual klines only — always ``[]`` for exchange futures."""
        return []

    def order_flow(self, symbol: str, *, interval: str = "1m", venue: str = "binance",
                   start: str | None = None, end: str | None = None) -> list[dict]:
        """Crypto aggTrades order flow only — always ``[]`` for exchange futures."""
        return []

    def fred(self, series_id: str) -> list[dict]:
        """Macro series live on the EODHD/FRED path, not here — always ``[]``."""
        return []

    def intraday(
        self,
        symbol: str,
        *,
        interval: str = "5m",
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> list[dict]:
        """This source serves daily continuous bars only — always ``[]`` intraday."""
        return []

    def historical_constituents(self, index_symbol: str) -> dict:
        """Futures have no index membership — always ``{}`` (use a static universe)."""
        return {}

    def exchange_symbols(self, *, delisted: bool = False) -> list[dict]:
        """No vendor symbol master here; ``available_symbols`` enumerates files.

        Returns the EODHD-style ``[{"Code": ...}]`` shape for the *active* set
        (root codes of the files present), and ``[]`` for ``delisted`` — a
        continuous series has no delisting.
        """
        if delisted:
            return []
        return [{"Code": _symbol_to_stem(sym)} for sym in self.available_symbols()]
