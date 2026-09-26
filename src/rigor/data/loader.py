"""High-level data API the rest of the framework calls.

Ties together client -> cache -> deterministic adjustment -> point-in-time
universe. Strategies should depend on this, never on EODHD directly, so that
the as-of/caching/adjustment guarantees are always in force.

Design note: full per-symbol history (bounded by the as-of date) is cached once;
``start``/``end`` are applied as a *local slice*. This keeps API usage low and
makes windowing a pure in-memory operation that can't change the underlying data.
"""

from __future__ import annotations

import pandas as pd

from .adjust import Method, adjust_ohlcv
from .cache import RawCache
from .client import EODHDClient
from .config import DataConfig
from .datasource import DataSource
from .pit_membership import SharadarMembership, TickerMapper
from .schemas import validate_ohlcv
from .universe import all_symbols_ever, members_on

_OHLCV_COLS = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
_PRICE_COLS = ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close")


def clean_nonpositive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality guard: NaN out non-positive prices.

    EODHD has 0.0 / penny-tick artifacts on delisted names. Without this, a bar
    that goes 0.0 -> 9.13 (or 0.005 -> 170) produces a +inf / +3,400,000% return
    that, if held, explodes the backtest. NaNing the bad bar makes the return
    across it NaN (-> 0 downstream) instead of a spurious move.
    """
    for col in _PRICE_COLS:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = float("nan")
    return df


_EARNINGS_COLS = ["report_date", "period", "eps_actual", "eps_estimate"]


def parse_earnings(raw: dict, as_of_date: str) -> pd.DataFrame:
    """Parse EODHD Earnings::History into a tidy, point-in-time-safe frame.

    Keeps only quarters with an actual EPS that were reported on/before
    ``as_of_date`` (drops future announcements and estimate-only rows), sorted by
    fiscal period.
    """
    rows = []
    for period, v in (raw or {}).items():
        if not isinstance(v, dict):
            continue
        rd, ea = v.get("reportDate"), v.get("epsActual")
        if rd and ea is not None and str(rd) <= as_of_date:
            est = v.get("epsEstimate")
            rows.append({
                "report_date": str(rd),
                "period": str(v.get("date") or period),
                "eps_actual": float(ea),
                "eps_estimate": float(est) if est is not None else float("nan"),
            })
    if not rows:
        return pd.DataFrame(columns=_EARNINGS_COLS)
    return pd.DataFrame(rows).drop_duplicates("period").sort_values("period").reset_index(drop=True)


def parse_funding(raw: list[dict], as_of_date: str) -> pd.DataFrame:
    """Parse normalised funding records into a tidy, point-in-time-safe frame.

    ``raw`` is ``[{"fundingTime": <ISO>, "fundingRate": <float>}, ...]``. Keeps
    only periods up to ``as_of_date`` (end of day), sorted by time.
    """
    cols = ["funding_time", "funding_rate"]
    if not raw:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(raw)
    df = df.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate"})
    df["funding_time"] = pd.to_datetime(df["funding_time"], format="ISO8601")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
    df = df[(df["funding_time"] < cutoff) & df["funding_rate"].notna()]
    return df[cols].sort_values("funding_time").reset_index(drop=True)


_PERP_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def parse_perp(raw: list[dict], as_of_date: str) -> pd.DataFrame:
    """Parse raw perpetual-futures klines into a tidy, point-in-time-safe frame.

    ``raw`` is ``[{"date","open","high","low","close","volume"}, ...]`` (the Binance
    klines shape normalised by the source). Keeps bars on/before ``as_of_date``,
    numeric, date-ascending, de-duplicated. Perpetual futures have no corporate
    actions, so ``adj_close`` mirrors ``close`` — the same shape as ``prices`` so a
    strategy can treat a perp series exactly like a spot series.
    """
    if not raw:
        return pd.DataFrame(columns=_PERP_COLS)
    df = pd.DataFrame(raw)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= pd.Timestamp(as_of_date)]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df["adj_close"] = df["close"]
    return df[_PERP_COLS]


_INTRADAY_COLS = ["dt", "session", "open", "high", "low", "close", "volume"]


def parse_intraday(raw: list[dict], as_of_date: str, session: str = "rth") -> pd.DataFrame:
    """Parse normalised intraday bars into a tidy, session-aware, PIT-safe frame.

    ``session="rth"`` (default, unchanged): vendor UTC ``datetime`` -> US/Eastern, filtered to the
    regular cash session (09:30-16:00 ET); ``session`` column is the ET trading date.
    ``session="all"`` (24/7 / 24x5 assets — crypto, forex): no time filter; ``dt`` is UTC (tz-naive)
    and the ``session`` column is the UTC calendar date. Bars after the as-of date are dropped.
    """
    if not raw:
        return pd.DataFrame(columns=_INTRADAY_COLS)
    df = pd.DataFrame(raw)
    dt_utc = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if session == "all":
        df = df.assign(dt=dt_utc.dt.tz_localize(None))   # keep UTC clock for 24h assets
    else:
        df = df.assign(dt=dt_utc.dt.tz_convert("America/New_York").dt.tz_localize(None))
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Same non-positive-price guard the EOD path applies: a 0.0 / negative print
    # (vendor artifact, or a genuinely negative settle such as WTI on 2020-04-20)
    # makes every `close/entry - 1` downstream meaningless. NaN the bar instead.
    df = clean_nonpositive_prices(df)
    df = df.dropna(subset=["dt", "close"])
    if session != "all":
        tod = df["dt"].dt.time
        df = df[(tod >= pd.Timestamp("09:30").time()) & (tod < pd.Timestamp("16:00").time())]
    df = df[df["dt"] <= pd.Timestamp(as_of_date) + pd.Timedelta(days=1)]
    df["session"] = df["dt"].dt.normalize()
    return df[_INTRADAY_COLS].sort_values("dt").reset_index(drop=True)


_ORDER_FLOW_COLS = ["dt", "session", "open", "high", "low", "close",
                    "volume", "buy_vol", "sell_vol", "delta", "trades"]


def parse_order_flow(raw: list[dict], as_of_date: str) -> pd.DataFrame:
    """Parse normalised aggressor-signed order-flow bars into a tidy, PIT-safe frame.

    ``raw`` is ``[{"datetime": <ISO UTC>, "open","high","low","close","volume",
    "buy_vol","sell_vol","trades"}, ...]`` (the Binance-aggTrades shape from the
    source). Crypto trades 24h with no exchange close, so — unlike ``parse_intraday``
    — there is **no cash-session filter**: ``dt`` is kept in UTC (tz-naive) and
    ``session`` is the UTC calendar day (so cumulative delta / volume profiles reset
    each UTC day). ``delta = buy_vol - sell_vol`` is derived here. Bars after the
    as-of date are dropped.
    """
    if not raw:
        return pd.DataFrame(columns=_ORDER_FLOW_COLS)
    df = pd.DataFrame(raw)
    dt_utc = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.assign(dt=dt_utc.dt.tz_localize(None))
    for c in ("open", "high", "low", "close", "volume", "buy_vol", "sell_vol", "trades"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["dt", "close"])
    df = df[df["dt"] <= pd.Timestamp(as_of_date) + pd.Timedelta(days=1)]
    df["delta"] = df["buy_vol"] - df["sell_vol"]
    df["session"] = df["dt"].dt.normalize()
    return df[_ORDER_FLOW_COLS].sort_values("dt").reset_index(drop=True)


def _parse_statement(stmt: dict, as_of_date: str) -> pd.DataFrame:
    """One EODHD quarterly financial statement (dict keyed by period) -> tidy frame.

    Indexed by fiscal-period date, with a ``filing_date`` column; only rows actually filed
    on/before ``as_of_date`` are kept (point-in-time safe). Numeric columns coerced.
    """
    rows = [v for v in (stmt or {}).values() if isinstance(v, dict)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["filing_date"] = pd.to_datetime(df.get("filing_date"), errors="coerce")
    # if a filing_date is missing, fall back to period date + 60d (conservative PIT proxy)
    df["filing_date"] = df["filing_date"].fillna(df["date"] + pd.Timedelta(days=60))
    df = df.dropna(subset=["date"])
    df = df[df["filing_date"] <= pd.Timestamp(as_of_date)]
    for c in df.columns:
        if c not in ("date", "filing_date", "currency_symbol"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def parse_fundamentals(raw: dict, as_of_date: str) -> dict:
    """Parse the EODHD fundamentals payload into PIT-safe quarterly statement frames.

    Returns {"balance","income","cashflow"} (DataFrames indexed by fiscal period, with a
    ``filing_date`` column, filtered to filings <= as-of) and "shares" (outstanding-shares
    Series). Empty frames if the name has no fundamentals.
    """
    fin = (raw or {}).get("Financials", {}) or {}

    def stmt(name: str) -> pd.DataFrame:
        return _parse_statement(fin.get(name, {}).get("quarterly", {}), as_of_date)

    out = {
        "balance": stmt("Balance_Sheet"),
        "income": stmt("Income_Statement"),
        "cashflow": stmt("Cash_Flow"),
    }
    sh = (raw or {}).get("outstandingShares", {}).get("quarterly", {}) or {}
    rows = [v for v in sh.values() if isinstance(v, dict)]
    if rows:
        sdf = pd.DataFrame(rows)
        d = pd.to_datetime(sdf.get("dateFormatted"), errors="coerce")
        s = pd.to_numeric(sdf.get("shares"), errors="coerce")
        ser = pd.Series(s.values, index=d).dropna().sort_index()
        out["shares"] = ser[ser.index <= pd.Timestamp(as_of_date)]
    else:
        out["shares"] = pd.Series(dtype="float64")
    return out


def parse_fred(raw: list[dict], as_of_date: str, pub_lag_days: int = 0) -> pd.Series:
    """Parse normalised FRED records into a tidy, point-in-time-safe Series.

    ``raw`` is ``[{"date": <ISO>, "value": <float|None>}, ...]``. Missing values
    are dropped; the index is the observation date (optionally shifted forward by
    ``pub_lag_days`` to model publication delay); observations after ``as_of_date``
    are removed so a frozen snapshot never leaks the future.
    """
    if not raw:
        return pd.Series(dtype="float64", name="value")
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    s = df.set_index("date")["value"].sort_index()
    if pub_lag_days:
        s.index = s.index + pd.Timedelta(days=int(pub_lag_days))
    cutoff = pd.Timestamp(as_of_date)
    return s[s.index <= cutoff].rename("value")


# International indices whose frozen membership log already carries an exchange
# suffix on each ticker (e.g. ALV.DE) -- only the EODHD exchange code differs.
INTL_INDICES = {"dax40", "cac40", "eurostoxx50"}
_INTL_SUFFIX = {"DE": "XETRA"}   # membership suffix -> EODHD suffix (others match)


def _intl_symbol(ticker: str) -> str | None:
    """Exchange-coded membership ticker (ALV.DE) -> EODHD code (ALV.XETRA).

    Returns None for delisted placeholders (SLUG-YYYYMM, no exchange suffix), which
    have no live EODHD code anyway.
    """
    if "." not in ticker:
        return None
    root, suf = ticker.rsplit(".", 1)
    return f"{root}.{_INTL_SUFFIX.get(suf, suf)}"


def _to_frame(raw_eod: list[dict]) -> pd.DataFrame:
    """Raw EOD records -> tidy, ascending OHLCV frame with numeric dtypes."""
    if not raw_eod:
        return pd.DataFrame(columns=_OHLCV_COLS)
    df = pd.DataFrame(raw_eod)
    for col in _OHLCV_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_OHLCV_COLS].copy()
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


class DataLoader:
    """High-level, as-of-pinned data access over a pluggable ``DataSource``.

    ``source`` defaults to the EODHD vendor client; inject another ``DataSource``
    (a different vendor, or a fake) to swap the backend without changing the cache,
    adjustment, or point-in-time membership layers above it.
    """

    def __init__(self, config: DataConfig | None = None, source: DataSource | None = None):
        self.cfg = config or DataConfig.from_env()
        self.source = source or EODHDClient(self.cfg)
        self.cache = RawCache(self.cfg.cache_dir, self.cfg.as_of_date, offline=self.cfg.offline)
        self._eodhd_codes: set[str] | None = None
        self._membership: dict[str, SharadarMembership] = {}

    @property
    def client(self) -> DataSource:
        """Back-compat alias for the raw vendor layer. Strategies ported from the
        sibling repo call ``data.client.<endpoint>`` (e.g. ``client.intraday``); the
        equivalent here is ``source``. New code should use ``source`` directly."""
        return self.source

    # -- prices ------------------------------------------------------------
    def prices(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        method: Method = "total_return",
    ) -> pd.DataFrame:
        """Adjusted OHLCV for one symbol as a DataFrame (date-ascending).

        ``method``: "total_return" (splits+divs), "split" (splits only), or
        "none" (raw). Adjusted columns are adj_open/high/low/close/volume.
        """
        raw_eod = self.cache.get_or_fetch("eod", symbol, lambda: self.source.eod(symbol))
        raw_splits = self.cache.get_or_fetch("splits", symbol, lambda: self.source.splits(symbol))
        raw_divs = self.cache.get_or_fetch("div", symbol, lambda: self.source.dividends(symbol))

        # Enforce the data contract once per ticker on the raw OHLCV, before
        # adjustment/cleaning — corrupt bars fail here with a clear, actionable
        # error (ticker + date + rule) instead of silently poisoning the engine.
        frame = validate_ohlcv(_to_frame(raw_eod), ticker=symbol)
        df = adjust_ohlcv(frame, raw_splits, raw_divs, method)
        df = clean_nonpositive_prices(df)
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        return df.reset_index(drop=True)

    def load_prices(
        self,
        symbols: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
        method: Method = "total_return",
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Wide panel of one field across many symbols (columns=symbols, index=date)."""
        series: dict[str, pd.Series] = {}
        for sym in symbols:
            df = self.prices(sym, start=start, end=end, method=method)
            if not df.empty:
                series[sym] = df.set_index("date")[field]
        if not series:
            return pd.DataFrame()
        panel = pd.DataFrame(series).sort_index()
        panel.index.name = "date"
        return panel

    # -- earnings ----------------------------------------------------------
    def earnings(self, symbol: str) -> pd.DataFrame:
        """Reported quarterly earnings for one symbol, point-in-time safe.

        Returns columns [report_date, period, eps_actual, eps_estimate] sorted by
        fiscal period. Only rows that were actually reported on/before the as-of
        date are kept (no future announcements / estimate-only rows leak in).
        """
        raw = self.cache.get_or_fetch("earnings", symbol, lambda: self.source.earnings(symbol))
        return parse_earnings(raw, self.cfg.as_of_date)

    # -- funding (crypto perpetuals) ---------------------------------------
    def funding(self, symbol: str) -> pd.DataFrame:
        """8-hourly perpetual funding-rate history for one symbol (e.g. BTCUSDT).

        Returns columns [funding_time, funding_rate] sorted by time, point-in-time
        safe (only periods up to the as-of date). Cached like prices; seedable from
        a frozen file or fetched live from Binance.
        """
        raw = self.cache.get_or_fetch("funding", symbol, lambda: self.source.funding(symbol))
        return parse_funding(raw, self.cfg.as_of_date)

    # -- perp price (crypto perpetuals) -----------------------------------
    def perp(self, symbol: str, *, interval: str = "1d", venue: str = "binance") -> pd.DataFrame:
        """Perpetual-futures OHLCV for one symbol (e.g. ``BTCUSDT``), default daily.

        Returns columns [date, open, high, low, close, adj_close, volume] —
        the same shape as :meth:`prices`, so a perp series can be used exactly like
        a spot series (``adj_close`` mirrors ``close``: perps have no corporate
        actions). Point-in-time safe (bars up to the as-of date), cached per
        ``(venue, interval, symbol)``. Fetched free (no key) from the crypto venue:
        ``venue="binance"`` (``fapi``, the same vendor as :meth:`funding`) or
        ``venue="bybit"`` (``v5``). Pair it with :meth:`funding` and the ``.CC`` spot
        from :meth:`prices` for perp-spot basis / carry, or two venues for a
        cross-exchange spread.
        """
        # Backward-compatible cache namespace: binance keeps the original
        # ``perp_<interval>`` key; other venues are namespaced so the same symbol
        # (e.g. BTCUSDT on Binance vs Bybit) never collides.
        ns = f"perp_{interval}" if venue == "binance" else f"perp_{venue}_{interval}"
        raw = self.cache.get_or_fetch(
            ns, symbol, lambda: self.source.perp(symbol, interval=interval, venue=venue)
        )
        return parse_perp(raw, self.cfg.as_of_date)

    # -- order flow (crypto aggressor-signed) ------------------------------
    def order_flow(
        self, symbol: str, *, interval: str = "1m", venue: str = "binance"
    ) -> pd.DataFrame:
        """Aggressor-signed order-flow bars for a crypto symbol (e.g. ``BTCUSDT``).

        Returns columns [dt (tz-naive UTC), session (UTC day), open, high, low,
        close, volume, buy_vol, sell_vol, delta, trades] — *true* signed volume from
        Binance ``aggTrades`` (the ``isBuyerMaker`` flag), the closest free
        equivalent of a CME footprint. ``delta = buy_vol - sell_vol``; cumulative
        delta and volume profiles are derived per UTC session by the strategy.
        Point-in-time safe (bars up to the as-of date), cached per ``(venue,
        interval, symbol)`` like :meth:`perp` and :meth:`funding`. History is bounded
        to a recent window (see ``EODHDClient._OF_LOOKBACK_DAYS``) because each day's
        raw dump is large; this is research-grade microstructure data, not deep
        survivorship-checked history.
        """
        ns = f"order_flow_{interval}" if venue == "binance" else f"order_flow_{venue}_{interval}"
        raw = self.cache.get_or_fetch(
            ns, symbol, lambda: self.source.order_flow(symbol, interval=interval, venue=venue)
        )
        return parse_order_flow(raw, self.cfg.as_of_date)

    # -- intraday ----------------------------------------------------------
    def intraday(self, symbol: str, *, interval: str = "5m", session: str = "rth") -> pd.DataFrame:
        """Intraday OHLCV bars for one symbol, point-in-time safe (bars up to the as-of date).

        ``session="rth"`` (default): US/Eastern, regular cash session 09:30-16:00 ET only — for
        US equities/ETFs. ``session="all"``: full 24h in UTC with a UTC-date ``session`` column —
        for 24/7 crypto and 24x5 forex (the RTH filter would otherwise discard them).
        Returns columns [dt, session (date), open, high, low, close, volume]. Cached like prices.
        Note: EODHD intraday history is shallow (1m ~1y, 5m ~5y), so the frame starts
        wherever the vendor's history begins — strategies should handle the short window.
        """
        raw = self.cache.get_or_fetch(
            f"intraday_{interval}", symbol, lambda: self.source.intraday(symbol, interval=interval)
        )
        return parse_intraday(raw, self.cfg.as_of_date, session=session)

    # -- macro (FRED) ------------------------------------------------------
    def fred(self, series_id: str, *, pub_lag_days: int = 0) -> pd.Series:
        """A FRED macro series as a tidy, point-in-time-safe date-indexed Series.

        Cached like prices (fetch full history once, bounded by the as-of date).
        ``pub_lag_days`` optionally shifts each observation's *effective* date
        forward to model publication delay (default 0, matching the source
        strategies, which read each observation on its stamped date and
        forward-fill onto the trading calendar themselves).
        """
        raw = self.cache.get_or_fetch("fred", series_id, lambda: self.source.fred(series_id))
        return parse_fred(raw, self.cfg.as_of_date, pub_lag_days).rename(series_id)

    def series(self, series_id: str, *, pub_lag_days: int = 0) -> pd.Series:
        """A generic seeded alt-data time series (date-indexed, point-in-time safe).

        For published non-FRED index series (NOAA ENSO niño-3.4, Baltic Dry Index,
        Geopolitical-Risk index, ...) seeded into the ``series`` cache namespace as
        ``[{date,value}]``. Same parse/as-of/pub-lag semantics as ``fred``; no live
        fetch (these have no in-framework connector), so a miss in offline mode is a
        hard error pointing at the snapshot.
        """
        raw = self.cache.get_or_fetch("series", series_id, lambda: [])
        return parse_fred(raw, self.cfg.as_of_date, pub_lag_days).rename(series_id)

    # -- fundamentals ------------------------------------------------------
    def fundamentals(self, symbol: str) -> dict:
        """PIT-safe quarterly financial statements + shares for one symbol.

        Returns {"balance","income","cashflow","shares"} (see ``parse_fundamentals``),
        cached like prices. The full vendor payload is cached verbatim under the
        ``fundamentals`` namespace; parsing/PIT-filtering happens here against the as-of date.
        """
        raw = self.cache.get_or_fetch(
            "fundamentals", symbol, lambda: self.source.fundamentals(symbol))
        return parse_fundamentals(raw, self.cfg.as_of_date)

    # -- universe ----------------------------------------------------------
    def constituents(self, index_symbol: str) -> dict:
        return self.cache.get_or_fetch(
            "constituents", index_symbol, lambda: self.source.historical_constituents(index_symbol)
        )

    def load_universe(self, index_symbol: str, *, day: str | None = None) -> list[str]:
        """Point-in-time index membership on ``day`` (defaults to the as-of date)."""
        raw = self.constituents(index_symbol)
        return members_on(raw, day or self.cfg.as_of_date, exchange=self.cfg.exchange)

    def universe_superset(self, index_symbol: str) -> list[str]:
        """Every symbol that was ever a member (survivorship-free backtest universe)."""
        return all_symbols_ever(self.constituents(index_symbol), exchange=self.cfg.exchange)

    # -- deep point-in-time universe (Sharadar membership -> EODHD prices) --
    def _eodhd_code_set(self) -> set[str]:
        """Cached set of all EODHD codes (active + delisted) for ticker mapping."""
        if self._eodhd_codes is None:
            active = self.cache.get_or_fetch(
                "symlist", "US_active", lambda: self.source.exchange_symbols()
            )
            delisted = self.cache.get_or_fetch(
                "symlist", "US_delisted", lambda: self.source.exchange_symbols(delisted=True)
            )
            codes = {r["Code"] for r in active if isinstance(r, dict) and r.get("Code")}
            codes |= {r["Code"] for r in delisted if isinstance(r, dict) and r.get("Code")}
            self._eodhd_codes = codes
        return self._eodhd_codes

    def load_universe_pit(
        self, index: str = "sp500", *, day: str | None = None, with_report: bool = False
    ):
        """Deep survivorship-free PIT membership, mapped to EODHD price symbols.

        Uses the frozen Sharadar membership reference (back to 1957) rather than
        EODHD's shallow feed. Returns EODHD symbols that were index members on
        ``day``. With ``with_report=True`` also returns mapping coverage stats.
        """
        day = day or self.cfg.as_of_date
        if index not in self._membership:
            self._membership[index] = SharadarMembership(index)
        tickers = self._membership[index].members_on(day)
        if index in INTL_INDICES:
            # International tickers are already exchange-coded; just remap the suffix.
            mapped = {t: sym for t in tickers if (sym := _intl_symbol(t))}
            symbols = sorted(set(mapped.values()))
            if with_report:
                report = {
                    "day": day, "members": len(tickers), "mapped": len(mapped),
                    "unmapped": sorted(t for t in tickers if not _intl_symbol(t)),
                }
                return symbols, report
            return symbols
        mapper = TickerMapper(self._eodhd_code_set(), exchange=self.cfg.exchange)
        mapped, unmapped = mapper.map_many(sorted(tickers))
        symbols = sorted(set(mapped.values()))
        if with_report:
            report = {
                "day": day,
                "members": len(tickers),
                "mapped": len(mapped),
                "unmapped": sorted(unmapped),
            }
            return symbols, report
        return symbols
