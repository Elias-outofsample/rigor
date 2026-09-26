"""Thin, typed client over the EODHD REST API.

Returns *raw* parsed JSON (lists/dicts) exactly as EODHD provides it. Parsing
into DataFrames, caching, and adjustment happen in higher layers -- this keeps
the cached payload faithful to the vendor response, which matters for
reproducibility.

All requests:
  * go through ``truststore`` TLS (verification stays on),
  * are bounded by the as-of date where the endpoint supports it,
  * retry transient failures with backoff.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import DataConfig
from .tls import enable_tls


class EODHDError(RuntimeError):
    """Raised for non-transient EODHD API errors (4xx other than rate limit)."""


class EODHDClient:
    def __init__(self, config: DataConfig):
        self.cfg = config
        enable_tls()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "rigor/1.0"})
        self._last_request_ts = 0.0

    # -- low-level ---------------------------------------------------------
    def _get(self, path: str, **params: Any) -> Any:
        params.setdefault("api_token", self.cfg.api_key)
        params.setdefault("fmt", "json")
        url = f"{self.cfg.base_url}/{path}"

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.timeout)
            except requests.RequestException as exc:  # network/TLS/timeout
                last_exc = exc
                time.sleep(0.5 * (2**attempt))
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                # rate limited (429) or transient server error (5xx, e.g. 502/503/504)
                # -> back off and retry rather than fail the whole run.
                last_exc = EODHDError(f"{resp.status_code} for {path}: {resp.text[:120]}")
                time.sleep(1.0 * (2**attempt))
                continue
            if resp.status_code == 404:
                return []  # unknown symbol -> empty, not fatal
            raise EODHDError(f"{resp.status_code} for {path}: {resp.text[:200]}")

        raise EODHDError(f"request failed after retries for {path}: {last_exc}")

    def _throttle(self) -> None:
        gap = self.cfg.min_request_interval_s
        if gap <= 0:
            return
        wait = gap - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- endpoints ---------------------------------------------------------
    def eod(self, symbol: str, *, start: str | None = None, end: str | None = None) -> list[dict]:
        """Raw daily OHLCV (unadjusted close + adjusted_close) for one symbol.

        ``end`` defaults to the config as-of date so no future bars leak in.
        """
        params: dict[str, Any] = {"order": "a"}  # ascending by date
        if start:
            params["from"] = start
        params["to"] = end or self.cfg.as_of_date
        data = self._get(f"eod/{self._sym(symbol)}", **params)
        return data if isinstance(data, list) else []

    def splits(self, symbol: str) -> list[dict]:
        data = self._get(f"splits/{self._sym(symbol)}", to=self.cfg.as_of_date)
        return data if isinstance(data, list) else []

    def dividends(self, symbol: str) -> list[dict]:
        data = self._get(f"div/{self._sym(symbol)}", to=self.cfg.as_of_date)
        return data if isinstance(data, list) else []

    def historical_constituents(self, index_symbol: str) -> dict:
        """Point-in-time index membership (StartDate/EndDate/IsActiveNow/IsDelisted)."""
        data = self._get(
            f"fundamentals/{index_symbol}", filter="HistoricalTickerComponents"
        )
        return data if isinstance(data, dict) else {}

    def earnings(self, symbol: str) -> dict:
        """Historical quarterly earnings: reportDate (announcement), period date,
        epsActual / epsEstimate. Keyed by fiscal-period end date."""
        data = self._get(f"fundamentals/{self._sym(symbol)}", filter="Earnings::History")
        return data if isinstance(data, dict) else {}

    def fundamentals(self, symbol: str) -> dict:
        """Full EODHD fundamentals payload for one symbol (General/Highlights/Financials
        [Balance_Sheet/Income_Statement/Cash_Flow quarterly+yearly]/outstandingShares/...).

        Returned verbatim; parsing into point-in-time factor frames happens in the loader.
        Statements carry ``filing_date``, so downstream factors can be made PIT-safe.
        """
        data = self._get(f"fundamentals/{self._sym(symbol)}")
        return data if isinstance(data, dict) else {}

    def funding(self, symbol: str, *, start: str = "2019-01-01") -> list[dict]:
        """Binance USDT-perpetual funding-rate history (8-hourly), paginated.

        Normalised to ``[{"fundingTime": <ISO str>, "fundingRate": <float>}]`` so
        the cached shape is identical whether fetched live or seeded from a file.
        """
        import pandas as pd  # local: only this endpoint needs it

        base = "https://fapi.binance.com/fapi/v1/fundingRate"
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        out: list[dict] = []
        while True:
            self._throttle()
            params: dict[str, str | int] = {"symbol": symbol, "startTime": start_ms, "limit": 1000}
            resp = self._session.get(base, params=params, timeout=self.cfg.timeout)
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            for r in batch:
                out.append({
                    "fundingTime": pd.Timestamp(int(r["fundingTime"]), unit="ms").isoformat(),
                    "fundingRate": float(r["fundingRate"]),
                })
            if len(batch) < 1000:
                break
            start_ms = int(batch[-1]["fundingTime"]) + 1
        return out

    def perp(self, symbol: str, *, interval: str = "1d", venue: str = "binance",
             start: str = "2019-09-01", end: str | None = None) -> list[dict]:
        """Perpetual-futures OHLCV klines (default daily) from a crypto venue.

        ``venue="binance"`` -> ``fapi /fapi/v1/klines``; ``venue="bybit"`` ->
        ``v5 /v5/market/kline`` (category=linear). Both are free, no key — the same
        family as :meth:`funding`. Normalised to
        ``[{"date","open","high","low","close","volume"}]`` (``date`` is the bar's
        UTC open-time day). Bounded by ``end`` (defaulting to the as-of date) so a
        frozen snapshot never includes a post-as-of bar.
        """
        import pandas as pd  # local: only these endpoints need it

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        cutoff = pd.Timestamp(end or self.cfg.as_of_date) + pd.Timedelta(days=1)
        end_ms = int(cutoff.timestamp() * 1000) - 1  # inclusive of the as-of day
        if venue == "bybit":
            return self._bybit_perp(symbol, interval, start_ms, end_ms)
        return self._binance_perp(symbol, interval, start_ms, end_ms)

    def _binance_perp(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        """Binance USDT-perpetual klines (``fapi``), paginated forward."""
        import pandas as pd

        base = "https://fapi.binance.com/fapi/v1/klines"
        out: list[dict] = []
        while start_ms < end_ms:
            self._throttle()
            params: dict[str, str | int] = {
                "symbol": symbol, "interval": interval,
                "startTime": start_ms, "endTime": end_ms, "limit": 1500,
            }
            resp = self._session.get(base, params=params, timeout=self.cfg.timeout)
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            for k in batch:
                out.append({
                    "date": pd.Timestamp(int(k[0]), unit="ms").strftime("%Y-%m-%d"),
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                    "close": float(k[4]), "volume": float(k[5]),
                })
            if len(batch) < 1500:
                break
            start_ms = int(batch[-1][0]) + 1
        return out

    # Bybit kline interval codes (daily/hourly/etc.) keyed by our interval string.
    _BYBIT_INTERVAL = {"1d": "D", "1w": "W", "1h": "60", "5m": "5", "1m": "1"}

    def _bybit_perp(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        """Bybit USDT-perpetual (category=linear) klines (``v5``), paginated backward.

        Bybit returns newest-first in pages of <=1000; we page backward via ``end``.
        Order doesn't matter downstream (``parse_perp`` sorts + de-dups).
        """
        import pandas as pd

        base = "https://api.bybit.com/v5/market/kline"
        bint = self._BYBIT_INTERVAL.get(interval, "D")
        out: list[dict] = []
        cur_end = end_ms
        while cur_end > start_ms:
            self._throttle()
            params: dict[str, str | int] = {
                "category": "linear", "symbol": symbol, "interval": bint,
                "start": start_ms, "end": cur_end, "limit": 1000,
            }
            resp = self._session.get(base, params=params, timeout=self.cfg.timeout)
            if resp.status_code != 200:
                break
            data = resp.json()
            rows = ((data or {}).get("result") or {}).get("list") or []
            if not rows:
                break
            for k in rows:  # [startMs, open, high, low, close, volume, turnover]
                out.append({
                    "date": pd.Timestamp(int(k[0]), unit="ms").strftime("%Y-%m-%d"),
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                    "close": float(k[4]), "volume": float(k[5]),
                })
            oldest = int(rows[-1][0])  # rows are newest-first -> last is oldest
            if len(rows) < 1000:
                break
            cur_end = oldest - 1
        return out

    # Default order-flow history window (days back from the as-of date). Binance
    # Vision archives go back years, but each day's aggTrades dump is ~10-50 MB;
    # bounding the lookback keeps the one-time build tractable while still giving a
    # multi-hundred-session backtest. Overridable via the ``start`` argument.
    _OF_LOOKBACK_DAYS = 365
    # Our interval string -> pandas resample rule for aggregating raw trades.
    _OF_RESAMPLE = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}

    def order_flow(
        self, symbol: str, *, interval: str = "1m", venue: str = "binance",
        start: str | None = None, end: str | None = None,
    ) -> list[dict]:
        """Aggressor-signed order-flow bars for a crypto symbol (e.g. ``BTCUSDT``).

        Source: Binance Vision public **spot ``aggTrades`` daily dumps**
        (``data.binance.vision``), which carry the ``isBuyerMaker`` aggressor flag —
        so this is *true* signed volume, not a tick-rule approximation. Each day is
        downloaded, aggregated to ``interval`` bars, and the raw zip discarded.
        ``buy_vol`` = taker-bought qty (``isBuyerMaker == False``), ``sell_vol`` =
        taker-sold qty (``isBuyerMaker == True``); ``delta = buy_vol - sell_vol``.

        Normalised to ``[{"datetime": <ISO UTC>, "open","high","low","close",
        "volume","buy_vol","sell_vol","trades"}]`` (``delta`` is derived in the
        loader). Bounded to ``[start, end]`` (``end`` defaults to the as-of date;
        ``start`` defaults to ``_OF_LOOKBACK_DAYS`` before it) so a frozen snapshot
        never includes a post-as-of bar. Only ``venue="binance"`` is supported;
        other venues return ``[]``. Days the archive is missing are skipped.
        """
        import pandas as pd  # local: only this endpoint needs it

        if venue != "binance":
            return []
        end_day = pd.Timestamp(end or self.cfg.as_of_date).normalize()
        start_day = (
            pd.Timestamp(start).normalize() if start
            else end_day - pd.Timedelta(days=self._OF_LOOKBACK_DAYS)
        )
        rule = self._OF_RESAMPLE.get(interval, "1min")
        days = list(pd.date_range(start_day, end_day, freq="D"))

        from concurrent.futures import ThreadPoolExecutor

        out: list[dict] = []
        # Binance Vision is a static-file CDN (not the rate-limited API), so the
        # daily dumps fetch concurrently — independent of the EODHD throttle.
        with ThreadPoolExecutor(max_workers=8) as ex:
            for recs in ex.map(lambda d: self._binance_vision_of_day(symbol, d, rule), days):
                out.extend(recs)
        out.sort(key=lambda r: r["datetime"])
        return out

    def _binance_vision_of_day(self, symbol: str, day: Any, rule: str) -> list[dict]:
        """Download one day of Binance spot aggTrades and aggregate to OF bars.

        Returns ``[]`` for a missing/404/empty day or any transient error, so one
        bad day never fails the whole build. Thread-safe: uses a fresh ``requests``
        call (not the shared session) since this runs in a thread pool.
        """
        import io
        import zipfile

        import pandas as pd

        sym = symbol.upper()
        date = pd.Timestamp(day).strftime("%Y-%m-%d")
        url = (f"https://data.binance.vision/data/spot/daily/aggTrades/"
               f"{sym}/{sym}-aggTrades-{date}.zip")
        try:
            resp = requests.get(url, timeout=self.cfg.timeout,
                                headers={"User-Agent": "rigor/1.0"})
            if resp.status_code != 200 or not resp.content:
                return []
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            df = pd.read_csv(zf.open(zf.namelist()[0]), header=None,
                             usecols=[1, 2, 5, 6], names=["price", "qty", "ts", "maker"])
        except Exception:
            return []
        if df.empty:
            return []
        ts = df["ts"].astype("int64")
        if int(ts.iloc[0]) > 1_000_000_000_000_000:  # 2025+ dumps are microseconds
            ts = ts // 1000
        df = df.set_index(pd.to_datetime(ts, unit="ms", utc=True))
        maker = df["maker"].astype(bool)  # True => buyer is maker => taker SOLD
        buy = df["qty"].where(~maker, 0.0)
        sell = df["qty"].where(maker, 0.0)
        price = df["price"].resample(rule)
        bars = pd.DataFrame({
            "open": price.first(), "high": price.max(),
            "low": price.min(), "close": price.last(),
            "volume": df["qty"].resample(rule).sum(),
            "buy_vol": buy.resample(rule).sum(),
            "sell_vol": sell.resample(rule).sum(),
            "trades": df["qty"].resample(rule).count(),
        }).dropna(subset=["open"])
        # Positional unpack (name=None) is robust to the index name; column order
        # matches the DataFrame above: open, high, low, close, volume, buy, sell, trades.
        return [{
            "datetime": idx.isoformat(),
            "open": float(o), "high": float(h), "low": float(lo), "close": float(cl),
            "volume": float(v), "buy_vol": float(bv), "sell_vol": float(sv),
            "trades": int(tr),
        } for idx, o, h, lo, cl, v, bv, sv, tr in bars.itertuples(index=True, name=None)]

    @staticmethod
    def _norm_fred_value(v: str) -> float | None:
        v = (v or "").strip()
        try:
            return None if v in (".", "") else float(v)
        except ValueError:
            return None

    def _fred_fetch(self, url: str, params: dict, parse) -> list[dict]:
        """Run a FRED request with retry on transient 5xx / network errors."""
        retries = max(self.cfg.max_retries, 4)
        last_exc: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.0 * (2**min(attempt, 4)))
                continue
            if resp.status_code == 200:
                return parse(resp.text)
            if resp.status_code == 404:
                return []
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = EODHDError(f"{resp.status_code} for FRED {url}")
                time.sleep(1.0 * (2**min(attempt, 4)))
                continue
            raise EODHDError(f"{resp.status_code} for FRED: {resp.text[:200]}")
        raise EODHDError(f"FRED request failed after retries: {last_exc}")

    def fred(self, series_id: str) -> list[dict]:
        """A FRED macro series, normalised to ``[{"date": "YYYY-MM-DD", "value": <float|None>}]``
        (missing observations, ``"."`` in FRED, become ``None``). Bounded by the as-of date
        so no future observation leaks into a frozen snapshot.

        Uses the **official FRED API** (api.stlouisfed.org) when ``FRED_API_KEY`` is set --
        reliable and rate-limited -- and falls back to the public graph CSV endpoint (no key,
        but it 5xx's under load) if the key is absent or the API call fails for any reason.
        """
        import csv
        import io
        import json

        def _api_parse(text: str) -> list[dict]:
            obs = (json.loads(text) or {}).get("observations", [])
            return [{"date": o.get("date"), "value": self._norm_fred_value(o.get("value"))}
                    for o in obs]

        def _csv_parse(text: str) -> list[dict]:
            out: list[dict] = []
            for row in list(csv.reader(io.StringIO(text)))[1:]:  # skip header
                if len(row) < 2 or not row[0].strip():
                    continue
                out.append({"date": row[0].strip(), "value": self._norm_fred_value(row[1])})
            return out

        if self.cfg.fred_api_key:
            try:
                return self._fred_fetch(
                    "https://api.stlouisfed.org/fred/series/observations",
                    {"series_id": series_id, "api_key": self.cfg.fred_api_key,
                     "file_type": "json", "observation_end": self.cfg.as_of_date},
                    _api_parse,
                )
            except EODHDError:
                pass  # fall back to the public CSV endpoint below
        return self._fred_fetch(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            {"id": series_id, "coed": self.cfg.as_of_date}, _csv_parse,
        )

    def intraday(
        self, symbol: str, *, interval: str = "5m", start: str = "2000-01-01",
        end: str | None = None,
    ) -> list[dict]:
        """Intraday OHLCV bars for one symbol, paginated over the full range.

        EODHD caps each intraday request to a bounded window, so we walk the range in
        chunks (120 days for 1m, else 600) and concatenate. Bounded by the as-of date.
        Normalised to ``[{"datetime": <ISO UTC>, "open","high","low","close","volume"}]``.
        Empty windows (before the vendor's intraday history begins) are skipped.
        """
        import pandas as pd  # local: only this endpoint needs it

        base = f"https://eodhd.com/api/intraday/{self._sym(symbol)}"
        end_ts = int(pd.Timestamp(end or self.cfg.as_of_date).timestamp())
        cur = int(pd.Timestamp(start).timestamp())
        step = (120 if interval == "1m" else 600) * 86400
        out: list[dict] = []
        seen: set[int] = set()
        while cur < end_ts:
            hi = min(cur + step, end_ts)
            params: dict[str, str | int] = {"api_token": self.cfg.api_key, "interval": interval,
                                            "from": cur, "to": hi, "fmt": "json"}
            for attempt in range(self.cfg.max_retries):
                self._throttle()
                try:
                    resp = self._session.get(base, params=params, timeout=self.cfg.timeout)
                except requests.RequestException:
                    time.sleep(0.5 * (2**attempt))
                    continue
                if resp.status_code == 200:
                    batch = resp.json()
                    if isinstance(batch, list):
                        for r in batch:
                            ts = r.get("timestamp")
                            if ts in seen:
                                continue
                            seen.add(ts)
                            out.append({
                                "datetime": r.get("datetime"),
                                "open": r.get("open"), "high": r.get("high"),
                                "low": r.get("low"), "close": r.get("close"),
                                "volume": r.get("volume"),
                            })
                    break
                if resp.status_code == 429:
                    time.sleep(1.0 * (2**attempt))
                    continue
                break
            cur = hi
        return out

    def exchange_symbols(
        self, *, delisted: bool = False, exchange: str | None = None
    ) -> list[dict]:
        """Full symbol list for an exchange (active, or delisted if delisted=True)."""
        exch = exchange or self.cfg.exchange
        params = {"delisted": 1} if delisted else {}
        data = self._get(f"exchange-symbol-list/{exch}", **params)
        return data if isinstance(data, list) else []

    def delisted_symbols(self, exchange: str | None = None) -> list[dict]:
        """All delisted tickers for an exchange (for ticker-reuse resolution)."""
        return self.exchange_symbols(delisted=True, exchange=exchange)

    # -- helpers -----------------------------------------------------------
    def _sym(self, symbol: str) -> str:
        """Ensure an exchange suffix (AAPL -> AAPL.US); leave indices untouched."""
        return symbol if "." in symbol else f"{symbol}.{self.cfg.exchange}"
