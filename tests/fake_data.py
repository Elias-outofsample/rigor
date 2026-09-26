"""A deterministic, no-network stand-in for ``DataLoader`` used by the strategy
smoke test. It returns well-formed (positive prices, non-zero vols, long history)
*synthetic* data for any symbol/series, so every strategy's code path can execute
without an API key or a data snapshot. The numbers are meaningless — the smoke test
only checks that strategies RUN and produce output, not that they're any good.
"""
from __future__ import annotations

import functools
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_START = "2005-01-01"
_END = "2026-06-09"
_INTRA_START = "2024-06-03"  # ~1y of 5-minute bars: enough to run, fast to generate


def _seed(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


@functools.lru_cache(maxsize=1)
def _fake_cache_dir() -> str:
    """A temp cache dir laid out like the real one, so strategies that glob the
    filesystem for their universe (e.g. funding_carry_crypto) discover something."""
    root = Path(tempfile.mkdtemp(prefix="rigor_fake_cache_"))
    fdir = root / _END / "funding"
    fdir.mkdir(parents=True, exist_ok=True)
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"):
        (fdir / f"{sym}.json").write_text("{}", encoding="utf-8")
    return str(root)


class _FakeCache:
    """Stand-in for RawCache: never caches, just runs the fetch function."""

    def get_or_fetch(self, namespace: str, key: str, fetch):  # noqa: D401
        return fetch()


class _AllSet(set):
    """A set that 'contains' everything, so TickerMapper resolves any ticker."""

    def __contains__(self, _item) -> bool:  # noqa: D401
        return True


class _FakeClient:
    def __init__(self, parent: FakeDataLoader):
        self._p = parent

    def dividends(self, symbol: str) -> list:
        idx = self._p._daily.index[::63]  # ~quarterly
        return [{"date": d.strftime("%Y-%m-%d"), "value": 0.25} for d in idx]

    def fundamentals(self, symbol: str) -> dict:
        rng = np.random.default_rng(_seed("rawfund" + symbol))
        return {"General": {"Sector": "Technology", "IsDelisted": False},
                "Highlights": {"MarketCapitalization": float(rng.uniform(2e9, 2e12))}}

    def eod(self, symbol: str, **_) -> list:
        df = self._p.prices(symbol)
        return df.assign(date=df["date"]).to_dict("records")

    def historical_constituents(self, index_symbol: str) -> dict:
        return {str(i): {"Code": s.split(".")[0], "StartDate": _START, "EndDate": None,
                         "IsActiveNow": 1, "IsDelisted": 0}
                for i, s in enumerate(self._p._universe)}

    def intraday(self, symbol: str, *, interval: str = "5m", **_) -> list:
        # Raw vendor-style intraday records the production client returns; some
        # strategies call ``data.client.intraday(...)`` directly (rather than the
        # higher-level ``data.intraday``). Reuse the loader's synthetic 24h
        # generator (session="all") so 24/7 crypto/forex and RTH-filtered equity
        # strategies both get usable bars. Extra kwargs (start/end) are ignored.
        df = self._p.intraday(symbol, interval=interval, session="all")
        return [{"datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                 "open": float(o), "high": float(hi), "low": float(lo),
                 "close": float(c), "volume": float(v)}
                for dt, o, hi, lo, c, v in zip(df["dt"], df["open"], df["high"],
                                               df["low"], df["close"], df["volume"], strict=False)]


class FakeDataLoader:
    """Duck-typed DataLoader returning deterministic synthetic data for any request."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(exchange="US", as_of_date=_END, as_of=_END,
                                   cache_dir=_fake_cache_dir())
        self.cache = _FakeCache()
        self.client = _FakeClient(self)
        self._daily = pd.DataFrame(index=pd.bdate_range(_START, _END))
        self._universe = [f"{c}{c}{c}.US" for c in "ABCDEFGHIJKL"]  # 12 fake names

    # -- prices -----------------------------------------------------------
    def prices(self, symbol: str, **_) -> pd.DataFrame:
        idx = self._daily.index
        n = len(idx)
        rng = np.random.default_rng(_seed("px" + symbol))
        close = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, n)))
        openp = np.empty(n)
        openp[0] = close[0]
        openp[1:] = close[:-1] * (1 + rng.normal(0, 0.004, n - 1))
        wig = np.abs(rng.normal(0, 0.006, n))
        high = np.maximum(openp, close) * (1 + wig)
        low = np.minimum(openp, close) * (1 - wig)
        vol = rng.integers(5e5, 8e6, n).astype(float)
        df = pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "open": openp, "high": high, "low": low, "close": close,
            "adjusted_close": close, "volume": vol,
            "adj_open": openp, "adj_high": high, "adj_low": low,
            "adj_close": close, "adj_volume": vol,
        })
        return df.reset_index(drop=True)

    def load_prices(self, symbol: str, **_) -> pd.DataFrame:
        return self.prices(symbol)

    # -- macro series -----------------------------------------------------
    def fred(self, series_id: str, *, pub_lag_days: int = 0) -> pd.Series:
        idx = self._daily.index
        rng = np.random.default_rng(_seed("fred" + series_id))
        vals = 30.0 + np.cumsum(rng.normal(0, 0.05, len(idx)))
        s = pd.Series(np.clip(vals, 0.1, None), index=idx, name=series_id)
        return s.shift(pub_lag_days) if pub_lag_days else s

    def series(self, series_id: str, *, pub_lag_days: int = 0) -> pd.Series:
        return self.fred("series:" + series_id, pub_lag_days=pub_lag_days).rename(series_id)

    # -- crypto funding ---------------------------------------------------
    def funding(self, symbol: str) -> pd.DataFrame:
        ts = pd.date_range("2019-01-01", _END, freq="8h")
        rng = np.random.default_rng(_seed("fund" + symbol))
        return pd.DataFrame({"funding_time": ts,
                             "funding_rate": rng.normal(0.0001, 0.0004, len(ts))})

    # -- perp price (crypto perpetuals) -----------------------------------
    def perp(self, symbol: str, *, interval: str = "1d", venue: str = "binance") -> pd.DataFrame:
        """Synthetic perpetual OHLCV, same shape as ``prices`` (date/OHLC/adj_close/
        volume). The ``venue`` seeds a slightly different series so a cross-exchange
        spread strategy gets two distinct legs and still builds under the smoke test."""
        df = self.prices(f"{venue}:{symbol}" if venue != "binance" else symbol)
        return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

    # -- order flow (crypto aggressor-signed) -----------------------------
    def order_flow(self, symbol: str, *, interval: str = "1m",
                   venue: str = "binance") -> pd.DataFrame:
        """Synthetic crypto order-flow bars: 24h UTC sessions with OHLCV plus
        aggressor-signed buy_vol/sell_vol/delta/trades, same columns as the real
        ``DataLoader.order_flow``. Kept compact (~160 bars/session over ~80 sessions)
        so the smoke test runs fast while still exercising the per-session signal
        loops (warmup/lookback up to ~60)."""
        rng = np.random.default_rng(_seed("of" + symbol))
        sessions = pd.bdate_range(_INTRA_START, _END)[-80:]
        bars = pd.date_range("00:00", "23:50", freq="9min")  # ~160 bars/UTC day
        rows = []
        base = 100.0
        for sess in sessions:
            steps = rng.normal(0, 0.0015, len(bars))
            c = base * np.exp(np.cumsum(steps))
            o = np.empty(len(bars))
            o[0] = c[0]
            o[1:] = c[:-1]
            wig = np.abs(rng.normal(0, 0.001, len(bars)))
            vol = rng.uniform(50, 500, len(bars))
            buy_frac = np.clip(0.5 + rng.normal(0, 0.18, len(bars)), 0.05, 0.95)
            for i in range(len(bars)):
                dt = pd.Timestamp.combine(sess.date(), bars[i].time())
                hi = max(o[i], c[i]) * (1 + wig[i])
                lo = min(o[i], c[i]) * (1 - wig[i])
                bv = vol[i] * buy_frac[i]
                sv = vol[i] - bv
                rows.append((dt, sess.normalize(), o[i], hi, lo, c[i], vol[i],
                             bv, sv, bv - sv, int(rng.integers(20, 400))))
            base = c[-1]
        cols = ["dt", "session", "open", "high", "low", "close",
                "volume", "buy_vol", "sell_vol", "delta", "trades"]
        return pd.DataFrame(rows, columns=cols)

    # -- intraday ---------------------------------------------------------
    def intraday(
        self, symbol: str, *, interval: str = "5m", session: str = "rth"
    ) -> pd.DataFrame:
        # Mirror the real DataLoader.intraday signature: session="rth" (default)
        # yields the US regular cash session (09:30-15:55 ET); session="all" yields
        # full 24h bars (00:00-23:55) with a UTC-date session column, for 24/7
        # crypto / 24x5 forex strategies. Calendar-day range for "all" so every
        # day (incl. weekends) carries bars.
        rng = np.random.default_rng(_seed("intra" + symbol + session))
        rows = []
        if session == "all":
            sessions = pd.date_range("2026-03-01", _END, freq="D")
            bars = pd.date_range("00:00", "23:55", freq="5min").time
        else:
            sessions = pd.bdate_range(_INTRA_START, _END)
            bars = pd.date_range("09:30", "15:55", freq="5min").time
        base = 100.0
        for sess in sessions:
            steps = rng.normal(0, 0.0015, len(bars))
            c = base * np.exp(np.cumsum(steps))
            o = np.empty(len(bars))
            o[0] = c[0]
            o[1:] = c[:-1]
            wig = np.abs(rng.normal(0, 0.001, len(bars)))
            for i, t in enumerate(bars):
                dt = pd.Timestamp.combine(sess, t)
                hi = max(o[i], c[i]) * (1 + wig[i])
                lo = min(o[i], c[i]) * (1 - wig[i])
                rows.append((dt, sess.normalize(), o[i], hi, lo, c[i],
                             float(rng.integers(1e4, 1e5))))
            base = c[-1]
        cols = ["dt", "session", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(rows, columns=cols)

    # -- fundamentals / earnings -----------------------------------------
    def fundamentals(self, symbol: str) -> dict:
        periods = pd.date_range("2005-03-31", _END, freq="QE")
        filing = periods + pd.Timedelta(days=45)
        rng = np.random.default_rng(_seed("fund" + symbol))
        n = len(periods)
        rev = np.cumsum(np.abs(rng.normal(50, 5, n))) + 1000
        ni = rev * rng.uniform(0.05, 0.15, n)
        income = pd.DataFrame({
            "filing_date": filing, "netIncome": ni, "totalRevenue": rev,
            "operatingIncome": ni * 1.3, "ebitda": ni * 1.6, "grossProfit": rev * 0.4,
            "ebit": ni * 1.2}, index=periods)
        balance = pd.DataFrame({
            "filing_date": filing, "totalStockholderEquity": rev * 2,
            "totalDebt": rev * 0.5, "shortLongTermDebtTotal": rev * 0.5,
            "cashAndShortTermInvestments": rev * 0.3, "cash": rev * 0.2,
            "totalAssets": rev * 3, "totalLiab": rev * 1.2}, index=periods)
        cashflow = pd.DataFrame({"filing_date": filing, "freeCashFlow": ni * 0.8,
                                 "capitalExpenditures": -rev * 0.1}, index=periods)
        shares = pd.Series(np.full(n, 1e9) + rng.normal(0, 1e6, n),
                           index=pd.Index(periods, name="dateFormatted"))
        return {"balance": balance, "income": income, "cashflow": cashflow, "shares": shares}

    def earnings(self, symbol: str) -> pd.DataFrame:
        periods = pd.date_range("2005-03-31", _END, freq="QE")
        rng = np.random.default_rng(_seed("earn" + symbol))
        eps = rng.normal(1.0, 0.3, len(periods))
        return pd.DataFrame({"report_date": periods + pd.Timedelta(days=30),
                             "period": periods, "eps_actual": eps,
                             "eps_estimate": eps + rng.normal(0, 0.1, len(periods))})

    # -- universe / membership -------------------------------------------
    def load_universe_pit(self, index: str = "sp500", *, day=None, with_report: bool = False):
        return (list(self._universe), {"coverage": 1.0}) if with_report else list(self._universe)

    def load_universe(self, index_symbol: str, *, day=None) -> list:
        return list(self._universe)

    def universe_superset(self, index_symbol: str) -> list:
        return list(self._universe)

    def constituents(self, index_symbol: str) -> dict:
        return self.client.historical_constituents(index_symbol)

    def _eodhd_code_set(self) -> set:
        return _AllSet()
