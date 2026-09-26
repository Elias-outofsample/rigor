"""Market data for the portfolio builder — benchmark returns + sector exposure.

Both go through the Rigor ``DataLoader`` (EODHD), so the app uses the same data
source as the rest of the framework. Everything degrades gracefully when there's
no key / no network (returns ``None`` / ``{"Unknown": 1.0}``) so the compute core
and tests never require a live connection.

  * ``load_benchmark_returns`` — daily returns of a benchmark (default SPY) for
    portfolio beta.
  * ``strategy_sector`` — resolve a strategy's sector from, in order: a committed
    ``strategies/sectors.json`` override, the EODHD ``General::Sector`` of the
    strategy's configured ticker(s), then a name heuristic, else ``Unknown``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = ["load_benchmark_returns", "strategy_sector", "build_factor_proxies", "clear_caches"]

# ETF-proxy factor model (long minus short). Mirrors a Fama-French style set with
# liquid US ETFs the DataLoader can fetch, since Ken-French factors aren't on the
# single EODHD source. ``None`` short leg = use the long leg's return outright.
_FACTOR_ETFS = {
    "MKT": ("SPY", None),       # market excess (vs cash≈0)
    "SIZE": ("IWM", "SPY"),     # small minus large
    "VALUE": ("IWD", "IWF"),    # value minus growth
    "MOM": ("MTUM", "SPY"),     # momentum minus market
    "DURATION": ("IEF", None),  # 7-10y Treasuries (rates)
    "CREDIT": ("HYG", "IEF"),   # high-yield minus Treasuries (credit)
}

_NAME_HEURISTICS = {
    "crypto": "Crypto", "btc": "Crypto", "eth": "Crypto", "funding": "Crypto",
    "gold": "Precious Metals", "silver": "Precious Metals", "metal": "Precious Metals",
    "oil": "Energy", "gas": "Energy", "uranium": "Energy", "carbon": "Energy",
    "lumber": "Materials", "lithium": "Materials", "shipping": "Industrials",
    "semiconductor": "Technology", "bond": "Fixed Income", "credit": "Fixed Income",
    "vol": "Volatility", "vix": "Volatility", "reinsurer": "Insurance",
}


def _loader(data=None):
    if data is not None:
        return data
    try:
        from rigor.data import DataConfig, DataLoader
        return DataLoader(DataConfig.from_env())
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_benchmark_returns(ticker: str = "SPY", *, as_of: str | None = None) -> pd.Series | None:
    """Daily returns of ``ticker`` via the Rigor DataLoader, or ``None`` offline."""
    data = _loader()
    if data is None:
        return None
    try:
        px = data.prices(ticker, start="2000-01-01", end=as_of)
        s = px.set_index(pd.DatetimeIndex(px["date"]))["adj_close"].astype("float64")
        return s.pct_change().dropna().rename(ticker)
    except Exception:
        return None


@lru_cache(maxsize=512)
def _eodhd_sector(ticker: str) -> str | None:
    data = _loader()
    if data is None:
        return None
    try:
        general = (data.fundamentals(ticker) or {}).get("General") or {}
        return general.get("Sector") or general.get("GicSector") or None
    except Exception:
        return None


def _name_heuristic(name: str) -> str | None:
    low = name.lower()
    for kw, sector in _NAME_HEURISTICS.items():
        if kw in low:
            return sector
    return None


def strategy_sector(slug: str, *, name: str = "", tickers: list[str] | None = None,
                    strategy_root: str | Path | None = None, data=None) -> dict:
    """Resolve a strategy's sector exposure to ``{sector: 1.0}`` (best-effort)."""
    # 1) committed override file
    if strategy_root is not None:
        override = Path(strategy_root) / "sectors.json"
        if override.exists():
            try:
                mapping = json.loads(override.read_text(encoding="utf-8"))
                if slug in mapping:
                    return {str(mapping[slug]): 1.0}
            except Exception:
                pass
    # 2) EODHD sector of the configured ticker(s)
    for t in (tickers or []):
        sec = _eodhd_sector(t) if data is None else (
            (data.fundamentals(t) or {}).get("General", {}).get("Sector"))
        if sec:
            return {str(sec): 1.0}
    # 3) name heuristic, else Unknown
    return {_name_heuristic(name or slug) or "Unknown": 1.0}


def build_factor_proxies(as_of: str | None = None) -> dict[str, pd.Series]:
    """Build the ETF-proxy factor return series for a factor regression.

    Returns ``{factor_name: daily_return_series}``. Each leg is fetched through
    the shared ``DataLoader``; a factor is included only when all its legs load.
    Returns ``{}`` offline / without a data key, so the caller can show "factor
    data unavailable" rather than fail.
    """
    fetched: dict[str, pd.Series | None] = {}

    def _ret(ticker: str) -> pd.Series | None:
        if ticker not in fetched:
            fetched[ticker] = load_benchmark_returns(ticker, as_of=as_of)
        return fetched[ticker]

    out: dict[str, pd.Series] = {}
    for factor, (long_t, short_t) in _FACTOR_ETFS.items():
        lng = _ret(long_t)
        if lng is None:
            continue
        if short_t is None:
            out[factor] = lng.rename(factor)
            continue
        sht = _ret(short_t)
        if sht is None:
            continue
        spread = (lng - sht).dropna()
        if not spread.empty:
            out[factor] = spread.rename(factor)
    return out


def clear_caches() -> None:
    load_benchmark_returns.cache_clear()
    _eodhd_sector.cache_clear()
