"""Rigor data layer: single-source EODHD, reproducible by design.

Typical use:

    from rigor.data import DataConfig, DataLoader

    cfg = DataConfig.from_env(as_of="2026-06-01")   # pin the as-of date
    dl = DataLoader(cfg)

    px = dl.prices("AAPL", start="2010-01-01")       # adjusted OHLCV
    universe = dl.load_universe("GSPC.INDX", day="2008-06-30")  # PIT members
"""

from .adjust import adjust_ohlcv, parse_dividends, parse_splits
from .client import EODHDClient, EODHDError
from .config import DataConfig, get_api_key, load_env
from .datasource import DataSource
from .futures_loader import (
    CATALOG,
    DATA_DIR,
    FUTURES_SYMBOLS,
    SESSIONS,
    FuturesSpec,
    LocalFuturesLoader,
    build_loader,
)
from .futures_source import (
    COMMODITY_FUTURES,
    COMMODITY_FUTURES_SYMBOLS,
    ETF_PROXY_TO_FUTURES,
    FuturesDataError,
    FuturesDataSource,
    commodity_futures_universe,
)
from .loader import DataLoader
from .pit_membership import SharadarMembership, TickerMapper
from .schemas import DataContractError, validate_ohlcv, validate_returns
from .universe import all_symbols_ever, members_on, parse_constituents

__all__ = [
    "DataConfig",
    "DataLoader",
    "DataSource",
    "build_loader",
    "LocalFuturesLoader",
    "CATALOG",
    "DATA_DIR",
    "FUTURES_SYMBOLS",
    "SESSIONS",
    "FuturesSpec",
    "FuturesDataSource",
    "FuturesDataError",
    "commodity_futures_universe",
    "COMMODITY_FUTURES",
    "COMMODITY_FUTURES_SYMBOLS",
    "ETF_PROXY_TO_FUTURES",
    "EODHDClient",
    "EODHDError",
    "adjust_ohlcv",
    "parse_splits",
    "parse_dividends",
    "parse_constituents",
    "members_on",
    "all_symbols_ever",
    "SharadarMembership",
    "TickerMapper",
    "DataContractError",
    "validate_ohlcv",
    "validate_returns",
    "get_api_key",
    "load_env",
]
