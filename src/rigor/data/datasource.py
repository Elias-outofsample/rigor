"""Vendor-agnostic data source contract.

``DataLoader`` is built on a ``DataSource`` — the low-level, vendor-specific client
that returns *raw* parsed JSON (lists/dicts) exactly as the vendor provides it. The
reference implementation is :class:`rigor.data.client.EODHDClient`; the test suite's
``_FakeClient`` is another. Injecting a different ``DataSource`` into ``DataLoader``
swaps the vendor without touching the as-of cache, split/dividend adjustment, or
point-in-time membership layers above it.

This decouples the data layer from a single vendor: the cache keys, ``OfflineCacheMiss``
semantics and PIT membership all live above the source and are unaffected by it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Raw, as-of-bounded market & reference data from one vendor.

    Every method returns vendor-native parsed JSON (lists/dicts); parsing, caching
    and adjustment happen in ``DataLoader`` above. Implementations bound their
    results by the as-of date wherever the endpoint supports it.
    """

    def eod(self, symbol: str, *, start: str | None = None,
            end: str | None = None) -> list[dict]:
        ...

    def splits(self, symbol: str) -> list[dict]:
        ...

    def dividends(self, symbol: str) -> list[dict]:
        ...

    def earnings(self, symbol: str) -> dict:
        ...

    def fundamentals(self, symbol: str) -> dict:
        ...

    def funding(self, symbol: str, *, start: str = "2019-01-01") -> list[dict]:
        ...

    def perp(self, symbol: str, *, interval: str = "1d", venue: str = "binance",
             start: str = "2019-09-01", end: str | None = None) -> list[dict]:
        ...

    def order_flow(self, symbol: str, *, interval: str = "1m", venue: str = "binance",
                   start: str | None = None, end: str | None = None) -> list[dict]:
        ...

    def fred(self, series_id: str) -> list[dict]:
        ...

    def intraday(self, symbol: str, *, interval: str = "5m", start: str = "2000-01-01",
                 end: str | None = None) -> list[dict]:
        ...

    def historical_constituents(self, index_symbol: str) -> dict:
        ...

    def exchange_symbols(self, *, delisted: bool = False) -> list[dict]:
        ...
