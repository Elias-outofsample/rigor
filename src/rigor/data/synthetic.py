"""A zero-edge synthetic market for demos: no key, no network, no signal.

``SyntheticDataLoader`` answers ``prices(symbol)`` like the real ``DataLoader`` with a
geometric random walk per symbol (deterministic seed from the symbol name). With the
default zero drift nothing is predictable by construction, so any "edge" a strategy
shows on it is luck — which is exactly what the validation battery must detect.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class SyntheticDataLoader:
    """Duck-typed ``DataLoader`` serving deterministic random-walk prices."""

    def __init__(
        self,
        start: str = "2008-01-02",
        end: str = "2024-12-31",
        *,
        drift_annual: float = 0.0,
        vol_annual: float = 0.20,
        seed: int = 7,
    ) -> None:
        self.cfg = SimpleNamespace(exchange="US", as_of=end, as_of_date=end, offline=True)
        self._index = pd.bdate_range(start, end)
        self._mu = drift_annual / 252
        self._sigma = vol_annual / np.sqrt(252)
        self._seed = seed

    def prices(self, symbol: str, start: str | None = None, end: str | None = None,
               **_: object) -> pd.DataFrame:
        idx = self._index
        rng = np.random.default_rng(_seed(f"{self._seed}:{symbol}"))
        log_ret = rng.normal(self._mu - 0.5 * self._sigma**2, self._sigma, len(idx))
        close = 100.0 * np.exp(np.cumsum(log_ret))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wiggle = np.abs(rng.normal(0, 0.004, len(idx)))
        df = pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "open": open_, "high": np.maximum(open_, close) * (1 + wiggle),
            "low": np.minimum(open_, close) * (1 - wiggle), "close": close,
            "adj_close": close, "adjusted_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        })
        if start is not None:
            df = df[df["date"] >= str(start)]
        if end is not None:
            df = df[df["date"] <= str(end)]
        return df.reset_index(drop=True)

    def load_prices(self, symbol: str, **kw: object) -> pd.DataFrame:
        return self.prices(symbol, **kw)  # type: ignore[arg-type]

    def fred(self, series_id: str, *, pub_lag_days: int = 0) -> pd.Series:
        """A flat zero risk-free rate: the demo's Sharpe ratios are excess returns."""
        return pd.Series(0.0, index=self._index, name=series_id)
