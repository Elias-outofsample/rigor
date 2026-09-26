"""Moving-average crossover, long/short, on one instrument — the ``rigor demo`` strategy.

Long when the fast SMA is above the slow SMA, short otherwise. The signal uses closes up
to *t*; ``simulate_weights`` applies it from *t+1*, so the strategy is causal. It has a
real parameter grid (5 fast x 5 slow windows) so the optimiser and CPCV/PBO have
something to search — and, on a zero-drift random walk, something to overfit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data: Any) -> None:
        super().__init__(config)
        self.data = data
        extra = config.extra
        self.symbol = extra.get("symbol", "NOISE")
        self.fast = int(extra.get("fast", 20))
        self.slow = int(extra.get("slow", 100))

    def default_params(self) -> dict[str, int]:
        return {"fast": self.fast, "slow": self.slow}

    def param_grid(self) -> dict[str, list[int]]:
        return {"fast": [5, 10, 20, 30, 50], "slow": [60, 100, 150, 200, 250]}

    def build_cache(self) -> dict[str, Any]:
        px = self.data.prices(self.symbol, start=self.config.start_date, end=self.config.end_date)
        close = pd.Series(px["adj_close"].to_numpy(float), index=pd.DatetimeIndex(px["date"]))
        return {"dates": close.index, "close": close}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]):
        p = {**self.default_params(), **(params or {})}
        close = cache["close"]
        fast = close.rolling(int(p["fast"])).mean()
        slow = close.rolling(int(p["slow"])).mean()
        signal = np.where(fast > slow, 1.0, -1.0)
        signal[slow.isna().to_numpy()] = 0.0
        weights = pd.DataFrame({self.symbol: signal}, index=close.index)
        prices = close.to_frame(self.symbol)
        return simulate_weights(weights, prices, commission_bps=self.config.commission_bps)


def build(config: StrategyConfig, data: Any) -> Strategy:
    return Strategy(config, data)
