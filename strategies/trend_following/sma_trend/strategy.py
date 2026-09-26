"""SMA Trend (SPY) (sma_trend) — trend_following.

Implements the Rigor strategy contract:
  build_cache  -> data/indicators via the DataLoader (single-source EODHD, as-of)
  run_backtest -> signal -> target weights -> simulate_weights (standard accounting)
  param_grid   -> the parameter sweep

The module-level ``build(config, data)`` factory is what the runner calls.
"""

from __future__ import annotations

import pandas as pd

from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        self.symbol = config.extra.get("symbol", "SPY")

    def build_cache(self) -> dict:
        px = self.data.prices(
            self.symbol, start=self.config.start_date, end=self.config.end_date
        )
        px = px.set_index(pd.DatetimeIndex(px["date"]))
        return {"dates": px.index, "close": px["adj_close"].astype("float64")}

    def param_grid(self) -> dict:
        # TODO: replace with this strategy's real parameter sweep.
        return {"sma_window": [100, 200]}

    def run_backtest(self, cache: dict, params: dict) -> pd.Series:
        # TODO: replace with this strategy's real signal -> target weights.
        close = cache["close"]
        window = int(params["sma_window"])
        weight = (close > close.rolling(window).mean()).astype("float64")
        weights = pd.DataFrame({self.symbol: weight}, index=close.index)
        prices = pd.DataFrame({self.symbol: close}, index=close.index)
        return simulate_weights(
            weights, prices, commission_bps=self.config.commission_bps
        )


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
