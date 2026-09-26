"""Faber GTAA — 5-asset Global Tactical Asset Allocation — trend_following.

Faber (2007): a fixed basket of five asset-class ETFs (SPY US stocks, EFA foreign
stocks, IEF bonds, VNQ real estate, GSG commodities). Each month-end, hold a sleeve
equal-weight (1/N over the *active* sleeves) when its monthly close is above its SMA
over ``sma_months`` months; otherwise that sleeve is cash. No active sleeve -> cash.

The monthly trend DECISION is taken on monthly bars, but the position is **held
across the month's daily bars** (weights stamped at each month-end trading day on a
DAILY close panel), so the engine produces a DAILY net-return series — the
momentum_s7 monthly-rebalance / daily-MTM pattern.

Ported from the TradingDesk EGB engine; operating param = engine default
sma_months=10. Deviation: total-return adjusted prices; a single commission_bps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.assets = list(e.get("assets", ["SPY", "EFA", "IEF", "VNQ", "GSG"]))
        self.sma_months = int(e.get("sma_months", 10))

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        cols = {}
        for sym in self.assets:
            df = self.data.prices(sym, start=start, end=end)
            if not df.empty:
                cols[sym] = df.set_index(pd.DatetimeIndex(df["date"]))["adj_close"]
        close = pd.DataFrame(cols).sort_index()[self.assets].dropna(how="all")
        return {"dates": close.index, "close": close.astype("float64")}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        _p = params or {}
        sma_months = int(_p.get("sma_months", self.sma_months))
        close = cache["close"]
        cal = close.index
        # Monthly trend decision (causal: SMA on monthly closes).
        monthly = close.resample("ME").last()
        sma = monthly.rolling(sma_months).mean()
        above = (monthly > sma).astype(float)
        n_active = above.sum(axis=1).replace(0, np.nan)
        w_monthly = above.div(n_active, axis=0).fillna(0.0)
        w_monthly.index = w_monthly.index.to_period("M")

        rebal = pd.Series(cal, index=cal).resample("ME").last().dropna()
        rebal_dates = pd.DatetimeIndex(rebal.to_numpy())
        weights = pd.DataFrame(0.0, index=rebal_dates, columns=close.columns)
        for d in rebal_dates:
            per = d.to_period("M")
            if per in w_monthly.index:
                weights.loc[d] = w_monthly.loc[per].to_numpy()
        return simulate_weights(
            weights, close, commission_bps=self.config.commission_bps
        )

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: sweep the knobs run_backtest reads; canonical value first.
        return {
            'sma_months': [10, 5, 15],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
