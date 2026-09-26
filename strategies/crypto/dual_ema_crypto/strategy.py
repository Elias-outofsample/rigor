"""Dual-EMA Trend (BTC + ETH) — crypto.

Long a coin when the fast EMA is above the slow EMA *and* price is above its
SMA_TREND (macro uptrend); exit when either flips. Shorts optional (disabled at the
promoted optimum). Equal-weight across BTC and ETH, long-only daily on EOD crypto.

Per-coin state machine (Clenow-style dual-EMA + macro regime):
  ema_cross = +1 if EMA(fast) > EMA(slow) else -1
  macro     = +1 if close > SMA(trend) else -1   (0 while SMA is warming up)
  enter long on a cross-up in an uptrend (or already-aligned + macro just flipped up);
  exit long on a cross-down or macro turning down.

Ported from the TradingDesk EGB engine; strict-promoted optimum EMA_FAST=12,
EMA_SLOW=28, SMA_TREND=150, ENABLE_SHORT=0 (CPCV verdict ROBUST). Per-coin target
weight = pos / N (N = number of coins); ``simulate_weights`` lags one bar and
charges commission_bps (= the engine's 0.10% commission + 5 bps slippage = 15 bps).

Note: the Rigor engine annualises on the inferred calendar (~252) rather than the
365-day crypto convention, so the reported Sharpe is ~0.83x the source's 365-day
figure (corr is unaffected).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


def _signal_series(close: pd.Series, ef: int, es: int, st: int, enable_short: bool) -> pd.Series:
    ema_f = close.ewm(span=ef, adjust=False).mean()
    ema_s = close.ewm(span=es, adjust=False).mean()
    sma = close.rolling(st).mean()
    ema_cross = np.where(ema_f > ema_s, 1, -1)
    macro = np.where(close > sma, 1, -1)
    macro = np.where(np.isnan(sma.to_numpy()), 0, macro)
    n = len(close)
    pos = np.zeros(n)
    cur = 0
    for i in range(1, n):
        ec, ecp = ema_cross[i], ema_cross[i - 1]
        mc, mcp = macro[i], macro[i - 1]
        cross_up = ec == 1 and ecp == -1
        cross_dn = ec == -1 and ecp == 1
        if cur == 1 and (cross_dn or mc == -1):
            cur = 0
        elif cur == -1 and (cross_up or mc == 1):
            cur = 0
        if cur == 0 and mc != 0:
            long_entry = (cross_up and mc == 1) or (ec == 1 and mc == 1 and mcp == -1)
            short_entry = enable_short and ((cross_dn and mc == -1) or (ec == -1 and mc == -1 and mcp == 1))
            if long_entry:
                cur = 1
            elif short_entry:
                cur = -1
        pos[i] = cur
    return pd.Series(pos, index=close.index)


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.tickers = list(e.get("tickers", ["BTC-USD.CC", "ETH-USD.CC"]))
        self.ema_fast = int(e.get("ema_fast", 12))
        self.ema_slow = int(e.get("ema_slow", 28))
        self.sma_trend = int(e.get("sma_trend", 150))
        self.enable_short = bool(int(e.get("enable_short", 0)))

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        cols = {}
        for sym in self.tickers:
            df = self.data.prices(sym, start=start, end=end)
            if not df.empty:
                cols[sym] = df.set_index(pd.DatetimeIndex(df["date"]))["adj_close"].astype("float64")
        close = pd.DataFrame(cols).sort_index()
        return {"dates": close.index, "close": close}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        _p = params or {}
        ema_fast = int(_p.get("ema_fast", self.ema_fast))
        ema_slow = int(_p.get("ema_slow", self.ema_slow))
        sma_trend = int(_p.get("sma_trend", self.sma_trend))
        enable_short = int(_p.get("enable_short", self.enable_short))
        close = cache["close"]
        n = max(len(self.tickers), 1)
        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for t in close.columns:
            s = close[t].dropna()
            pos = _signal_series(s, ema_fast, ema_slow, sma_trend, enable_short)
            weights[t] = pos.reindex(close.index).fillna(0.0) / n
        return simulate_weights(
            weights, close, commission_bps=self.config.commission_bps
        )

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: sweep the knobs run_backtest reads; canonical value first.
        return {
            'ema_fast': [12, 6, 18],
            'ema_slow': [28, 14, 42],
            'sma_trend': [150, 75, 225],
            'enable_short': [0, 1],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
