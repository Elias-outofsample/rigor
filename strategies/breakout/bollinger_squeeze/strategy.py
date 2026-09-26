"""Bollinger Squeeze Breakout (bollinger_squeeze) — breakout.

Single-instrument (QQQ) daily long/short volatility-contraction breakout. The
Bollinger bandwidth ((upper - lower)/MA) measures realised dispersion; when it
compresses to (near) its lowest level of the last ``sq_win`` sessions the bands
are in a "squeeze" — coiled, low-energy — which historically precedes a directional
expansion. The squeeze ARMS the strategy; the subsequent close THROUGH a band
fires the trade in that direction. The mid-band (MA20) is the trailing exit.

  Squeeze: bandwidth <= 1.05 * rolling-min bandwidth over ``sq_win`` bars.
  Long:    while armed and flat, close > upper band  -> +1 (disarm).
  Short:   while armed and flat, close < lower band  -> -1 (disarm).
  Exit:    long flattens when close < MA; short flattens when close > MA.

Causal by construction: the squeeze arm reads the PRIOR bar (t-1), and
``simulate_weights`` lags the resulting position one further bar before applying it
to returns. The position in {-1, 0, +1} IS the target weight on the instrument.

Reimplemented faithfully from an earlier research prototype on EODHD total-return-adjusted
daily OHLC (close = adj_close). Idle / research.
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
        self.symbol = e.get("symbol", "QQQ")
        self.n = int(e.get("n", 20))
        self.k = float(e.get("k", 2.0))
        self.sq_win = int(e.get("sq_win", 120))

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        px = self.data.prices(self.symbol, start=start, end=end)
        px = px.set_index(pd.DatetimeIndex(px["date"]))
        close = px["adj_close"].astype("float64")
        return {"dates": close.index, "close": close}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]):
        p = params or {}
        n = int(p.get("n", self.n))
        k = float(p.get("k", self.k))
        sq_win = int(p.get("sq_win", self.sq_win))

        close = cache["close"]
        ma = close.rolling(n).mean()
        sd = close.rolling(n).std()
        up = ma + k * sd
        dn = ma - k * sd
        bw = (up - dn) / ma
        squeeze = bw <= bw.rolling(sq_win).min() * 1.05

        cl_a = close.to_numpy()
        ma_a, up_a, dn_a = ma.to_numpy(), up.to_numpy(), dn.to_numpy()
        sq_a = squeeze.to_numpy()
        m = len(close)
        pos = np.zeros(m, dtype="float64")

        cur = 0
        armed = False
        for t in range(1, m):
            if bool(sq_a[t - 1]):
                armed = True
            if cur == 0 and armed:
                if cl_a[t] > up_a[t]:
                    cur = 1
                    armed = False
                elif cl_a[t] < dn_a[t]:
                    cur = -1
                    armed = False
            elif cur == 1 and cl_a[t] < ma_a[t]:
                cur = 0
            elif cur == -1 and cl_a[t] > ma_a[t]:
                cur = 0
            pos[t] = cur

        position = pd.Series(pos, index=close.index)
        w = pd.DataFrame({self.symbol: position})
        prices = pd.DataFrame({self.symbol: close})
        return simulate_weights(w, prices, commission_bps=self.config.commission_bps)

    def param_grid(self) -> dict[str, list]:
        return {"k": [2.0, 1.5], "sq_win": [120, 60]}


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
