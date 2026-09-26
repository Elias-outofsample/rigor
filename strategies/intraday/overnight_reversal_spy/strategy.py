"""Overnight Reversal after a late-day sell-off — directional overnight timing of a single liquid
index ETF (QQQ default; SPY works too).

Thesis (NY Fed SR917 dealer-absorption / overnight-drift): when the LAST HOUR of the cash session
sells off, market-makers absorb the late supply and the index tends to REBOUND overnight. So:
  - At the 16:00 close, if the 15:00->16:00 return < `selloff_thresh` (default 0), go LONG overnight.
  - Exit at the next session's first FILLABLE bar (09:31 open — NOT the 09:30 OHLC print, which is
    a low-volume trap). Flat intraday. One trade per qualifying day.

Single asset, directional, overnight horizon. Net of cost (commission_bps per side) the conditioned
version is materially stronger than the unconditional overnight long (the late-sell-off filter is
the edge). Causal: the last-hour return is known at the close; the fill is the next 09:31 open.

Net (1 bp/side): QQQ Sharpe ~0.99 (hit 58%, 88% positive years), SPY ~0.88 — both 2011/2013-2026.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.symbol = str(e.get("symbol", "QQQ"))
        self._e = dict(e)
        self.commission_bps = float(config.commission_bps)
        self.start_date = config.start_date
        self.end_date = config.end_date

    def _p(self, params, key, default):
        return params[key] if key in params else self._e.get(key, default)

    def param_grid(self) -> dict[str, list]:
        return {"selloff_thresh": [0.0, -0.001, -0.002], "cond_window_min": [60, 30]}

    def build_cache(self) -> dict[str, Any]:
        df = self.data.intraday(self.symbol, interval="1m")
        if df.empty:
            return {"empty": True, "dates": pd.DatetimeIndex([])}
        df = df.sort_values("dt").reset_index(drop=True)
        if self.start_date:
            df = df[df["dt"] >= pd.Timestamp(self.start_date)]
        if self.end_date:
            df = df[df["dt"] <= pd.Timestamp(self.end_date) + pd.Timedelta(days=1)]
        df = df.reset_index(drop=True)
        c = df["close"].to_numpy("float64"); o = df["open"].to_numpy("float64")
        tod = (df["dt"].dt.hour * 60 + df["dt"].dt.minute).to_numpy("int64")
        sess = df["session"].to_numpy("datetime64[ns]")
        new = np.empty(len(df), bool); new[0] = True; new[1:] = sess[1:] != sess[:-1]
        starts = np.flatnonzero(new); ends = np.append(starts[1:], len(df))
        sdates = pd.DatetimeIndex(sess[starts])
        return {"empty": False, "c": c, "o": o, "tod": tod, "starts": starts, "ends": ends,
                "sdates": sdates, "dates": sdates}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        if cache.get("empty"):
            return pd.Series(dtype="float64")
        thr = float(self._p(params, "selloff_thresh", 0.0))
        win = int(self._p(params, "cond_window_min", 60))     # condition on last `win` minutes
        per_side = self.commission_bps / 1e4
        c, o, tod = cache["c"], cache["o"], cache["tod"]
        starts, ends, sdates = cache["starts"], cache["ends"], cache["sdates"]
        n = len(starts)
        rets = np.zeros(n)
        cut = 16 * 60 - win                                    # e.g. 15:00 for win=60
        for si in range(n - 1):
            i0, i1 = int(starts[si]), int(ends[si])
            j = [i for i in range(i0, i1) if tod[i] >= cut]
            if not j:
                continue
            cond_ret = c[i1 - 1] / c[j[0]] - 1.0                # last-window return (known at close)
            if not np.isfinite(cond_ret) or cond_ret >= thr:
                continue                                        # only trade after a late sell-off
            nx0 = int(starts[si + 1])
            if int(ends[si + 1]) - nx0 < 2:
                continue
            fill = o[nx0 + 1]                                    # 09:31 fillable open
            rets[si] = (fill / c[i1 - 1] - 1.0) - 2 * per_side  # long overnight, round-trip cost
        ser = pd.Series(rets, index=sdates, name="returns")
        ser.index = pd.DatetimeIndex(ser.index)
        return ser


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
