"""Funding-Rate Extreme Mean-Reversion (crypto) — daily adaptation.

Ported from an earlier research prototype (build_funding_extreme_mr). The original fades
crowded perp positioning on HOURLY perp klines; the rigor data layer has 8-hourly
Binance funding history (``data.funding``) but no perp price klines, so this is a
DAILY adaptation traded on the crypto SPOT price (EODHD ``.CC``) — perp and spot
track within the basis, so funding extremes predict spot reversion too.

Signal: aggregate the 8-hourly funding to a daily rate; z-score over ``z_win`` days;
shift 1 day (no look-ahead). Fade over-extensions gated by a spot trend MA:
  funding z >= z_entry AND spot > MA(trend_win)  -> SHORT (crowded longs, over-extended up)
  funding z <= -z_entry AND spot < MA(trend_win) -> LONG  (crowded shorts, over-extended down)
  exit to flat once |z| <= z_exit (funding normalised).
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
        e = config.extra
        self.spot = str(e.get("spot_symbol", "BTC-USD.CC"))
        self.funding_symbol = str(e.get("funding_symbol", "BTCUSDT"))
        self.z_win = int(e.get("z_win", 30))
        self.z_entry = float(e.get("z_entry", 2.0))
        self.z_exit = float(e.get("z_exit", 0.5))
        self.trend_win = int(e.get("trend_win", 30))

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        px = self.data.prices(self.spot, start=start, end=end)
        close = px.set_index(pd.DatetimeIndex(px["date"]))["adj_close"].astype("float64")
        fund = self.data.funding(self.funding_symbol)
        if len(fund):
            fs = fund.set_index(pd.DatetimeIndex(fund["funding_time"]))["funding_rate"].astype("float64")
            daily_fund = fs.resample("D").sum()
            daily_fund.index = daily_fund.index.normalize()
        else:
            daily_fund = pd.Series(dtype="float64")
        return {"dates": close.index, "close": close, "daily_fund": daily_fund}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]):
        p = params or {}
        z_win = int(p.get("z_win", self.z_win))
        z_entry = float(p.get("z_entry", self.z_entry))
        z_exit = float(p.get("z_exit", self.z_exit))
        trend_win = int(p.get("trend_win", self.trend_win))

        close, cal, df = cache["close"], cache["dates"], cache["daily_fund"]
        fz = (df - df.rolling(z_win).mean()) / df.rolling(z_win).std()
        fz = fz.shift(1)
        z = fz.reindex(cal.union(fz.index)).ffill().reindex(cal)
        ma = close.rolling(trend_win).mean()
        above = close > ma

        zv, av, mav = z.to_numpy(), above.to_numpy(), ma.to_numpy()
        pos = np.zeros(len(cal))
        cur = 0.0
        for t in range(len(cal)):
            v = zv[t]
            if v != v or mav[t] != mav[t]:
                pos[t] = cur
                continue
            if cur == 0.0:
                if v >= z_entry and av[t]:
                    cur = -1.0
                elif v <= -z_entry and not av[t]:
                    cur = 1.0
            elif abs(v) <= z_exit:
                cur = 0.0
            pos[t] = cur

        w = pd.DataFrame({self.spot: pd.Series(pos, index=cal)})
        prices = pd.DataFrame({self.spot: close})
        return simulate_weights(w, prices, commission_bps=self.config.commission_bps)

    def param_grid(self) -> dict[str, list]:
        return {"z_win": [30, 60, 90], "z_entry": [2.0, 1.5, 2.5], "trend_win": [30, 50]}


def build(config: StrategyConfig, data: Any) -> Strategy:
    return Strategy(config, data)
