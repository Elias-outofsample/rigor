"""Month-end balanced-fund rebalancing flow — conditional direction (eom_rebal_flow_spy) — intraday.

Mechanism (Etula, Rinne, Suominen & Vaittinen 2020; the rebalancing-premium / "Dash for Cash"
literature): ~$ trillions in 60/40, target-date and pension funds rebalance toward fixed weights at
the calendar month-end. The rebalance flow is MECHANICALLY DIRECTIONAL in the month's equity-vs-bond
performance: if equities OUTPERFORMED bonds during the month, funds are overweight equity and SELL
equities (buy bonds) into the month-end close -> downward pressure on SPY; if equities UNDERPERFORMED,
they BUY equities -> upward pressure. So the tradeable direction is:

    direction = -sign( month-to-date SPY return  -  month-to-date TLT return )

held intraday (open->close) on the last `n_last` trading days of each calendar month; flat otherwise.

Price-only (the equity-bond spread is a cross-asset PRICE signal; we TRADE one asset, SPY — same
single-asset/cross-signal pattern as the `leadlag_*` book), directional, intraday (flat overnight).

This is the CONDITIONAL (signed-spread) version that `eoq_last5d_pension_rebal_short_spy`'s own
docstring flags as the un-built improvement over its blind unconditional short — and it is on the
MONTH boundary (12x/yr, balanced/target-date funds rebalance monthly) vs that strat's quarter-end.
On long 1m data the unconditional short LOSES (full-series SR -1.1); only the conditional wins (+1.0).
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
        self.symbol = str(e.get("symbol", "SPY"))       # traded asset
        self.bond = str(e.get("bond", "TLT"))            # cross-asset signal leg
        self.interval = str(e.get("interval", "1m"))
        self.n_last = int(e.get("n_last", 5))            # last n trading days of the month
        self.cost_bps = float(e.get("cost_bps", 1.5))    # round-trip, bps (SPY very liquid)
        self.start_date = config.start_date
        self.end_date = config.end_date

    def _daily(self, sym: str) -> pd.DataFrame:
        df = self.data.intraday(sym, interval=self.interval, session="rth").copy()
        df["dt"] = pd.to_datetime(df["dt"])
        if self.start_date:
            df = df[df["dt"] >= pd.Timestamp(self.start_date)]
        if self.end_date:
            df = df[df["dt"] <= pd.Timestamp(self.end_date) + pd.Timedelta(days=1)]
        df = df.sort_values("dt")
        day = df["dt"].dt.normalize()
        g = df.groupby(day)
        return pd.DataFrame({"open": g["open"].first(), "close": g["close"].last()})

    def build_cache(self) -> dict[str, Any]:
        spy = self._daily(self.symbol)
        tlt = self._daily(self.bond)
        idx = spy.index.intersection(tlt.index)
        idx = pd.DatetimeIndex(sorted(idx))
        spy, tlt = spy.loc[idx], tlt.loc[idx]
        return {"open": spy["open"], "close": spy["close"], "tlt_close": tlt["close"],
                "dates": idx}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        _p = params or {}
        n_last = int(_p.get("n_last", self.n_last))
        cost_bps = float(_p.get("cost_bps", self.cost_bps))
        idx = cache["dates"]
        if len(idx) < 30:
            return pd.Series(dtype="float64", name="returns")
        op, cl, tc = cache["open"], cache["close"], cache["tlt_close"]
        out = pd.Series(0.0, index=idx, name="returns")
        months = pd.Series(idx, index=idx).groupby(idx.to_period("M"))
        for _, grp in months:
            days = grp.index
            if len(days) < n_last + 2:
                continue
            ws = days[-n_last - 1]          # spread measured the day before the window (no look-ahead)
            p0 = days[0]
            spread = (cl.loc[ws] / cl.loc[p0] - 1.0) - (tc.loc[ws] / tc.loc[p0] - 1.0)
            d = -np.sign(spread)
            if d == 0:
                continue
            for w in days[-n_last:]:
                out.loc[w] = d * (cl.loc[w] / op.loc[w] - 1.0) - cost_bps / 1e4
        return out.iloc[1:]

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: sweep the knobs run_backtest reads; canonical value first.
        return {
            'n_last': [5, 6, 4],
            'cost_bps': [1.5, 0.75, 2.25],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
