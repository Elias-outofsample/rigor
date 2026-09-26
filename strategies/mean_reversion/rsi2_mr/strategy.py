"""RSI(2) Mean Reversion — S&P 500 (rsi2_mr) — mean_reversion.

Port of an earlier research version (RSI2-MR_S10, S&P 500 variant). Daily,
on point-in-time S&P 500 members:

  Entry eligible: RSI(2) < 10  AND  close > SMA(200)  AND  raw close > $1.
  Of the eligible names, hold the top N=5 by NATR(14) (most volatile first).
  Exit a position when close > the prior day's high, OR after 20 bars.
  Long-only, equal weight across the open positions.

Expressed for the close-to-close target-weight engine: a daily state machine
produces the set of open positions each day; that set becomes equal weights.

Deviations (documented): EODHD total-return-adjusted prices for the signals
(RSI/SMA/NATR); raw close only for the $1 liquidity gate; close-to-close
execution (original used next-open + per-share commission/slippage); single
commission_bps on turnover.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.data.pit_membership import SharadarMembership, TickerMapper
from rigor.engine import simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


def _wilder_rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _wilder_natr(high, low, close, period: int) -> pd.DataFrame:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / close * 100.0


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.index = e.get("index", "sp500")
        self.rsi_p = int(e.get("rsi_period", 2))
        self.rsi_thr = float(e.get("rsi_threshold", 10))
        self.sma_p = int(e.get("sma_period", 200))
        self.natr_p = int(e.get("natr_period", 14))
        self.max_pos = int(e.get("max_positions", 5))
        self.min_price = float(e.get("min_price", 1.0))
        self.time_stop = int(e.get("time_stop_bars", 20))
        # F5 price-extension filter: close <= SMA(p) * (1 - pct). pct=0 -> off.
        self.price_ext_sma_p = int(e.get("price_ext_sma_period", 10))
        self.price_ext_pct = float(e.get("price_ext_pct", 0) or 0)

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        warmup = (pd.Timestamp(start) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")
        spy = self.data.prices("SPY", start=warmup, end=end)
        cal = pd.DatetimeIndex(spy["date"])

        rebal = pd.Series(cal, index=cal).resample("ME").last().dropna()
        members: set[str] = set()
        for d in rebal.to_numpy():
            members.update(self.data.load_universe_pit(self.index, day=str(pd.Timestamp(d).date())))
        universe = sorted(members)

        adj_close, raw_close, adj_high, adj_low = {}, {}, {}, {}
        for sym in universe:
            df = self.data.prices(sym, start=warmup, end=end)
            if df.empty:
                continue
            df = df.set_index(pd.DatetimeIndex(df["date"]))
            adj_close[sym] = df["adj_close"]
            raw_close[sym] = df["close"]
            adj_high[sym] = df["adj_high"]
            adj_low[sym] = df["adj_low"]
        cols = sorted(adj_close)
        ac = pd.DataFrame(adj_close).reindex(cal)[cols].astype("float64")
        rc = pd.DataFrame(raw_close).reindex(cal)[cols].astype("float64")
        ah = pd.DataFrame(adj_high).reindex(cal)[cols].astype("float64")
        al = pd.DataFrame(adj_low).reindex(cal)[cols].astype("float64")

        eligible = (
            (_wilder_rsi(ac, self.rsi_p) < self.rsi_thr)
            & (ac > ac.rolling(self.sma_p).mean())
            & (rc > self.min_price)
            & self._membership_matrix(cal, cols)
        )
        if self.price_ext_pct > 0:  # F5: price extended >= pct below SMA(p)
            sma_ext = ac.rolling(self.price_ext_sma_p, min_periods=self.price_ext_sma_p).mean()
            eligible = eligible & (ac <= sma_ext * (1.0 - self.price_ext_pct))
        return {
            "dates": cal,
            "start": start,
            "adj_close": ac,
            "prev_high": ah.shift(1),
            "natr": _wilder_natr(ah, al, ac, self.natr_p),
            "eligible": eligible.fillna(False),
        }

    def _membership_matrix(self, cal: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
        """Daily boolean membership (date x symbol) from the frozen Sharadar feed."""
        mem = SharadarMembership(self.index)
        mapper = TickerMapper(self.data._eodhd_code_set(), exchange=self.data.cfg.exchange)
        colset = set(cols)
        out = pd.DataFrame(False, index=cal, columns=cols)
        cal_np = cal.to_numpy()
        for ticker, spans in mem.intervals.items():
            sym = mapper.map(ticker)
            if sym not in colset:
                continue
            for start, end in spans:
                mask = (cal_np >= np.datetime64(start)) & (
                    cal_np <= np.datetime64(end) if end else np.ones(len(cal_np), bool)
                )
                out.loc[mask, sym] = True
        return out

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        _p = params or {}
        min_price = float(_p.get("min_price", self.min_price))
        if params:
            import copy as _copy
            _s = _copy.copy(self)
            _s.min_price = min_price
            cache = _s.build_cache()
        _p = params or {}
        min_price = float(_p.get("min_price", min_price))
        if params:
            import copy as _copy
            _s = _copy.copy(self)
            _s.min_price = min_price
            cache = _s.build_cache()
        cal = cache["dates"]
        ac, prev_high = cache["adj_close"], cache["prev_high"]
        natr, eligible = cache["natr"], cache["eligible"]
        start_idx = int(cal.searchsorted(np.datetime64(cache["start"])))

        held: dict[str, int] = {}                      # symbol -> bars_held
        weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
        for i in range(start_idx, len(cal)):
            t = cal[i]
            px_t, ph_t = ac.iloc[i], prev_high.iloc[i]
            # exits: close > prior-day high, or time stop
            for sym in list(held):
                c, h = px_t.get(sym, np.nan), ph_t.get(sym, np.nan)
                if held[sym] >= self.time_stop or (np.isfinite(c) and np.isfinite(h) and c > h):
                    del held[sym]
            # entries: fill open slots with highest-NATR eligible names
            slots = self.max_pos - len(held)
            if slots > 0:
                elig = eligible.iloc[i]
                cands = [s for s in elig.index[elig.to_numpy()] if s not in held]
                cands.sort(key=lambda s: natr.iat[i, natr.columns.get_loc(s)], reverse=True)
                for sym in cands[:slots]:
                    held[sym] = 0
            for sym in held:
                held[sym] += 1
            if held:
                # Fixed-fractional sizing: 1/max_pos per slot, empty slots = cash
                # (matches the original `cash / remaining_slots`; 1/len would
                # over-invest when fewer than max_pos names qualify).
                w = 1.0 / self.max_pos
                weight_rows[t] = dict.fromkeys(held, w)

        if not weight_rows:
            return pd.Series(0.0, index=cal[start_idx + 1:], name="returns")
        cols = sorted({s for w in weight_rows.values() for s in w})
        weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(list(weight_rows)), columns=cols)
        for t, w in weight_rows.items():
            for sym, wt in w.items():
                weights.at[t, sym] = wt
        prices = ac[cols].loc[cal[start_idx]:]
        return simulate_weights(weights, prices, commission_bps=self.config.commission_bps)

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: cache-bound knobs swept via per-combo cache rebuild; canonical first.
        return {
            'min_price': [1.0, 0.5, 1.5],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
