"""Global TSMOM (S21) — time-series momentum across global ETFs.

Port of an earlier research version (CommodityTSMOM_S21, "Global ETFs" variant): classic
time-series (trend-following) momentum across 22 global ETFs spanning 8 asset classes
(US/intl/EM equity, govt/corp/HY/intl/EM bonds, gold/silver/commodities, REITs).

Academic basis: Moskowitz–Ooi–Pedersen (2012) "Time Series Momentum"; Hurst–Ooi–Pedersen
(2017) "A Century of Evidence on Trend-Following".

Signal (per ETF, independently): go long if the trailing 12-month (252d) return > 0,
else hold cash. Monthly rebalance (month-end). Long-only.

Sizing: inverse-volatility weights (1 / 20d annualised realised vol, vol floored at 3%),
scaled so the (diagonal) portfolio vol hits a 10% annual target, each name capped at
15% of NAV and total gross leverage capped at 2.0x. Daily returns are winsorised at
+/-10%; the unallocated sleeve earns nothing (cash). Era-based rebalance costs on
turnover.

Fixed monthly weights applied to daily returns -> simulated directly. Deviations
(documented): EODHD total-return-adjusted ETF prices (original used yfinance); a fixed
basket of liquid ETFs (no index membership, so no survivorship concern).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rigor.engine import result_from_returns
from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        e = config.extra
        self.instruments = e.get("instruments", [
            "SPY", "QQQ", "IWM", "IWD", "IWF", "EFA", "EWJ", "EEM", "TLT", "IEF", "SHY",
            "TIP", "LQD", "HYG", "BWX", "GLD", "SLV", "DBC", "VNQ", "RWX", "EMB", "DBA"])
        self.lookback = int(e.get("lookback_period", 252))
        self.vol_lb = int(e.get("vol_lookback", 20))
        self.target_vol = float(e.get("target_vol", 0.10))
        self.max_lev = float(e.get("max_leverage", 2.0))
        self.max_single = float(e.get("max_single_weight", 0.15))
        self.vol_floor = float(e.get("vol_floor", 0.03))
        self.winsorize = float(e.get("winsorize_daily_return", 0.10))
        self.cost_eras = e.get("cost_eras", [[0, 2010, 5], [2010, 2020, 3], [2020, 3000, 2]])
        self.first_trade = e.get("first_trade_date", "2008-07-01")

    def _cost(self, year):
        for lo, hi, bps in self.cost_eras:
            if lo <= year < hi:
                return bps / 1e4
        return self.cost_eras[-1][2] / 1e4

    def _weights(self, sig: pd.Series, vol: pd.Series) -> pd.Series:
        """Inverse-vol weights scaled to target portfolio vol; long-only, capped."""
        safe_vol = vol.clip(lower=self.vol_floor)
        raw = (sig / safe_vol).dropna()
        v = safe_vol.reindex(raw.index).dropna()
        common = raw.index.intersection(v.index)
        if len(common) == 0:
            return pd.Series(dtype="float64")
        w, vv = raw[common], v[common]
        port_vol = np.sqrt((w ** 2 * vv ** 2).sum())
        if port_vol < 1e-8:
            return pd.Series(dtype="float64")
        scaled = (w * (self.target_vol / port_vol)).clip(0.0, self.max_single)
        gross = scaled.abs().sum()
        if gross > self.max_lev:
            scaled = scaled * (self.max_lev / gross)
        return scaled

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        inst = self.instruments
        cl = {}
        for s in inst:
            df = self.data.prices(s, start=start, end=end)
            if df.empty:
                continue
            cl[s] = df.set_index(pd.DatetimeIndex(df["date"]))["adj_close"]
        cols = [s for s in inst if s in cl]
        C = pd.DataFrame(cl).reindex(columns=cols).sort_index()
        return {"close": C, "cols": cols}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        if params:
            import copy as _copy
            _s = _copy.copy(self)
            for _k, _v in params.items():
                if hasattr(_s, _k):
                    setattr(_s, _k, type(getattr(self, _k))(_v))
            return _s.run_backtest(_s.build_cache(), {})
        C = cache["close"]
        cal = C.index
        signals = (C.pct_change(self.lookback) > 0).astype(float)
        signals = signals.where(C.pct_change(self.lookback).notna())
        vol = C.pct_change().rolling(self.vol_lb, min_periods=self.vol_lb).std() * np.sqrt(252)
        daily_rets = C.pct_change().clip(-self.winsorize, self.winsorize)

        first = pd.Timestamp(self.first_trade)
        days = cal[cal >= first]
        rebal = pd.Series(cal, index=cal).resample("ME").last().dropna()
        rebal = pd.DatetimeIndex(rebal.to_numpy())
        rebal = rebal[(rebal >= first) & (rebal <= cal.max())]

        port = pd.Series(0.0, index=days)
        weights_rows: list[tuple[pd.Timestamp, pd.Series]] = []
        prev_w = pd.Series(0.0, index=C.columns)
        for i, rd in enumerate(rebal):
            sd = signals.index[signals.index <= rd]
            if len(sd) == 0:
                continue
            sd = sd[-1]
            w = self._weights(signals.loc[sd], vol.loc[sd]).reindex(C.columns, fill_value=0.0)
            if i < len(rebal) - 1:
                hold = days[(days > rd) & (days <= rebal[i + 1])]
            else:
                hold = days[days > rd]
            if len(hold) == 0:
                continue
            seg = (daily_rets.loc[hold, C.columns] * w).sum(axis=1)
            port.loc[hold] = seg
            cost = 0.5 * (w - prev_w).abs().sum() * self._cost(rd.year)
            if cost > 0:
                port.loc[hold[0]] -= cost
            weights_rows.append((rd, w))
            prev_w = w

        returns = port.iloc[1:].rename("returns")
        if weights_rows:
            rebal_dates = pd.DatetimeIndex([r[0] for r in weights_rows])
            W = pd.DataFrame([r[1] for r in weights_rows], index=rebal_dates, columns=C.columns)
            W_daily = W.reindex(returns.index, method="ffill").fillna(0.0)
            asset_ret = daily_rets.reindex(returns.index)
            return result_from_returns(returns, weights=W_daily, asset_returns=asset_ret)
        return result_from_returns(returns)

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled (shared engine): ranges built from each config's own values;
        # a swept combo rebuilds the cache from an overridden copy of self.
        def _i(v):
            v = int(v)
            c = (v, max(1, round(v * 0.5)), round(v * 1.5))
            return list(dict.fromkeys(x for x in c if x > 0))
        def _f(v):
            v = float(v)
            c = (v, round(v * 0.5, 6), round(v * 1.5, 6))
            return list(dict.fromkeys(x for x in c if x > 0))
        _g = {
            'lookback': _i(self.lookback),
            'vol_lb': _i(self.vol_lb),
            'winsorize': _f(self.winsorize),
        }
        return {k: v for k, v in _g.items() if len(v) >= 2}


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
