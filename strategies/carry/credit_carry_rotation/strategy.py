"""Credit Carry Rotation (S34) — HYG / IEF / EMB / SHY monthly rotation.

Port of an earlier research version (Carry_S34, `Credit` sub-strategy): a monthly rotation across credit /
duration ETFs driven by a credit-carry price-ratio signal plus a credit-stress macro overlay.
(The S34 *main* is a 60% futures + 40% credit blend; the futures leg has no EODHD data, so this
ports the portable credit-carry leg.)

Each month-end, on total-return-adjusted prices:
  - ratio = HYG / IEF, vs its `ratio_ma_period`-month moving average.
  - Macro overlay: if the FRED high-yield OAS (`BAMLH0A0HYM2`) > `oas_stress_high`, or the
    yield curve (`T10Y3M`) inverts past `curve_inversion_threshold` (disabled by default),
    de-risk to mostly SHY.
  - Otherwise allocate by the ratio signal: carry-on (ratio > MA + threshold) → overweight HYG
    + EMB; carry-off (ratio < MA - threshold) → IEF/SHY; neutral → balanced.

Monthly target weights applied to daily ETF returns → simulated directly; turnover charged
`cost_bps`. Deviations (documented): EODHD total-return-adjusted prices (original used yfinance
auto-adjust); FRED via the framework's seeded cache.
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
        self.carry = e.get("carry_etf", "HYG")
        self.dur = e.get("duration_etf", "IEF")
        self.safe = e.get("safe_etf", "SHY")
        self.em = e.get("em_etf", "EMB")
        self.cash = e.get("cash_etf", "BIL")
        self.etfs = [self.carry, self.dur, self.safe, self.em, self.cash]
        self.ratio_ma_period = int(e.get("ratio_ma_period", 9))
        self.ratio_threshold = float(e.get("ratio_threshold", 0.01))
        self.oas_fred = e.get("oas_fred", "BAMLH0A0HYM2")
        self.oas_stress_high = float(e.get("oas_stress_high", 8.0))
        self.curve_fred = e.get("curve_fred", "T10Y3M")
        self.curve_inv = float(e.get("curve_inversion_threshold", -99.0))
        self.hyg_max = float(e.get("hyg_weight_max", 0.70))
        self.hyg_min = float(e.get("hyg_weight_min", 0.20))
        self.emb_weight = float(e.get("emb_weight", 0.15))
        self.cost = float(e.get("cost_bps", 5)) / 1e4
        self.first_trade = e.get("first_trade_date", "2008-07-01")

    def _alloc(self, r, ma, oas, curve) -> dict[str, float]:
        if pd.isna(r) or pd.isna(ma):
            return {self.safe: 1.0}
        stress = pd.notna(oas) and oas > self.oas_stress_high
        inverted = pd.notna(curve) and curve < self.curve_inv
        if stress or inverted:
            return {self.safe: 1.0 - self.hyg_min, self.carry: self.hyg_min}
        sig = (r / ma) - 1.0
        if sig > self.ratio_threshold:
            hyg_w = self.hyg_max - self.emb_weight
            return {self.carry: hyg_w, self.em: self.emb_weight,
                    self.dur: 1.0 - hyg_w - self.emb_weight}
        if sig < -self.ratio_threshold:
            return {self.dur: 0.60, self.safe: 0.20, self.carry: self.hyg_min}
        return {self.carry: 0.45, self.dur: 0.25, self.safe: 0.15, self.em: 0.15}

    def build_cache(self) -> dict[str, Any]:
        start, end = self.config.start_date, self.config.end_date
        cl = {}
        for s in self.etfs:
            df = self.data.prices(s, start=start, end=end)
            if not df.empty:
                cl[s] = df.set_index(pd.DatetimeIndex(df["date"]))["adj_close"]
        C = pd.DataFrame(cl).sort_index()
        monthly = C.resample("ME").last()
        ratio = monthly[self.carry] / monthly[self.dur]
        ratio_ma = ratio.rolling(self.ratio_ma_period).mean()
        oas = self.data.fred(self.oas_fred)
        curve = self.data.fred(self.curve_fred)
        m_oas = oas.resample("ME").last().reindex(monthly.index, method="ffill")
        m_curve = curve.resample("ME").last().reindex(monthly.index, method="ffill")
        return {"close": C, "monthly_idx": monthly.index, "ratio": ratio,
                "ratio_ma": ratio_ma, "oas": m_oas, "curve": m_curve}

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> pd.Series:
        _p = params or {}
        ratio_ma_period = int(_p.get("ratio_ma_period", self.ratio_ma_period))
        if params:
            import copy as _copy
            _s = _copy.copy(self)
            _s.ratio_ma_period = ratio_ma_period
            cache = _s.build_cache()
        C = cache["close"]
        ratio, ratio_ma, m_oas, m_curve = cache["ratio"], cache["ratio_ma"], cache["oas"], cache["curve"]
        cal = C.index
        daily_rets = C.pct_change(fill_method=None)
        first = pd.Timestamp(self.first_trade)

        rebal = cache["monthly_idx"]
        rebal = rebal[(rebal >= first) & (rebal <= cal.max())]
        days = cal[cal >= rebal[0]]
        port = pd.Series(0.0, index=days)
        all_etfs = list(C.columns)
        weights_rows: list[tuple[pd.Timestamp, dict[str, float]]] = []
        prev_w: dict[str, float] = {}
        for i, rd in enumerate(rebal):
            w = self._alloc(ratio.get(rd, np.nan), ratio_ma.get(rd, np.nan),
                            m_oas.get(rd, np.nan), m_curve.get(rd, np.nan))
            hold = days[(days > rd) & (days <= rebal[i + 1])] if i < len(rebal) - 1 else days[days > rd]
            if len(hold) == 0:
                prev_w = w
                continue
            contrib = pd.Series(0.0, index=hold)
            for etf, wt in w.items():
                if etf in daily_rets:
                    contrib = contrib + wt * daily_rets.loc[hold, etf].fillna(0.0)
            port.loc[hold] = contrib
            turnover = sum(abs(w.get(e, 0.0) - prev_w.get(e, 0.0)) for e in set(w) | set(prev_w))
            if turnover > 0:
                port.loc[hold[0]] -= turnover * self.cost
            weights_rows.append((rd, w))
            prev_w = w
        returns = port.iloc[1:].rename("returns")
        if weights_rows:
            rebal_dates = pd.DatetimeIndex([r[0] for r in weights_rows])
            W = pd.DataFrame(
                [{etf: w.get(etf, 0.0) for etf in all_etfs} for _, w in weights_rows],
                index=rebal_dates,
            )
            W_daily = W.reindex(returns.index, method="ffill").fillna(0.0)
            asset_ret = daily_rets.reindex(returns.index)
            return result_from_returns(returns, weights=W_daily, asset_returns=asset_ret)
        return result_from_returns(returns)

    def param_grid(self) -> dict[str, list]:
        # Auto-backfilled: cache-bound knobs swept via per-combo cache rebuild; canonical first.
        return {
            'ratio_ma_period': [9, 4, 14],
        }


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
