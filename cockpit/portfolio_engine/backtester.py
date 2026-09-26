"""Portfolio backtest: combine weighted strategy returns into one return stream.

Strategies trade on their own date grids, so each bar is renormalised over the
*active* strategies while preserving the target gross exposure — a dormant leg's
weight is carried by the survivors (so a leveraged allocation stays leveraged),
and the average active share is reported so fictitious diversification is visible.
Metrics come from ``rigor.metrics`` (Sharpe/CAGR/MaxDD/Sortino/Calmar/vol) plus the
portfolio-specific extras in ``risk`` (VaR/CVaR/ulcer/diversification). Beta vs a
benchmark is computed when a benchmark return series is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rigor import metrics as _m

from . import risk as _risk
from .allocator import align_portfolio_returns
from .candidates import StrategyCandidate

__all__ = ["BacktestResult", "PortfolioBacktester"]


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    metrics: dict
    beta: dict = field(default_factory=dict)
    diversification: dict = field(default_factory=dict)
    avg_active_share: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)


class PortfolioBacktester:
    def __init__(self, selected: list[StrategyCandidate], weights: np.ndarray, *,
                 start_date: str | None = "2000-01-01", end_date: str | None = None,
                 align_frequency: str = "native", benchmark_returns: pd.Series | None = None,
                 dd_throttle_trigger: float = 0.0, dd_throttle_scale: float = 0.5,
                 margin_rate_annual: float = 0.0):
        self.selected = selected
        self.weights = np.asarray(weights, dtype=float)
        self.benchmark_returns = benchmark_returns
        self.dd_throttle_trigger = float(dd_throttle_trigger or 0.0)
        self.dd_throttle_scale = float(dd_throttle_scale or 0.5)
        self.margin_rate_annual = float(margin_rate_annual or 0.0)
        self.returns_matrix = align_portfolio_returns(
            {c.name: c.returns for c in selected}, start_date, end_date, align_frequency)

    def _apply_dd_throttle(self, raw: np.ndarray) -> np.ndarray:
        """De-risk (scale returns) once the running drawdown breaches the trigger,
        until the equity makes a new high. Cuts risk only, never adds."""
        if self.dd_throttle_trigger <= 0:
            return raw
        trigger = -abs(self.dd_throttle_trigger)
        out = np.empty_like(raw)
        equity = peak = 1.0
        throttled = False
        for i, r0 in enumerate(raw):
            r = r0 * (self.dd_throttle_scale if throttled else 1.0)
            out[i] = r
            equity *= (1.0 + r)
            peak = max(peak, equity)
            if equity / peak - 1.0 <= trigger:
                throttled = True
            elif equity >= peak:
                throttled = False
        return out

    def run(self) -> BacktestResult:
        rm = self.returns_matrix
        if rm.empty:
            raise ValueError("no overlapping returns to backtest")
        names = list(rm.columns)
        w = self.weights
        ret = rm.to_numpy(dtype=float)
        mask = ~np.isnan(ret)
        filled = np.where(mask, ret, 0.0)
        abs_w = np.abs(w)
        target_gross = float(abs_w.sum())
        active_w = mask.astype(float) @ abs_w
        active_share = active_w / target_gross if target_gross > 1e-12 else np.zeros_like(active_w)
        share_safe = np.where(active_share > 1e-12, active_share, np.nan)
        raw = np.nan_to_num((filled @ w) / share_safe, nan=0.0)
        if self.margin_rate_annual > 0:  # interest on gross exposure above 1.0
            lev = np.array([c.internal_leverage for c in self.selected])
            excess = max(float(np.sum(np.abs(w) * lev)) - 1.0, 0.0)
            raw = raw - excess * self.margin_rate_annual / 252.0
        raw = self._apply_dd_throttle(raw)
        port_ret = pd.Series(raw, index=rm.index, name="portfolio")
        avg_active = float(np.nanmean(np.where(active_share > 1e-12, active_share, np.nan))) \
            if np.isfinite(share_safe).any() else 1.0

        ppy = _m.periods_per_year_of(port_ret.index)
        arr = port_ret.to_numpy(dtype="float64")
        metrics = {
            "sharpe": float(_m.compute_sharpe(arr, ppy)),
            "cagr": float(_m.compute_cagr(arr, ppy)),
            "max_drawdown": float(_m.compute_max_drawdown(arr)),
            "sortino": float(_m.compute_sortino(arr, ppy)),
            "calmar": float(_m.compute_calmar(arr, ppy)),
            "volatility": float(_m.compute_volatility(arr, ppy)),
            "var_95": _risk.value_at_risk(port_ret, 0.05),
            "cvar_95": _risk.conditional_var(port_ret, 0.05),
            "ulcer_index": _risk.ulcer_index(port_ret),
            "n_obs": len(port_ret),
        }
        div = _risk.diversification_ratio(rm, w)
        beta = (_risk.beta_to_benchmark(port_ret, self.benchmark_returns)
                if self.benchmark_returns is not None else {})
        return BacktestResult(
            portfolio_returns=port_ret, metrics=metrics, beta=beta, diversification=div,
            avg_active_share=avg_active,
            weights={names[i]: float(w[i]) for i in range(len(names))})
