"""Portfolio allocation: turn selected strategies into capital weights.

``AllocationConfig`` declares the mode + sizing options; ``PortfolioAllocator``
builds the aligned returns matrix and produces the weight vector, applying
per-name overrides, an optional annualised vol target, and a gross-exposure cap.

Modes: the six standard weight methods (``weight_methods.METHODS``) plus
``fixed_percent`` (user weights), ``kelly`` (mean/variance Kelly, long-only), and
``regime_dependent`` (blends per-regime inverse-vol weights by the recent regime
mix; plain inverse-vol below ~1y of history. SMA/vol-bucket regime detector today).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import StrategyCandidate
from .weight_methods import METHODS, allocate_weights, inverse_vol

__all__ = ["AllocationConfig", "PortfolioAllocator", "align_portfolio_returns"]


@dataclass
class AllocationConfig:
    mode: str = "inverse_vol"
    max_weight: float | None = None        # cap on any single strategy weight
    vol_target: float | None = None        # annualised vol target; None disables
    kelly_fraction: float = 1.0            # fractional Kelly (0.25 = quarter-Kelly)
    leverage_cap: float = 2.0              # Σ|w_i|·lev_i cap; 0 disables
    overrides: dict[str, float] = field(default_factory=dict)   # name -> max weight
    fixed_weights: dict[str, float] = field(default_factory=dict)
    benchmark: str = "SPY"
    start_date: str | None = "2000-01-01"
    end_date: str | None = None
    align_frequency: str = "native"        # native | daily | monthly | none
    dd_throttle_trigger: float = 0.0       # de-risk when portfolio DD < -X; 0 disables
    dd_throttle_scale: float = 0.5         # multiplier applied while throttled
    margin_rate_annual: float = 0.0        # interest on gross exposure above 1.0


def align_portfolio_returns(
    cols: dict[str, pd.Series], start_date: str | None, end_date: str | None,
    align_frequency: str = "native",
) -> pd.DataFrame:
    """Trim each series to the window, optionally resample to a shared frequency,
    then outer-join on the union of dates (NaN = strategy inactive that bar)."""
    trimmed: dict[str, pd.Series] = {}
    for name, s in cols.items():
        s = s.dropna()
        if start_date is not None:
            s = s.loc[s.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            s = s.loc[s.index <= pd.Timestamp(end_date)]
        if not s.empty:
            trimmed[name] = s
    if not trimmed:
        return pd.DataFrame()
    if align_frequency == "none":
        return pd.concat(trimmed, axis=1).dropna(how="any")
    if align_frequency == "monthly":
        trimmed = {n: s.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
                   for n, s in trimmed.items()}
    return pd.concat(trimmed, axis=1).sort_index()


class PortfolioAllocator:
    def __init__(self, selected: list[StrategyCandidate], config: AllocationConfig):
        if not selected:
            raise ValueError("no strategies selected")
        self.selected = selected
        self.config = config
        self.names = [c.name for c in selected]
        self.returns_matrix = align_portfolio_returns(
            {c.name: c.returns for c in selected},
            config.start_date, config.end_date, config.align_frequency)

    def allocate(self) -> np.ndarray:
        mode = self.config.mode
        if mode == "fixed_percent":
            w = self._weights_fixed()
        elif mode == "kelly":
            w = self._weights_kelly()
        elif mode == "regime_dependent":
            w = self._weights_regime_dependent()
        elif mode in METHODS:
            w = allocate_weights(self.returns_matrix, mode, max_weight=self.config.max_weight)
        else:
            raise ValueError(f"unknown allocation mode: {mode}")
        w = self._apply_overrides(w)
        w = self._apply_vol_target(w)
        return self._apply_leverage_cap(w)

    def _weights_fixed(self) -> np.ndarray:
        w = np.array([float(self.config.fixed_weights.get(n, 0.0)) for n in self.names])
        s = w.sum()
        return w / s if s > 0 else np.ones(len(self.names)) / len(self.names)

    def _weights_regime_dependent(self) -> np.ndarray:
        """Blend per-regime inverse-vol weights by the recent regime mix. Regimes
        are detected from the equal-weight portfolio's own path (offline-safe).
        Falls back to plain inverse-vol when there is < ~1y of history or no regime
        has enough observations to weight."""
        from .analysis import market_regimes
        rm = self.returns_matrix.dropna(how="any")
        fallback = allocate_weights(self.returns_matrix, "inverse_vol",
                                    max_weight=self.config.max_weight)
        if len(rm) < 220:
            return fallback
        regimes = market_regimes(rm.mean(axis=1)).reindex(rm.index).ffill()
        per_regime = {}
        for reg in regimes.dropna().unique():
            sub = rm.loc[regimes == reg]
            if len(sub) >= 30:
                per_regime[reg] = inverse_vol(sub.cov().to_numpy())
        if not per_regime:
            return fallback
        shares = regimes.iloc[-63:].value_counts(normalize=True).to_dict()
        blended, total = np.zeros(rm.shape[1]), 0.0
        for reg, w in per_regime.items():
            share = float(shares.get(reg, 0.0))
            blended += share * w
            total += share
        if total <= 0 or blended.sum() <= 0:
            return fallback
        from .weight_methods import _cap_and_normalise
        return _cap_and_normalise(blended / total, self.config.max_weight)

    def _weights_kelly(self) -> np.ndarray:
        rm = self.returns_matrix.dropna(how="any")
        if len(rm) < 2:
            return np.ones(len(self.names)) / len(self.names)
        cov = rm.cov().to_numpy()
        mean = rm.mean().to_numpy()
        try:
            raw = np.linalg.solve(cov + 1e-8 * np.eye(len(mean)), mean)
        except np.linalg.LinAlgError:
            raw = mean
        raw = np.clip(raw * float(self.config.kelly_fraction or 1.0), 0.0, None)
        return raw / raw.sum() if raw.sum() > 0 else np.ones(len(self.names)) / len(self.names)

    def _apply_overrides(self, w: np.ndarray) -> np.ndarray:
        if not self.config.overrides:
            return w
        w = w.copy()
        for name, cap in self.config.overrides.items():
            if name in self.names:
                w[self.names.index(name)] = min(w[self.names.index(name)], float(cap))
        s = w.sum()
        return w / s if s > 0 else np.ones(len(w)) / len(w)

    def _apply_vol_target(self, w: np.ndarray) -> np.ndarray:
        target = self.config.vol_target
        if not target or target <= 0:
            return w
        rm = self.returns_matrix.dropna(how="any")
        if len(rm) < 2:
            return w
        from rigor import metrics as _m
        ppy = _m.periods_per_year_of(rm.index)
        port_var = float(w @ rm.cov().to_numpy() @ w)
        port_vol = np.sqrt(port_var * ppy) if port_var > 0 else 0.0
        return w * (float(target) / port_vol) if port_vol > 0 else w

    def _apply_leverage_cap(self, w: np.ndarray) -> np.ndarray:
        cap = float(self.config.leverage_cap or 0.0)
        if cap <= 0:
            return w
        lev = np.array([c.internal_leverage for c in self.selected])
        gross = float(np.sum(np.abs(w) * lev))
        return w * (cap / gross) if gross > cap + 1e-9 else w
