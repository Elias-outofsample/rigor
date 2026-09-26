"""VWAP Regime Router, causal (ES) — `vwap_regime_route_causal_es`.

D. Lee (SSRN 6438039) routing A/C, look-ahead removed.

Thin shim: all logic lives in the shared `rigor.strategies.vwap_papers_engine`
(mechanism `regime_routed`), which is a bit-for-bit port of the reference
implementation. Instrument: ES (CME e-mini S&P 500), Databento 1-minute bars resampled to
5m, RTH only, flat overnight. See `README.md` for the thesis and the
measured result.
"""
from __future__ import annotations

from rigor.strategies.vwap_papers_engine import Strategy, build as _build
from rigor.strategy import StrategyConfig

__all__ = ["Strategy", "build"]


def build(config: StrategyConfig, data) -> Strategy:
    return _build(config, data)
