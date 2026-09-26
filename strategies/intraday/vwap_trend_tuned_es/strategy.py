"""VWAP Trend, IS-tuned (OOS window) (ES) — `vwap_trend_tuned_es`.

Mechanism `vwap_trend` with the parameterisation selected by an in-sample grid search
(720 cells swept on 2010-06-07..2020-11-02); this folder reports the untouched
out-of-sample window 2020-11-03..2026-06-01. All logic lives in the shared
`rigor.strategies.vwap_papers_engine`. See `README.md`.
"""
from __future__ import annotations

from rigor.strategies.vwap_papers_engine import Strategy, build as _build
from rigor.strategy import StrategyConfig

__all__ = ["Strategy", "build"]


def build(config: StrategyConfig, data) -> Strategy:
    return _build(config, data)
