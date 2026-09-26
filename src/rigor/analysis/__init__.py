"""Advanced analysis layer — factor attribution, risk metrics, signal quality,
overfit diagnostics, regimes, and the unified strategy verdict.

This package is **purely additive** and **opt-in by design**: nothing in the core
data/engine/metrics/validation path imports the analyst modules, so a strategy
that doesn't call them is unaffected. They exist to be reached for deliberately
during a deep-dive, not run on every backtest. Everything here builds on
``rigor.metrics`` / ``rigor.validation`` (one stack, no duplication).

The full catalogue — one line per module (what it computes, that it's opt-in, and
how to call it) — lives in ``docs/analysis-toolkit.md``. Read that first to
discover which tool answers a given question; each module is unit-tested even
though it has no production call-site.

Some capabilities need heavier libraries (statsmodels, arch, hmmlearn,
pandas-datareader). Those are **optional**: install them with
``pip install -e ".[analysis]"``. Without them, the affected functions
degrade gracefully (a transparent fallback) or raise a clear install message —
the core install and every strategy keep working regardless.
"""
from __future__ import annotations

from . import (
    alpha_robustness,
    attribution,
    capture,
    cost,
    diagnostics,
    distribution,
    edge_classifier,
    entry_signal_ic,
    exposure,
    factor,
    kelly,
    labeling,
    lookahead,
    predictive_ability,
    quality,
    regime,
    risk,
    scenario_response,
    signal_quality,
    structural,
    survival,
    tail_risk,
    torture,
    verdict,
)
from .distribution import analyze_distribution
from .entry_signal_ic import compute_entry_signal_ic
from .kelly import (
    kelly_continuous,
    kelly_fractional,
    kelly_full,
    kelly_half,
    kelly_multi_asset,
    kelly_shrinkage,
)
from .quality import strategy_badge
from .survival import compute_survival
from .torture import TortureResult, Weakness, compute_torture_score

__all__ = [
    "risk", "labeling", "lookahead", "signal_quality", "structural",
    "predictive_ability", "attribution", "exposure", "edge_classifier",
    "verdict", "tail_risk", "regime", "cost", "factor", "alpha_robustness",
    "diagnostics", "torture", "kelly", "capture",
    "distribution", "entry_signal_ic", "quality", "scenario_response", "survival",
    "compute_torture_score", "TortureResult", "Weakness",
    "kelly_full", "kelly_half", "kelly_fractional", "kelly_continuous",
    "kelly_multi_asset", "kelly_shrinkage",
    "analyze_distribution", "compute_entry_signal_ic", "strategy_badge",
    "compute_survival",
]
