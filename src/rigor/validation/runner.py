"""Validation orchestrator: run the battery on a BacktestResult / strategy."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .. import metrics as _m
from ..engine import BacktestResult
from .cpcv import cpcv_pbo
from .overfit import (
    deflated_sharpe_ratio,
    haircut_sharpe,
    min_track_record_length,
    newey_west_sharpe,
    pezier_white_adjusted_sharpe,
    probabilistic_sharpe_ratio,
)
from .robustness import bootstrap_sharpe_ci, per_year, walk_forward
from .verdict import compute_verdict


def _harvey_t(sharpe_ann: float, n_obs: int, ppy: int,
              skew: float = 0.0, kurt: float = 0.0) -> float:
    """Harvey-Liu-Zhu (2016) significance t-stat on the *per-period* Sharpe with the
    Lo (2002) non-normal SE. A left tail (skew < 0) or fat tail (kurt > 0, excess)
    inflates the SE and lowers t, so the gate isn't fooled by skewed payoffs; the
    prior form used the annualised Sharpe in the SE and ignored skew/kurtosis."""
    if n_obs <= 1 or ppy <= 0:
        return 0.0
    sr_pp = sharpe_ann / np.sqrt(ppy)
    se_sq = 1.0 + 0.5 * sr_pp**2 - skew * sr_pp + (kurt / 4.0) * sr_pp**2
    if se_sq <= 0:
        return 0.0
    return float(sr_pp * np.sqrt(n_obs - 1.0) / np.sqrt(se_sq))


def validate(result: BacktestResult, *, n_trials: int = 1, cpcv: dict | None = None) -> dict:
    """Full validation battery on a single result. ``n_trials`` deflates DSR/haircut."""
    r = result.returns.dropna()
    m = result.metrics
    arr = r.to_numpy(dtype="float64")
    n = len(arr)
    ppy = _m.periods_per_year_of(r.index)
    sharpe_ann = float(m.get("sharpe", 0.0))
    sharpe_pp = sharpe_ann / np.sqrt(ppy)
    sk = float(skew(arr, bias=False)) if n > 2 else 0.0
    ku = float(kurtosis(arr, fisher=True, bias=False)) if n > 3 else 0.0  # excess

    hc = haircut_sharpe(sharpe_pp, n_trials, n, sk, ku)
    overfit = {
        "psr": probabilistic_sharpe_ratio(sharpe_pp, 0.0, n, sk, ku),
        "dsr": deflated_sharpe_ratio(sharpe_pp, n_trials, n, sk, ku),
        "haircut_sharpe": hc["haircut_sharpe"],
        "haircut_pct": hc["haircut_pct"],
        "min_track_record_length": min_track_record_length(sharpe_pp, 0.0, sk, ku),
        "harvey_t": _harvey_t(sharpe_ann, n, ppy, sk, ku),
        # Pezier-White skew/kurtosis-adjusted (annualised) Sharpe: penalises
        # negative skew / fat tails relative to the plain Sharpe. Additive — the
        # verdict ignores unknown keys, so this surfaces in summary.json without
        # changing any gate or committed verdict.
        "pezier_white_adjusted_sharpe": pezier_white_adjusted_sharpe(arr, ppy),
        "newey_west": newey_west_sharpe(arr, ppy),
        "n_trials": int(n_trials),
        "skew": round(sk, 4),
        "excess_kurtosis": round(ku, 4),
    }
    robustness = {
        "walk_forward": walk_forward(r),
        "per_year": per_year(r),
        "bootstrap": bootstrap_sharpe_ci(r),
    }
    v = compute_verdict(m, overfit, robustness, cpcv)
    return {"verdict": v.as_dict(), "overfit": overfit, "robustness": robustness, "cpcv": cpcv}


def _expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return []
    keys = list(grid)
    return [
        dict(zip(keys, vals, strict=True))
        for vals in itertools.product(*(grid[k] for k in keys))
    ]


# Rebalance frequency -> nominal holding/label horizon in *bars*. CPCV runs on
# the daily-aligned returns matrix, so the horizon is expressed in daily bars: a
# label realised on a rebalance carries information until the next rebalance, so
# the holding horizon is ~1 rebalance period. These are conservative round
# numbers (a trading week ≈ 5 bars, a trading month ≈ 21 bars).
_REBALANCE_PURGE_BARS = {"daily": 1, "weekly": 5, "monthly": 21}
_DEFAULT_PURGE_BARS = 5  # fallback when rebalance_freq is unknown/odd (~1 trading week)


def _purge_gap_for(strategy) -> int:
    """Pick a CPCV purge half-width (bars) from the strategy's label horizon.

    The cleanest discoverable proxy for a strategy's holding/label horizon is its
    ``config.rebalance_freq`` (daily/weekly/monthly) — a label stays informative
    for about one rebalance period. Unknown/odd values fall back to a small
    constant (~1 trading week), matching AFML's "purge the label horizon" rule.
    """
    cfg = getattr(strategy, "config", None)
    freq = str(getattr(cfg, "rebalance_freq", "")).lower()
    return _REBALANCE_PURGE_BARS.get(freq, _DEFAULT_PURGE_BARS)


def validate_strategy(
    strategy, result: BacktestResult | None = None, *, n_trials: int | None = None
) -> dict:
    """Validate a strategy; if it has a multi-value param_grid, also run CPCV/PBO.

    Running the grid re-runs the strategy once per combo (cheap for small grids;
    skipped for single-config strategies, where CPCV is not applicable).
    """
    if result is None:
        result = strategy.backtest()
    combos = _expand_grid(strategy.param_grid())
    cpcv = None
    nt = n_trials
    if len(combos) >= 2:
        cache = strategy.cache
        series = []
        for c in combos:
            ret = strategy.run_backtest(cache, c)
            if isinstance(ret, BacktestResult):
                ret = ret.returns
            series.append(ret if isinstance(ret, pd.Series) else pd.Series(np.asarray(ret)))
        mat = pd.concat(series, axis=1).dropna()
        if len(mat) >= 20:
            ppy = _m.periods_per_year_of(mat.index)
            # Purge + embargo (AFML Ch. 7) to keep PBO from being optimistically
            # biased by serial correlation across the IS/OOS boundary:
            #   * purge_gap = the strategy's label/holding horizon in bars,
            #     derived from rebalance_freq (else ~1 trading week);
            #   * embargo_pct = 0.01 — drop ~1% of samples on the post-test edge,
            #     AFML's common default for the residual-autocorrelation embargo.
            cpcv = cpcv_pbo(
                mat.to_numpy().T, ppy=ppy,
                purge_gap=_purge_gap_for(strategy), embargo_pct=0.01,
            )
        nt = nt or len(combos)
    return validate(result, n_trials=nt or 1, cpcv=cpcv)
