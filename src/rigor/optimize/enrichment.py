"""Stage-3 enrichment metrics — robustness scalars from an equity curve.

A single pass over a config's equity curve yields the metrics the 5-pillar
selection score consumes, beyond the headline Sharpe/CAGR/MaxDD:

  * **roc310** — temporal decay: last-third vs first-third CAGR (front-loaded
    alpha that has since died scores low).
  * **stability** — Spearman trend of equity vs time (monotone up = 1).
  * **k_ratio** — log-equity regression slope / its standard error (trend
    strength relative to noise).
  * **eta** — drawdown recovery rate (fraction of >2% drawdowns that recover).
  * **psr / upi / rolling_consistency / sortino** — the robust pillars (PSR,
    Ulcer Performance Index, min/max rolling-Sharpe, Sortino).
  * **recency** — last-3y CAGR vs overall CAGR (catches a curve flat for years).

Ported faithfully from the prior research library's ``compute_stage3_enrichment``,
reusing ``rigor.metrics`` and ``rigor.validation`` primitives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr

from .. import metrics as _m
from ..validation.overfit import probabilistic_sharpe_ratio

__all__ = ["enrich_equity", "rolling_sharpe", "ulcer_performance_index"]


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) and np.isfinite(b) else default


def rolling_sharpe(returns: pd.Series, window: int, ppy: int) -> pd.Series:
    """Annualised Sharpe over a sliding ``window`` (NaN until the window fills)."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std) * np.sqrt(ppy)


def ulcer_performance_index(equity: np.ndarray, returns: np.ndarray, ppy: int) -> float:
    """CAGR / Ulcer Index — penalises drawdown depth *and* duration."""
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / np.where(peak == 0, 1.0, peak)
    ulcer = float(np.sqrt(np.mean((dd_pct * 100.0) ** 2)))
    if ulcer <= 0:
        return 0.0
    n = len(returns)
    cum = float(equity[-1] / equity[0]) if equity[0] > 0 else 1.0
    cagr = (cum ** (ppy / n) - 1.0) if cum > 0 and n > 0 else 0.0
    return float(cagr / (ulcer / 100.0))


def enrich_equity(returns: pd.Series, ppy: int | None = None) -> dict:
    """Compute the stage-3 enrichment scalars for one config's return series.

    Returns neutral values for short/degenerate series (mirrors the source:
    near-flat equity yields NaN robust pillars so the neutral-NaN normaliser
    maps them to the mid score).
    """
    r = returns.dropna()
    ppy = ppy or _m.periods_per_year_of(r.index)
    arr = (1.0 + r.to_numpy(dtype="float64")).cumprod()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = len(arr)

    out = {"roc310": 5.0, "stability": 0.0, "k_ratio": 0.0, "eta": 0.0,
           "psr": 0.5, "upi": 0.0, "rolling_consistency": 0.0, "sortino": 0.0,
           "recency": 0.0}
    if n < 30:
        return out

    # ROC310 — last-third vs first-third CAGR
    third = n // 3
    def _ann(e):
        if len(e) < 2 or e[0] <= 0:
            return 0.0
        years = len(e) / ppy
        return float((e[-1] / e[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    cagr_first, cagr_last = _ann(arr[:third]), _ann(arr[-third:])
    if cagr_last <= 0:
        out["roc310"] = 0.0
    else:
        ratio = _safe_divide(cagr_last, max(abs(cagr_first), 1e-6), 1.0)
        out["roc310"] = round(float(np.clip((ratio / 2.0) * 10.0, 0.0, 10.0)), 3)

    # Spearman trend + K-ratio
    t = np.arange(n, dtype="float64")
    try:
        rho, _ = spearmanr(t, arr)
        out["stability"] = round(float(rho), 4)
    except Exception:
        pass
    if n >= 10:
        try:
            log_eq = np.log(arr)
            coeffs = np.polyfit(t, log_eq, 1)
            resid = log_eq - np.polyval(coeffs, t)
            se = resid.std() / np.sqrt(n) if n > 1 else 1e-12
            out["k_ratio"] = round(_safe_divide(coeffs[0], se, 0.0), 4)
        except Exception:
            pass

    # ETA — drawdown recovery rate
    peak = np.maximum.accumulate(arr)
    dd_pct = (arr - peak) / np.where(peak == 0, 1.0, peak)
    starts = np.where((dd_pct < -0.02) & (np.diff(np.append(0.0, dd_pct)) < 0))[0]
    if len(starts) > 0:
        rec = 0
        for s in starts[:20]:
            trough = s + int(np.argmin(dd_pct[s:s + ppy]))
            window = arr[trough:trough + ppy]
            if len(window) > 0 and window.max() >= arr[s]:
                rec += 1
        out["eta"] = round(rec / len(starts[:20]), 4)

    ret = np.diff(arr) / arr[:-1]
    ret = ret[np.isfinite(ret)]
    n_r = len(ret)
    if n_r < 30:
        return out

    # Degenerate-series guard: near-flat equity -> NaN robust pillars
    n_active = int(np.sum(np.abs(ret) > 1e-9))
    total_ret = float(arr[-1] / arr[0] - 1.0) if arr[0] > 0 else 0.0
    years = n_r / ppy
    if n_active < max(60, int(0.05 * n_r)) or abs(total_ret) < 0.005 * years:
        out.update(psr=np.nan, upi=np.nan, rolling_consistency=np.nan, sortino=np.nan)
        return out

    mean_r, std_r = float(np.mean(ret)), float(np.std(ret, ddof=1))
    sharpe = (mean_r / std_r) * np.sqrt(ppy) if std_r > 0 else 0.0
    sk = float(skew(ret, bias=False)) if n_r > 3 else 0.0
    ku = float(kurtosis(ret, fisher=True, bias=False)) if n_r > 3 else 0.0
    out["psr"] = probabilistic_sharpe_ratio(sharpe, 0.0, n_r, sk, ku)
    out["upi"] = ulcer_performance_index(arr, ret, ppy)
    out["sortino"] = float(_m.compute_sortino(ret, ppy))

    window = min(ppy, n_r // 2)
    if window >= 30 and n_r >= window * 2:
        roll = rolling_sharpe(pd.Series(ret), window, ppy).dropna()
        if len(roll) >= 2:
            mn, mx = float(roll.min()), float(roll.max())
            if mx > 0:
                out["rolling_consistency"] = max(mn / mx, 0.0)

    if n_r >= ppy * 3 and arr[0] > 0 and arr[-ppy * 3] > 0:
        overall = (arr[-1] / arr[0]) ** (ppy / n_r) - 1.0
        recent = (arr[-1] / arr[-ppy * 3]) ** (1.0 / 3.0) - 1.0
        out["recency"] = float(np.clip(recent / overall, -1.0, 2.0)) if abs(overall) > 1e-6 else 0.0
    return out
