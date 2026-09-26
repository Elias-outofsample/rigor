"""Temporal robustness: walk-forward fold consistency, per-year, bootstrap CI.

These operate on a single returns series (deterministic). The walk-forward here
is *fixed-strategy* out-of-sample fold consistency (does the edge hold across
contiguous sub-periods), NOT re-optimisation walk-forward — that needs the param
grid and belongs to the optimisation layer. Block bootstrap uses a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import metrics as _m


def walk_forward(returns: pd.Series, n_folds: int = 8) -> dict:
    """Split into ``n_folds`` contiguous folds; measure sub-period consistency."""
    r = returns.dropna()
    ppy = _m.periods_per_year_of(r.index)
    full = _m.compute_sharpe(r.to_numpy(), ppy)
    folds = np.array_split(r.to_numpy(), n_folds)
    sharpes = [_m.compute_sharpe(f, ppy) for f in folds if len(f) > 1]
    if not sharpes:
        return {"n_folds": 0, "fold_sharpes": [], "efficiency": 0.0, "consistency": 0.0,
                "cv": float("inf"), "verdict": "FRAGILE"}
    sh = np.array(sharpes)
    mean = float(sh.mean())
    efficiency = float(mean / full) if full > 0 else 0.0
    consistency = float((sh > 0).mean())
    cv = float(sh.std(ddof=1) / abs(mean)) if mean != 0 else float("inf")
    if cv < 0.5 and consistency >= 0.7:
        verdict = "ROBUST"
    elif cv < 1.0 and consistency >= 0.5:
        verdict = "MARGINAL"
    else:
        verdict = "FRAGILE"
    return {
        "n_folds": len(sharpes),
        "fold_sharpes": [round(x, 3) for x in sharpes],
        "full_sharpe": round(full, 3),
        "efficiency": round(efficiency, 3),
        "consistency": round(consistency, 3),
        "cv": round(cv, 3),
        "verdict": verdict,
    }


def per_year(returns: pd.Series) -> dict:
    """Per-calendar-year Sharpe + return; positive fraction and worst year."""
    r = returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        return {"years": {}, "positive_fraction": 0.0, "worst_sharpe": 0.0, "n_years": 0}
    out = {}
    for year, grp in r.groupby(r.index.year):
        out[int(year)] = {
            "sharpe": round(_m.compute_sharpe(grp.to_numpy()), 3),
            "return": round(float((1 + grp).prod() - 1), 4),
        }
    sharpes = [v["sharpe"] for v in out.values()]
    rets = [v["return"] for v in out.values()]
    return {
        "years": out,
        "n_years": len(out),
        "positive_fraction": round(float(np.mean([x > 0 for x in rets])), 3) if rets else 0.0,
        "worst_sharpe": round(min(sharpes), 3) if sharpes else 0.0,
    }


def _optimal_block_size(r: np.ndarray) -> int:
    """Politis & White (2004) automatic block length."""
    n = len(r)
    if n < 20:
        return max(1, n // 2)
    x = r - r.mean()
    var0 = float(x @ x / n)
    if var0 == 0:
        return 1
    kn = max(5, int(np.ceil(np.log10(n))))
    max_lag = min(n // 3, max(50, 2 * kn))
    R = np.array([var0] + [float(x[k:] @ x[:-k] / n) for k in range(1, max_lag + 1)])
    rho = R / R[0]
    thr = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = max_lag // 2
    for m in range(1, max_lag - kn + 1):
        if np.all(np.abs(rho[m + 1: m + 1 + kn]) < thr):
            m_hat = m
            break
    M = min(2 * m_hat, max_lag)
    if M < 1:
        return 1
    lags = np.arange(-M, M + 1)
    lam = 0.5 * (1.0 + np.cos(np.pi * lags / M))
    R_full = np.concatenate([R[1:M + 1][::-1], [R[0]], R[1:M + 1]])
    G = float(np.sum(lam * np.abs(lags) * R_full))
    g0 = float(np.sum(lam * R_full))
    if g0 == 0:
        return 1
    block = int(np.ceil((2.0 * G**2 / g0**2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)))
    return max(1, min(block, n // 2))


def bootstrap_sharpe_ci(returns: pd.Series, n_bootstrap: int = 2000, ci: float = 0.95) -> dict:
    """Stationary block-bootstrap Sharpe CI (BCa). Deterministic (seed=42)."""
    arr = returns.dropna().to_numpy(dtype="float64")
    n = len(arr)
    ppy = _m.periods_per_year_of(returns.index)
    if n < 30:
        return {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_positive": 0.0}
    block = _optimal_block_size(arr)
    point = float(arr.mean() / arr.std(ddof=1) * np.sqrt(ppy))
    rng = np.random.default_rng(42)
    scale = np.sqrt(ppy)
    boot = np.empty(n_bootstrap)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n] % n
        s = arr[idx]
        sd = s.std(ddof=1)
        boot[b] = s.mean() / sd * scale if sd > 0 else 0.0
    alpha = (1 - ci) / 2
    # BCa adjustment
    prop = np.clip(np.mean(boot < point), 1 / (n_bootstrap + 1), 1 - 1 / (n_bootstrap + 1))
    z0 = norm.ppf(prop)
    sum_r, sumsq = arr.sum(), (arr * arr).sum()
    m_loo = (sum_r - arr) / (n - 1)
    var_loo = np.maximum((sumsq - arr * arr) / (n - 1) - m_loo**2, 0.0)
    std_loo = np.sqrt(var_loo * (n - 1) / (n - 2))
    jack = np.where(std_loo > 0, m_loo / std_loo * scale, 0.0)
    diffs = jack.mean() - jack
    denom = 6.0 * (np.sum(diffs**2)) ** 1.5
    a = float(np.sum(diffs**3) / denom) if denom > 0 else 0.0
    za, z1a = norm.ppf(alpha), norm.ppf(1 - alpha)
    a1 = norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    a2 = norm.cdf(z0 + (z0 + z1a) / (1 - a * (z0 + z1a)))
    return {
        "sharpe": round(point, 3),
        "ci_lower": round(float(np.percentile(boot, a1 * 100)), 3),
        "ci_upper": round(float(np.percentile(boot, a2 * 100)), 3),
        "p_positive": round(float(np.mean(boot > 0)), 3),
        "block_size": int(block),
    }
