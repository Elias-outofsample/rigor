"""Structural-break detection & Masters permutation tests (#7, part).

Walk-forward and per-year *average* over time; these *locate* when an edge broke:

  * ``cusum_mean`` — Page CUSUM (Brownian-bridge) for a mean shift.
  * ``cusum_variance`` — Brown-Durbin-Evans CUSUM-of-squares for a variance shift.
  * ``edge_decay`` — slope of rolling Sharpe over time (decaying edge).
  * ``multiple_breaks`` — ICSS (Inclan-Tiao) iterative variance-break segmentation.
  * ``masters_block_permutation`` — block sign-flip permutation Sharpe test.
  * ``masters_optimization_bias`` — bootstrap best-of-N null → inflation factor for
    an *optimised* Sharpe (the empirical analogue of expected-max-Sharpe).

Pure NumPy/SciPy; Sharpe/CAGR reuse ``rigor.metrics``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = [
    "cusum_mean", "cusum_variance", "edge_decay", "multiple_breaks",
    "masters_block_permutation", "masters_optimization_bias",
]


def _arr(returns) -> np.ndarray:
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    r = r.astype("float64")
    return r[np.isfinite(r)]


def _sharpe(r: np.ndarray, ppy: int) -> float:
    return float(_m.compute_sharpe(r, ppy)) if len(r) >= 2 else 0.0


def cusum_mean(returns, confidence: float = 0.95) -> dict:
    """Page (1954) CUSUM for a mean change; sup of the Brownian bridge vs the
    Kolmogorov critical value."""
    r = _arr(returns)
    n = len(r)
    if n < 20:
        return {"has_break": False, "break_index": -1}
    sigma = r.std(ddof=1)
    if sigma == 0:
        return {"has_break": False, "break_index": -1}
    bridge = np.cumsum(r - r.mean()) / (sigma * np.sqrt(n))
    sup = float(np.max(np.abs(bridge)))
    crit = {0.90: 1.22, 0.95: 1.36, 0.99: 1.63}.get(confidence, 1.36)
    return {"has_break": sup > crit, "break_index": int(np.argmax(np.abs(bridge))),
            "sup_statistic": sup, "critical_value": crit}


def cusum_variance(returns, confidence: float = 0.95) -> dict:
    """Brown-Durbin-Evans CUSUM-of-squares for a variance change."""
    r = _arr(returns)
    n = len(r)
    if n < 20 or np.sum(r**2) == 0:
        return {"has_break": False, "break_index": -1}
    dev = np.cumsum(r**2) / np.sum(r**2) - np.arange(1, n + 1) / n
    sup = float(np.max(np.abs(dev)))
    crit = 0.1 + 0.85 / np.sqrt(n)
    return {"has_break": sup > crit, "sup_statistic": sup, "critical_value": float(crit),
            "break_index": int(np.argmax(np.abs(dev)))}


def edge_decay(returns, window: int = 252) -> dict:
    """Slope of the rolling-Sharpe series over time (negative = decaying edge)."""
    r = returns if isinstance(returns, pd.Series) else pd.Series(returns)
    n = len(r)
    if n < window * 2:
        return {"has_decay": False, "slope": 0.0}
    ppy = (_m.periods_per_year_of(r.index)
           if isinstance(r.index, pd.DatetimeIndex) else _m.TRADING_DAYS)
    step = max(1, window // 4)
    pts = [(i + window // 2, _sharpe(r.iloc[i:i + window].to_numpy(), ppy))
           for i in range(0, n - window + 1, step)]
    if len(pts) < 3:
        return {"has_decay": False, "slope": 0.0}
    x = np.array([p[0] for p in pts], dtype="float64")
    y = np.array([p[1] for p in pts], dtype="float64")
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1]
    annual = slope * ppy
    return {"has_decay": bool(annual < -0.1), "slope": float(slope),
            "slope_annual": float(annual), "rolling_sharpes": [p[1] for p in pts]}


def multiple_breaks(returns, max_breaks: int = 5) -> dict:
    """ICSS (Inclan-Tiao 1994) iterative CUSUM-of-squares → multiple variance breaks."""
    r = _arr(returns)
    n = len(r)
    if n < 50 or np.sum(r**2) == 0:
        return {"n_breaks": 0, "break_points": []}
    sq = r**2

    def stat(start, end):
        seg = sq[start:end]
        m = len(seg)
        if m < 10:
            return 0.0, start
        cs = np.cumsum(seg)
        if cs[-1] == 0:
            return 0.0, start
        d = cs / cs[-1] - np.arange(1, m + 1) / m
        i = int(np.argmax(np.abs(d)))
        return float(np.sqrt(m / 2) * abs(d[i])), start + i

    crit, breaks, segments = 1.358, [], [(0, n)]
    for _ in range(max_breaks):
        nxt, found = [], False
        for start, end in segments:
            s, bp = stat(start, end)
            if s > crit and start + 5 < bp < end - 5:
                breaks.append(bp)
                nxt += [(start, bp), (bp, end)]
                found = True
            else:
                nxt.append((start, end))
        segments = nxt
        if not found:
            break
    return {"n_breaks": len(breaks), "break_points": sorted(breaks)}


def masters_block_permutation(returns, *, n_perm: int = 1000, block_len: int = 5,
                              ppy: int = 252, seed: int = 42) -> dict:
    """Block sign-flip permutation Sharpe test (Masters 2018): flip random blocks'
    signs, recompute Sharpe; p = P(null Sharpe ≥ observed)."""
    r = _arr(returns)
    n = len(r)
    if n < 60:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "verdict": "insufficient_data"}
    obs = _sharpe(r, ppy)
    rng = np.random.default_rng(seed)
    n_blocks = (n + block_len - 1) // block_len
    null = np.array([_sharpe(r * np.repeat(rng.choice([-1, 1], size=n_blocks), block_len)[:n], ppy)
                     for _ in range(n_perm)])
    p = float((1 + int((null >= obs).sum())) / (1 + n_perm))
    verdict = "EDGE_CONFIRMED" if p < 0.05 else "BORDERLINE" if p < 0.20 else "NO_EDGE"
    return {"observed_sharpe": obs, "p_value": p, "verdict": verdict,
            "null_p5": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95))}


def masters_optimization_bias(returns, n_alternatives: int, *, n_perm: int = 200,
                              ppy: int = 252, seed: int = 42) -> dict:
    """Bootstrap best-of-N null for an *optimised* Sharpe: at each draw take the max
    Sharpe over ``n_alternatives`` bootstrap samples; report the inflation factor
    (observed / E[max]) and p-value. Empirical analogue of expected-max-Sharpe."""
    r = _arr(returns)
    n = len(r)
    if n < 30 or n_alternatives < 2:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "verdict": "insufficient_data"}
    obs = _sharpe(r, ppy)
    rng = np.random.default_rng(seed)
    null_max = np.array([
        max(_sharpe(r[rng.integers(0, n, size=n)], ppy) for _ in range(n_alternatives))
        for _ in range(n_perm)])
    e_max = float(null_max.mean())
    p = float((1 + int((null_max >= obs).sum())) / (1 + n_perm))
    return {"observed_sharpe": obs, "null_e_max_sharpe": e_max,
            "inflation_factor": obs / e_max if abs(e_max) > 1e-9 else float("inf"),
            "p_value": p, "n_alternatives": n_alternatives,
            "verdict": ("EDGE_BEYOND_OPTIM_BIAS" if p < 0.05
                        else "BORDERLINE" if p < 0.20 else "OPTIM_BIAS_LIKELY")}
