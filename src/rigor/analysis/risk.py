"""Advanced risk & drawdown metrics (#10).

The ratios ``rigor.metrics`` doesn't carry: VaR/CVaR (historical + Cornish-Fisher),
Omega, Ulcer Index + UPI, tail ratio, Rachev, CDaR, pain index, gain-to-pain,
Sterling, Burke — plus drawdown-event extraction (depth/duration/recovery),
time-under-water, and the current underwater streak. Pure NumPy; CAGR comes from
``rigor.metrics`` so a Sterling/Burke ratio here matches the catalog's CAGR.

Ported from the prior research library. All functions accept an array or a
pandas Series of periodic returns.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import metrics as _m

__all__ = [
    "value_at_risk", "conditional_var", "cornish_fisher_var", "gaussian_var",
    "omega", "ulcer_index", "ulcer_performance_index", "tail_ratio", "rachev_ratio",
    "cdar", "pain_index", "gain_to_pain", "sterling_ratio", "burke_ratio",
    "regime_conditional_risk", "risk_summary",
    "DrawdownEvent", "drawdown_series", "top_drawdowns",
    "time_under_water", "current_underwater",
]


def _arr(returns) -> np.ndarray:
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    r = r.astype("float64")
    return r[np.isfinite(r)]


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) and np.isfinite(b) else default


def _cagr(returns, ppy: int | None) -> float:
    arr = _arr(returns)
    if ppy is None:
        ppy = (_m.periods_per_year_of(returns.index)
               if isinstance(returns, pd.Series) else _m.TRADING_DAYS)
    return float(_m.compute_cagr(arr, ppy)) if len(arr) >= 2 else 0.0


# --- VaR / CVaR ------------------------------------------------------------

def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical VaR: the ``alpha`` quantile of returns (a negative number)."""
    r = _arr(returns)
    return float(np.percentile(r, alpha * 100)) if len(r) else 0.0


def conditional_var(returns, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of returns at or below the VaR threshold."""
    r = _arr(returns)
    if not len(r):
        return 0.0
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def cornish_fisher_var(returns, alpha: float = 0.05) -> float:
    """Modified VaR with a Cornish-Fisher skew/kurtosis expansion of the quantile."""
    r = _arr(returns)
    if len(r) < 4:
        return value_at_risk(r, alpha)
    from scipy.stats import kurtosis, skew
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    s = float(skew(r, bias=False))
    k = float(kurtosis(r, fisher=True, bias=False))
    z = norm.ppf(alpha)
    zc = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
          - (2 * z**3 - 5 * z) * s**2 / 36)
    return float(mu + zc * sd)


def gaussian_var(returns, alpha: float = 0.05) -> float:
    """Delta-normal (plain Gaussian / parametric) VaR at confidence level ``1 − alpha``.

    Returns ``μ + z_alpha · σ`` where z_alpha is the standard-normal quantile at
    ``alpha`` (a negative number for alpha < 0.5) — so the result is a **negative
    number** representing the loss at the given confidence level, consistent with the
    sign convention of :func:`value_at_risk` (historical-sim) and
    :func:`cornish_fisher_var` in this module.

    Contrast with the other two VaR estimators
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * :func:`value_at_risk` (historical-sim) — uses the empirical ``alpha``-quantile
      of the actual return distribution; no distributional assumption; sensitive to
      sample size and tail sparsity.
    * :func:`cornish_fisher_var` (parametric-with-moments) — extends the Gaussian
      quantile with a Cornish-Fisher expansion that adjusts for observed skewness and
      excess kurtosis; closer to historical-sim for fat-tailed returns.
    * ``gaussian_var`` (this function) — pure parametric; assumes returns are iid
      Normal(μ, σ²); fastest and most stable with small samples but underestimates
      tail risk for skewed / fat-tailed strategies.

    Uses ``statistics.NormalDist`` (Python 3.8+ stdlib) — no scipy/statsmodels
    dependency required for this function (scipy is imported elsewhere in this module
    for Cornish-Fisher but is not used here).

    The inline equivalent in ``rigor.metrics._gaussian_var_inline`` uses the same
    formula and will produce identical results for the same input array.
    """
    r = _arr(returns)
    if len(r) < 2:
        return 0.0
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z_alpha = statistics.NormalDist().inv_cdf(alpha)
    return float(mu + z_alpha * sigma)


# --- gain/loss & tail ratios ----------------------------------------------

def omega(returns, threshold: float = 0.0) -> float:
    """Omega ratio (Keating-Shadwick): Σ gains above / Σ losses below threshold.

    ``threshold`` is a **per-period** return in the same units as ``returns`` (not
    an annual figure) — for a risk-free MAR pass ``rf_annual / periods_per_year``.
    """
    r = _arr(returns)
    if not len(r):
        return 0.0
    return _safe_div(np.sum(np.maximum(r - threshold, 0.0)),
                     np.sum(np.maximum(threshold - r, 0.0)))


def tail_ratio(returns, confidence: float = 0.95) -> float:
    """Right-tail / left-tail magnitude ratio (>1 = favourable asymmetry)."""
    r = _arr(returns)
    if len(r) < 2:
        return 0.0
    return _safe_div(np.percentile(r, confidence * 100),
                     abs(np.percentile(r, (1 - confidence) * 100)))


def rachev_ratio(returns, alpha: float = 0.05, beta: float = 0.05) -> float:
    """Rachev ratio: mean of the right tail / mean magnitude of the left tail."""
    r = _arr(returns)
    if len(r) < 2:
        return 0.0
    gains = r[r >= np.percentile(r, (1 - alpha) * 100)]
    losses = r[r <= np.percentile(r, beta * 100)]
    if not len(gains) or not len(losses):
        return 0.0
    return _safe_div(float(gains.mean()), abs(float(losses.mean())))


def gain_to_pain(returns) -> float:
    """Gain-to-pain (Schwager): Σ returns / Σ |negative returns|."""
    r = _arr(returns)
    if not len(r):
        return 0.0
    return _safe_div(float(r.sum()), float(np.abs(r[r < 0]).sum()))


# --- drawdown-based ratios -------------------------------------------------

def _underwater(r: np.ndarray) -> np.ndarray:
    """Underwater curve ``eq/peak - 1`` (<= 0) for a prepared returns array.

    Computed in log space (``log1p`` / ``cumsum`` / ``expm1``) so a long winning
    streak cannot overflow ``cumprod`` to ``+inf`` and turn the whole drawdown
    path into NaN. Equivalent to the direct equity ratio for realistic inputs;
    only the overflow tail differs. Mirrors ``rigor.metrics.compute_max_drawdown``.
    """
    if not len(r):
        return r
    log_cum = np.cumsum(np.log1p(r))
    return np.expm1(log_cum - np.maximum.accumulate(log_cum))


def ulcer_index(returns) -> float:
    """Martin Ulcer Index: RMS of the percent drawdown path."""
    r = _arr(returns)
    if not len(r):
        return 0.0
    dd = _underwater(r) * 100.0
    return float(np.sqrt(np.mean(dd**2)))


def ulcer_performance_index(returns, ppy: int | None = None, risk_free: float = 0.0) -> float:
    """UPI / Martin ratio = (CAGR − rf) / Ulcer Index."""
    ui = ulcer_index(returns)
    return float((_cagr(returns, ppy) - risk_free) / (ui / 100.0)) if ui > 0 else 0.0


def cdar(returns, confidence: float = 0.95) -> float:
    """Conditional Drawdown at Risk (Chekhlov): mean of the worst (1−c) drawdowns."""
    r = _arr(returns)
    if not len(r):
        return 0.0
    dd = -_underwater(r)                         # positive drawdown depths
    tail = dd[dd >= np.percentile(dd, confidence * 100)]
    return float(tail.mean()) if len(tail) else 0.0


def pain_index(returns) -> float:
    """Pain index: mean depth of the drawdown path."""
    r = _arr(returns)
    if not len(r):
        return 0.0
    return float(np.abs(_underwater(r)).mean())


def sterling_ratio(returns, n_drawdowns: int = 5, ppy: int | None = None) -> float:
    """Sterling ratio = CAGR / mean depth of the top-N drawdowns."""
    events = top_drawdowns(returns, n=n_drawdowns)
    if not events:
        return 0.0
    return _safe_div(_cagr(returns, ppy), float(np.mean([abs(e.depth) for e in events])))


def burke_ratio(returns, n_drawdowns: int = 5, ppy: int | None = None) -> float:
    """Burke ratio = CAGR / sqrt(mean of squared top-N drawdown depths)."""
    events = top_drawdowns(returns, n=n_drawdowns)
    if not events:
        return 0.0
    depths = np.array([e.depth for e in events], dtype="float64")
    return _safe_div(_cagr(returns, ppy), float(np.sqrt(np.mean(depths**2))))


def regime_conditional_risk(returns, regime_labels, *, confidence: float = 0.95,
                            ppy: int = 252) -> dict:
    """VaR / CVaR / Sharpe conditional on each regime label."""
    r = _arr(returns)
    reg = (regime_labels.to_numpy() if isinstance(regime_labels, pd.Series)
           else np.asarray(regime_labels))
    n = min(len(r), len(reg))
    r, reg = r[:n], reg[:n]
    scale = float(np.sqrt(ppy))
    out: dict = {"confidence": float(confidence)}
    for label in pd.unique(reg):
        sub = r[reg == label]
        if len(sub) < 5:
            continue
        var = float(np.percentile(sub, (1 - confidence) * 100))
        tail = sub[sub <= var]
        vol = float(sub.std(ddof=1))
        out[str(label)] = {
            "var": var, "cvar": float(tail.mean()) if len(tail) else var,
            "mean_return": float(sub.mean()), "vol": vol,
            "sharpe": float(sub.mean() / vol * scale) if vol > 1e-12 else 0.0,
            "n": int(len(sub)),
        }
    return out


# --- drawdown events -------------------------------------------------------

@dataclass
class DrawdownEvent:
    start: int
    trough: int
    end: int            # recovery index, -1 if still underwater
    depth: float
    duration_days: int
    drawdown_days: int
    recovery_days: int


def drawdown_series(returns) -> pd.Series:
    """Underwater series (equity / running-max − 1)."""
    if isinstance(returns, pd.Series):
        vals = _underwater(returns.fillna(0.0).to_numpy(dtype="float64"))
        return pd.Series(vals, index=returns.index)
    r = np.asarray(returns, dtype="float64")
    if not len(r):
        return pd.Series(dtype="float64")
    return pd.Series(_underwater(r))


def top_drawdowns(returns, n: int = 10) -> list[DrawdownEvent]:
    """The N deepest drawdown episodes with peak/trough/recovery indices."""
    r = _arr(returns)
    if not len(r):
        return []
    dd = _underwater(r)
    events: list[DrawdownEvent] = []
    in_dd, start, trough_i, trough_v = False, 0, 0, 0.0
    for i in range(len(dd)):
        if dd[i] < -1e-10:
            if not in_dd:
                in_dd, start, trough_i, trough_v = True, i, i, dd[i]
            elif dd[i] < trough_v:
                trough_i, trough_v = i, dd[i]
        elif in_dd:
            events.append(DrawdownEvent(start, trough_i, i, float(trough_v),
                                        i - start, trough_i - start, i - trough_i))
            in_dd = False
    if in_dd:
        events.append(DrawdownEvent(start, trough_i, -1, float(trough_v),
                                    len(dd) - start, trough_i - start, -1))
    events.sort(key=lambda e: e.depth)
    return events[:n]


def time_under_water(returns) -> dict:
    """Distribution of drawdown durations + recovery times across all episodes."""
    events = top_drawdowns(returns, n=100000)
    if not events:
        return {"mean": 0.0, "median": 0.0, "max": 0, "p95": 0.0, "count": 0}
    dur = [e.duration_days for e in events]
    rec = [e.recovery_days for e in events if e.recovery_days >= 0]
    return {
        "mean": float(np.mean(dur)), "median": float(np.median(dur)),
        "max": int(np.max(dur)), "count": len(events),
        "p95": float(np.percentile(dur, 95)) if len(dur) >= 2 else float(dur[0]),
        "avg_recovery": float(np.mean(rec)) if rec else -1.0,
        "max_recovery": int(np.max(rec)) if rec else -1,
    }


def current_underwater(returns) -> dict:
    """Ongoing drawdown depth + consecutive days underwater at the end of the series."""
    dd = drawdown_series(returns)
    if not len(dd) or float(dd.iloc[-1]) >= -1e-10:
        return {"is_underwater": False, "current_dd": 0.0, "days_underwater": 0}
    days = 0
    for i in range(len(dd) - 1, -1, -1):
        if dd.iloc[i] < -1e-10:
            days += 1
        else:
            break
    return {"is_underwater": True, "current_dd": float(dd.iloc[-1]), "days_underwater": days}


def risk_summary(returns, ppy: int | None = None) -> dict:
    """One call returning the whole battery (for a report card / tearsheet)."""
    return {
        "var_95": value_at_risk(returns, 0.05), "cvar_95": conditional_var(returns, 0.05),
        "cornish_fisher_var_95": cornish_fisher_var(returns, 0.05),
        "var_normal_95": gaussian_var(returns, 0.05),
        "omega": omega(returns), "tail_ratio": tail_ratio(returns),
        "rachev": rachev_ratio(returns), "gain_to_pain": gain_to_pain(returns),
        "ulcer_index": ulcer_index(returns), "upi": ulcer_performance_index(returns, ppy),
        "cdar_95": cdar(returns), "pain_index": pain_index(returns),
        "sterling": sterling_ratio(returns, ppy=ppy), "burke": burke_ratio(returns, ppy=ppy),
    }
