"""Up/down capture ratios & stress-conditional crash-convexity flag (#capture).

Capture ratios answer "how much of the market does the strategy actually keep?"
— distinct from beta (a regression *slope*). Down-capture is the ratio of the
strategy's mean return to the benchmark's mean return on the benchmark's *down*
bars; below 1 (and below up-capture) is the convex profile one wants.

  * ``compute_capture_ratios`` — Morningstar up/down capture + their ratio,
    conditioned on the sign of the benchmark return.
  * ``compute_stress_capture`` — the same idea conditioned on the *tails* of the
    benchmark (worst/best ``stress_quantile`` bars), with a ``crash_convex`` flag.
    A down-capture that stays low on ``bench<0`` but explodes in the crash tail
    exposes a hidden fragility the sign-conditioned version misses.

Pure numpy/pandas, deterministic. Both functions degrade gracefully: a missing,
empty, or too-short benchmark yields a neutral result (zeros / ``False`` flag)
rather than raising or propagating NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.math_utils import safe_divide

__all__ = [
    "compute_capture_ratios",
    "compute_stress_capture",
]

_MIN_OBS = 20  # aligned observations required before any ratio is computed
_MIN_TAIL = 5  # observations required in a tail/regime before it is trusted


def _align(returns, benchmark) -> pd.DataFrame:
    """Index-align strategy & benchmark into a 2-col frame, dropping non-overlap.

    Accepts Series or array-likes (positionally paired if not Series). Returns an
    empty frame when either input is empty or there is no finite overlap.
    """
    r = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns))
    b = benchmark if isinstance(benchmark, pd.Series) else pd.Series(np.asarray(benchmark))
    df = pd.DataFrame({"r": r.astype("float64"), "b": b.astype("float64")})
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def _capture(seg: pd.DataFrame) -> float:
    """Mean(strategy) / mean(benchmark) over a regime, neutral if too few bars."""
    if len(seg) < 1:
        return 0.0
    return safe_divide(float(seg["r"].mean()), float(seg["b"].mean()))


def compute_capture_ratios(returns, benchmark) -> dict:
    """Morningstar up/down capture ratios (conditioned on the benchmark sign).

    ``up_capture``   = mean(strat | bench>0) / mean(bench | bench>0)
    ``down_capture`` = mean(strat | bench<0) / mean(bench | bench<0)
    ``capture_ratio``= up_capture / down_capture (>1 is favourable: keep more of
    the upside than the downside).

    Returns a neutral dict (all zeros) when fewer than ``_MIN_OBS`` aligned
    observations exist — never raises, never returns NaN.
    """
    df = _align(returns, benchmark)
    if len(df) < _MIN_OBS:
        return {"up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0}

    up = df[df["b"] > 0.0]
    down = df[df["b"] < 0.0]
    up_capture = _capture(up) if len(up) > 0 else 0.0
    down_capture = _capture(down) if len(down) > 0 else 0.0
    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": safe_divide(up_capture, down_capture),
    }


def compute_stress_capture(returns, benchmark, stress_quantile: float = 0.10) -> dict:
    """Stress-conditional capture on the benchmark *tails*, with a convexity flag.

    Restricts the "down" regime to the worst ``stress_quantile`` benchmark bars
    (a genuine crash, not just a red day) and "up" to the best ``stress_quantile``.
    A strategy whose tail down-capture blows past its up-capture is concave in the
    crash — ``crash_convex`` is True only when the tail profile is genuinely convex
    (positive up-capture and a strictly smaller, non-negative down-capture).

    Args:
        stress_quantile: tail width, clamped to (0, 0.5). 0.10 = worst/best 10%.

    Returns dict with ``up_capture`` / ``down_capture`` / ``capture_ratio``,
    ``crash_convex`` (bool), and ``n_up`` / ``n_down`` (tail sizes). Neutral
    result (zeros, ``crash_convex=False``) on fewer than ``_MIN_OBS`` aligned
    observations or a tail thinner than ``_MIN_TAIL`` points.
    """
    q = float(np.clip(stress_quantile, 1e-6, 0.5 - 1e-6))
    neutral = {
        "up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0,
        "crash_convex": False, "n_up": 0, "n_down": 0,
    }
    df = _align(returns, benchmark)
    if len(df) < _MIN_OBS:
        return neutral

    lo = df["b"].quantile(q)
    hi = df["b"].quantile(1.0 - q)
    down = df[df["b"] <= lo]
    up = df[df["b"] >= hi]
    if len(down) < _MIN_TAIL or len(up) < _MIN_TAIL:
        return neutral

    up_capture = _capture(up)
    down_capture = _capture(down)
    crash_convex = up_capture > 0.0 and 0.0 <= down_capture < up_capture
    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": safe_divide(up_capture, down_capture),
        "crash_convex": bool(crash_convex),
        "n_up": int(len(up)),
        "n_down": int(len(down)),
    }
