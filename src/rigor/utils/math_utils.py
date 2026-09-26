"""Numerical helpers that eliminate recurring silent-error bug classes.

All functions are pure numpy/stdlib — no external dependencies beyond numpy and pandas.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Return a/b, falling back to *default* when b==0 or the result is non-finite.

    Handles numpy scalar inputs transparently.

    Examples
    --------
    >>> safe_divide(1, 0)
    0.0
    >>> safe_divide(1, 0, default=9)
    9
    >>> safe_divide(np.float64(2), np.float64(4))
    0.5
    """
    try:
        result = float(a) / float(b)
    except (ZeroDivisionError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def clip_safe(x: float, lo: float, hi: float) -> float:
    """Clip *x* to [lo, hi], mapping NaN to *lo*.

    Unlike numpy.clip, this never silently propagates NaN through the result.

    Examples
    --------
    >>> clip_safe(np.nan, 0, 1)
    0.0
    >>> clip_safe(5.0, 0, 1)
    1.0
    """
    v = float(x)
    if not math.isfinite(v):
        return float(lo)
    return float(np.clip(v, float(lo), float(hi)))


def max_consecutive(mask: np.ndarray | pd.Series) -> int:
    """Return the length of the longest consecutive True run in *mask*.

    Useful for measuring underwater streaks, drawdown duration, or signal persistence.

    Parameters
    ----------
    mask:
        Boolean array or Series.  NaN values are treated as False.

    Examples
    --------
    >>> max_consecutive(np.array([False, True, True, True, False, True]))
    3
    """
    if isinstance(mask, pd.Series):
        arr = mask.fillna(False).to_numpy(dtype=bool)
    else:
        arr = np.asarray(mask, dtype=bool)

    if arr.size == 0:
        return 0

    max_run = 0
    current = 0
    for val in arr:
        if val:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run
