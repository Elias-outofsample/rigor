"""Forensic reconstruction of a strategy's entry signal via candidate IC.

The backtest engine doesn't persist the raw indicator that fired each trade, so
we cannot read a strategy's *own* signal. Instead we reconstruct a battery of
standard candidate signals from price history at each trade's entry and measure
how well each rank-predicts the trade's realised forward return across the whole
trade panel.

This reveals **which known anomaly** the entries actually load on — short-term
reversal, 12-1 momentum, RSI2, IBS, distance-from-MA, low-vol — and whether that
edge has genuine cross-sectional predictive power **and** monotonicity (a real
signal lines its quintiles up, it isn't just strong in the extremes). It is the
"dim-1 signal" test done honestly from data that exists.

This is a *different, forensic* use of rank-IC from
:mod:`rigor.analysis.signal_quality`: there ``rank_ic`` scores one known signal's
forward predictivity; here we score a *panel of competing reconstructed
candidates* to identify the dominant driver. We reuse
``signal_quality.rank_ic`` rather than re-implement it. Pure numpy/pandas.

References
----------
Connors & Alvarez (2009), RSI2 / IBS short-term mean-reversion.
Jegadeesh & Titman (1993), 12-1 momentum. Lehmann (1990), short-term reversal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signal_quality import rank_ic, xi_ic

__all__ = [
    "quintile_monotonicity",
    "reconstruct_candidate_signals",
    "compute_entry_signal_ic",
]

# Minimum panel size per bucket / overall for a result to be meaningful.
_MIN_PER_BUCKET = 5


def quintile_monotonicity(signal, forward_returns, n_buckets: int = 5) -> dict:
    """Mean forward return per signal quintile + a monotonicity flag.

    A real signal shows a roughly monotone progression from the low-signal bucket
    to the high-signal bucket — not just strong extremes with a noisy middle.
    ``spread`` is the top-minus-bottom bucket mean. Buckets are formed by an
    equal-count rank split (stable on ties via a mergesort argsort), so this is
    robust to non-uniform signal distributions.
    """
    s = np.asarray(signal, dtype="float64")
    f = np.asarray(forward_returns, dtype="float64")
    mask = np.isfinite(s) & np.isfinite(f)
    s, f = s[mask], f[mask]
    if len(s) < n_buckets * _MIN_PER_BUCKET:
        return {"buckets": None, "monotonic": None, "spread": None,
                "n": int(len(s))}
    order = np.argsort(s, kind="mergesort")
    buckets = np.array_split(order, n_buckets)
    means = [float(np.mean(f[b])) for b in buckets]
    diffs = np.diff(means)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return {
        "buckets": [round(m, 6) for m in means],
        "monotonic": monotonic,
        "spread": round(means[-1] - means[0], 6),
        "n": int(len(s)),
    }


def reconstruct_candidate_signals(
    close: pd.Series,
    entry_index,
    *,
    rsi_window: int = 2,
    mom_window: int = 252,
    mom_skip: int = 21,
    ma_window: int = 50,
    vol_window: int = 20,
) -> dict:
    """Reconstruct standard candidate entry signals at each trade's entry bar.

    Builds, from a single ``close`` price series, the classic anomaly signals as
    of each entry timestamp in ``entry_index`` (positional or label-based):

      * ``rsi2`` — Connors RSI on ``rsi_window`` bars (low ⇒ oversold).
      * ``momentum`` — Jegadeesh-Titman ``mom_window``-``mom_skip`` return.
      * ``reversal`` — negated prior 1-bar return (short-term reversal).
      * ``ibs`` — internal-bar strength proxy from the close vs its rolling range.
      * ``ma_distance`` — (close − SMA) / SMA, distance from the moving average.
      * ``low_vol`` — negated rolling realised vol (high value ⇒ calm regime).

    Returns ``{signal_name: 1-D float array aligned to entry_index}`` ready for
    :func:`compute_entry_signal_ic`. Entries before a signal's warm-up window get
    ``NaN`` (excluded downstream). Pure pandas/numpy.
    """
    c = pd.Series(close).astype("float64")
    ret1 = c.pct_change()

    # RSI (Wilder) on rsi_window.
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_window, min_periods=rsi_window).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_window, min_periods=rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    momentum = c.shift(mom_skip) / c.shift(mom_window) - 1.0
    reversal = -ret1
    roll = c.rolling(vol_window, min_periods=max(2, vol_window // 2))
    rng = roll.max() - roll.min()
    ibs = (c - roll.min()) / rng.replace(0.0, np.nan)
    sma = c.rolling(ma_window, min_periods=max(2, ma_window // 2)).mean()
    ma_distance = (c - sma) / sma.replace(0.0, np.nan)
    low_vol = -(ret1.rolling(vol_window, min_periods=max(2, vol_window // 2)).std())

    frame = pd.DataFrame({
        "rsi2": rsi,
        "momentum": momentum,
        "reversal": reversal,
        "ibs": ibs,
        "ma_distance": ma_distance,
        "low_vol": low_vol,
    })

    idx = pd.Index(entry_index)
    if idx.isin(frame.index).all():
        at_entry = frame.reindex(idx)
    else:  # treat as positional locations into the price series
        pos = np.asarray(entry_index, dtype="int64")
        pos = pos[(pos >= 0) & (pos < len(frame))]
        at_entry = frame.iloc[pos]
    return {name: at_entry[name].to_numpy(dtype="float64") for name in frame.columns}


def compute_entry_signal_ic(signals: dict, forward_returns) -> dict:
    """Per-candidate rank IC + significance + quintile monotonicity.

    Args:
        signals: ``{signal_name: 1-D array aligned to forward_returns}`` — e.g. the
            output of :func:`reconstruct_candidate_signals`.
        forward_returns: realised per-trade forward return.

    Returns a dict with a per-signal list (sorted by ``|IC|``), the dominant
    signal name + its IC, and the panel size. Each row carries the Spearman rank
    IC, Chatterjee's ξ (``xi`` — complements the Spearman IC by catching
    non-monotonic signal→return dependence the rank IC is blind to), an
    approximate t-stat (``IC·√(m−1)`` under H0), the usable count, the
    monotonicity flag and quintile means. Degenerate inputs (empty / mismatched /
    too short) yield an empty signal list and ``dominant=None`` without raising.
    """
    fwd = np.asarray(forward_returns, dtype="float64")
    n = int(np.isfinite(fwd).sum())
    rows: list[dict] = []
    for name, raw in signals.items():
        s = np.asarray(raw, dtype="float64")
        if s.shape != fwd.shape:
            continue
        ic = rank_ic(s, fwd)
        xi = xi_ic(s, fwd)
        m = int((np.isfinite(s) & np.isfinite(fwd)).sum())
        # Spearman IC ~ N(0, 1/(m-1)) under H0 → t ≈ IC·√(m-1).
        t_stat = float(ic * np.sqrt(max(m - 1, 1))) if m > 2 else 0.0
        mono = quintile_monotonicity(s, fwd)
        rows.append({
            "signal": name,
            "ic": round(ic, 4),
            "xi": round(xi, 4),
            "t_stat": round(t_stat, 2),
            "n": m,
            "monotonic": mono["monotonic"],
            "quintile_spread": mono["spread"],
            "quintiles": mono["buckets"],
        })
    rows.sort(key=lambda r: abs(r["ic"]), reverse=True)
    dominant = rows[0] if rows else None
    return {
        "n_trades": n,
        "signals": rows,
        "dominant": dominant["signal"] if dominant else None,
        "dominant_ic": dominant["ic"] if dominant else None,
    }
