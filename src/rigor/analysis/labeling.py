"""Triple-barrier labeling, meta-labeling & sequential bootstrap (#8).

López de Prado, *Advances in Financial Machine Learning* Ch.3-4. Pure NumPy/
pandas, no heavy dependencies — port verbatim.

  * ``triple_barrier_labels`` — label each event +1/-1/0 by which of a vol-scaled
    take-profit / stop-loss / timeout barrier is hit first.
  * ``meta_labels`` — binary "was the primary signal right?" for a second model.
  * ``sample_weights_uniqueness`` — down-weight overlapping (concurrent) labels.
  * ``sequential_bootstrap_indices`` — resample favouring *unique* labels.

Why it matters: when labels overlap (long holding horizons), a uniform bootstrap
over-represents redundant labels, inflates the effective sample size, and makes
confidence intervals too tight. Uniqueness weights + sequential bootstrap correct
that bias — directly relevant to overlapping-window strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "triple_barrier_labels", "meta_labels",
    "sample_weights_uniqueness", "sequential_bootstrap_indices",
]


def triple_barrier_labels(close: pd.Series, events: pd.DatetimeIndex,
                          pt_sl: tuple[float, float], num_days: int) -> pd.DataFrame:
    """Label each event +1 (take-profit) / -1 (stop-loss) / 0 (timeout).

    ``pt_sl`` are TP/SL multipliers of the EWMA(span=100) daily log-vol;
    ``num_days`` is the vertical (timeout) barrier. Returns a frame indexed by
    event date with columns ``label``, ``ret``, ``t_exit``.
    """
    close = close.astype("float64")
    pt_mult, sl_mult = float(pt_sl[0]), float(pt_sl[1])
    log_ret = np.log(close / close.shift(1)).dropna()
    vol = log_ret.ewm(span=100).std().reindex(close.index).ffill()

    rows = []
    for t0 in events:
        if t0 not in close.index:
            continue
        p0 = close.loc[t0]
        loc0 = close.index.get_loc(t0)
        fwd = close.iloc[loc0 + 1: min(loc0 + num_days + 1, len(close))]
        if fwd.empty:
            rows.append({"label": 0, "ret": 0.0, "t_exit": t0})
            continue
        sigma = float(vol.loc[t0]) if not np.isnan(vol.loc[t0]) else 0.0
        tp = p0 * (1.0 + pt_mult * sigma) if pt_mult > 0 else np.inf
        sl = p0 * (1.0 - sl_mult * sigma) if sl_mult > 0 else -np.inf
        label, t_exit, exit_p = 0, fwd.index[-1], fwd.iloc[-1]
        for t1, p1 in fwd.items():
            if p1 >= tp:
                label, t_exit, exit_p = 1, t1, p1
                break
            if p1 <= sl:
                label, t_exit, exit_p = -1, t1, p1
                break
        rows.append({"label": label, "ret": float((exit_p - p0) / p0) if p0 else 0.0,
                     "t_exit": t_exit})

    valid = events[events.isin(close.index)]
    if not rows:
        return pd.DataFrame(columns=["label", "ret", "t_exit"])
    df = pd.DataFrame(rows, index=valid)
    df["label"] = df["label"].astype("int8")
    df["ret"] = df["ret"].astype("float64")
    return df


def meta_labels(primary_signals: pd.Series, returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary meta-label: 1 if the primary signal's sign matched the forward
    return, 0 otherwise (0 for no-signal bars)."""
    fwd = returns.reindex(primary_signals.index).shift(-horizon)
    meta = np.where(primary_signals == 0, 0,
                    np.where(np.sign(primary_signals) == np.sign(fwd), 1, 0))
    return pd.Series(meta, index=primary_signals.index, dtype="int8", name="meta_label")


def sample_weights_uniqueness(events: pd.DatetimeIndex, close: pd.Series,
                              num_days: int = 20) -> pd.Series:
    """Per-label weight = mean of 1/concurrency over the label's holding span
    (overlapping labels get less weight). Normalised to sum to n_events."""
    dates = close.index
    valid = events[events.isin(dates)]
    spans = []
    for t0 in valid:
        loc0 = dates.get_loc(t0)
        spans.append((t0, dates[min(loc0 + num_days, len(dates) - 1)]))
    conc = pd.Series(0.0, index=dates, dtype="float64")
    for t0, t1 in spans:
        conc.loc[(dates >= t0) & (dates <= t1)] += 1.0
    weights = []
    for t0, t1 in spans:
        span = conc.loc[(dates >= t0) & (dates <= t1)].replace(0.0, 1.0)
        weights.append(float((1.0 / span).mean()))
    w = pd.Series(weights, index=valid, dtype="float64", name="sample_weight")
    total = w.sum()
    return w / total * len(w) if total > 0 else w


def sequential_bootstrap_indices(sample_weights, n_samples: int | None = None,
                                 seed: int = 42) -> np.ndarray:
    """Sequential bootstrap (AFML §4.5.2): draw indices with probability ∝ label
    uniqueness, penalising already-drawn indices so redundant labels don't
    dominate the resample."""
    w = (sample_weights.to_numpy() if hasattr(sample_weights, "to_numpy")
         else np.asarray(sample_weights, dtype="float64")).astype("float64")
    n = len(w)
    if n == 0:
        return np.array([], dtype="int64")
    n_samples = n_samples or n
    rng = np.random.default_rng(seed)
    probs = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    out = np.empty(n_samples, dtype="int64")
    drawn = np.zeros(n, dtype="float64")
    for k in range(n_samples):
        adj = probs / (1.0 + drawn)
        adj = adj / adj.sum() if adj.sum() > 0 else probs
        i = int(rng.choice(n, p=adj))
        out[k] = i
        drawn[i] += 1.0
    return out
