"""Walk-forward optimisation: does the *search itself* generalise?

Re-select the best config on each in-sample window and measure that config out
of sample. The gap between in-sample and out-of-sample Sharpe (and how often the
winner flips) tells you whether the optimisation is finding signal or fitting
noise — a different question from "is the winner good", which the verdict answers.

Two windowing schemes, mirroring the prior research library:

  * ``expanding`` (anchored) — the in-sample window starts at bar 0 and grows
    each fold; uses all history to date. Default.
  * ``rolling`` — a fixed-length in-sample window slides forward, so each fold
    re-selects on only recent history (surfaces regime sensitivity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = ["walk_forward_opt"]


def walk_forward_opt(
    matrix: pd.DataFrame, n_folds: int = 5, *, mode: str = "expanding", purge_gap: int = 0
) -> dict:
    """Per fold: pick the best config column on the in-sample window, then measure
    that column out-of-sample. ``matrix`` is (bars x configs).

    ``mode`` is ``"expanding"`` (in-sample grows from bar 0) or ``"rolling"``
    (fixed-width in-sample window slides forward).

    ``purge_gap`` drops that many in-sample bars immediately before each
    out-of-sample window so a label/lookback horizon cannot leak across the
    IS/OOS boundary — the contiguous-fold analogue of ``cpcv_pbo``'s purge.
    Opt-in; the default of 0 leaves results byte-identical.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}")
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
    n = len(matrix)
    step = n // (n_folds + 1)
    if step < 10:
        return {"n_folds": 0, "mode": mode, "note": "series too short for walk-forward"}
    ppy = _m.periods_per_year_of(matrix.index)
    arr = matrix.to_numpy(dtype="float64")
    is_sh, oos_sh, winners = [], [], []
    for k in range(1, n_folds + 1):
        is_start = 0 if mode == "expanding" else max(0, step * k - step)
        is_end = step * k
        oos_end = min(step * (k + 1), n)
        # Purge the in-sample bars adjacent to the OOS window (no-op when purge_gap=0).
        is_block = arr[is_start:max(is_start, is_end - purge_gap)]
        oos_block = arr[is_end:oos_end]
        if len(is_block) < 20 or len(oos_block) < 5:
            continue
        is_sharpes = np.array([_m.compute_sharpe(is_block[:, j], ppy) for j in range(arr.shape[1])])
        if np.all(np.isnan(is_sharpes)):
            continue
        j = int(np.nanargmax(is_sharpes))
        winners.append(j)
        is_sh.append(float(is_sharpes[j]))
        oos_sh.append(_m.compute_sharpe(oos_block[:, j], ppy))
    if not oos_sh:
        return {"n_folds": 0, "mode": mode, "note": "no usable folds"}
    is_mean, oos_mean = float(np.nanmean(is_sh)), float(np.nanmean(oos_sh))
    modal = max(set(winners), key=winners.count)
    return {
        "n_folds": len(oos_sh),
        "mode": mode,
        "is_sharpe_mean": round(is_mean, 4),
        "oos_sharpe_mean": round(oos_mean, 4),
        "is_to_oos_degradation": round(is_mean - oos_mean, 4),
        "oos_positive_folds": int(np.sum(np.array(oos_sh) > 0)),
        "param_stability": round(winners.count(modal) / len(winners), 3),
    }
