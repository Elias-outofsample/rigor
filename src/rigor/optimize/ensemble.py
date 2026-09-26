"""Parameter ensemble: don't bet on one config, average a diverse few.

A single winning config is a point estimate that can be a noise spike. An
ensemble picks ``k`` configs that are both high-Sharpe **and** spread out in
parameter space (greedy max-diversity selection), then equal-weights their
return streams. If the ensemble Sharpe holds up near the single winner's, the
edge is broad (robust to the exact params); if the ensemble collapses, the
winner was a lucky point. Ported from the prior research library's
``build_ensemble``, reusing ``rigor.metrics``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = ["build_ensemble"]


def _numeric_param_matrix(param_dicts: list[dict]) -> np.ndarray | None:
    """Normalised matrix of the numeric params (for diversity distance), or None."""
    keys = [k for k in param_dicts[0]
            if all(isinstance(p.get(k), (int, float)) for p in param_dicts)]
    if not keys:
        return None
    arr = np.array([[float(p[k]) for k in keys] for p in param_dicts], dtype="float64")
    rng = arr.max(axis=0) - arr.min(axis=0)
    rng = np.where(rng == 0, 1.0, rng)
    return (arr - arr.min(axis=0)) / rng


def build_ensemble(rows: list[dict], n_members: int = 5, *, diversity_weight: float = 0.5) -> dict:
    """Select up to ``n_members`` diverse high-Sharpe configs and equal-weight them.

    ``rows`` is the ranked ``{"params", "sharpe", "returns"}`` list (best first).
    Returns the chosen members, the equal-weight ensemble's Sharpe, and how it
    compares to the single best config.
    """
    n = min(n_members, len(rows))
    if n < 2:
        return {"members": [], "note": "need >= 2 configs for an ensemble"}

    sharpes = np.array([r["sharpe"] if r["sharpe"] == r["sharpe"] else -1e18 for r in rows])
    s_rng = sharpes.max() - sharpes.min()
    score = (sharpes - sharpes.min()) / s_rng if s_rng > 0 else np.full(len(rows), 0.5)
    pmat = _numeric_param_matrix([r["params"] for r in rows])

    selected = [int(np.argmax(score))]
    while len(selected) < n:
        best_cand, best_val = -1, -np.inf
        for i in range(len(rows)):
            if i in selected:
                continue
            if pmat is not None:
                min_dist = float(np.min(np.linalg.norm(pmat[i] - pmat[selected], axis=1)))
            else:
                min_dist = 0.0
            val = score[i] + diversity_weight * min_dist
            if val > best_val:
                best_val, best_cand = val, i
        if best_cand < 0:
            break
        selected.append(best_cand)

    streams = pd.concat([rows[i]["returns"].rename(i) for i in selected], axis=1).dropna()
    ens_returns = streams.mean(axis=1)
    ppy = _m.periods_per_year_of(ens_returns.index)
    ens_sharpe = _m.compute_sharpe(ens_returns.to_numpy(dtype="float64"), ppy)
    best_sharpe = float(rows[0]["sharpe"])
    return {
        "n_members": len(selected),
        "members": [rows[i]["params"] for i in selected],
        "ensemble_sharpe": round(float(ens_sharpe), 4),
        "best_single_sharpe": round(best_sharpe, 4),
        "ensemble_vs_best": round(float(ens_sharpe) - best_sharpe, 4),
    }
