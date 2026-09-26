"""Per-regime optimisation — which config wins in which market state.

A single global winner can be a blend that is mediocre everywhere. This slices
the searched configs' return streams by market regime and finds the best config
*within each regime* (train/test split per regime so the per-regime pick is
honest), then reports a **consensus** — the config that wins the most regimes.
Ported from the prior research library's ``optimize_per_regime``.

Regimes default to volatility terciles (low/med/high) of the cross-config mean
return — always available, no external series needed; pass ``regime_labels`` to
use your own (e.g. an HMM or trend state).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = ["label_vol_terciles", "optimize_per_regime"]


def label_vol_terciles(returns: pd.Series, window: int = 21) -> pd.Series:
    """Label each bar 0/1/2 by the tercile of its trailing realised volatility."""
    vol = returns.rolling(window).std()
    valid = vol.dropna()
    if len(valid) < 3:
        return pd.Series(0, index=returns.index)
    lo, hi = valid.quantile([1 / 3, 2 / 3])
    lab = pd.Series(1, index=returns.index)
    lab[vol <= lo] = 0
    lab[vol > hi] = 2
    return lab


def _sharpe(returns: np.ndarray, ppy: int) -> float:
    return _m.compute_sharpe(returns, ppy) if len(returns) >= 2 else float("nan")


def optimize_per_regime(
    rows: list[dict], *, regime_labels: pd.Series | None = None,
    min_regime_obs: int = 60, min_oos_obs: int = 30, train_frac: float = 0.7,
) -> dict:
    """For each regime: pick the best config on a train slice, measure it OOS.

    ``rows`` is the ranked ``{"params", "returns"}`` list. Returns per-regime
    winners, the consensus config, and regime coverage.
    """
    matrix = pd.concat([r["returns"].rename(i) for i, r in enumerate(rows)], axis=1).dropna()
    if len(matrix) < min_regime_obs:
        return {"note": "series too short for regime optimisation"}
    ppy = _m.periods_per_year_of(matrix.index)
    labels = (label_vol_terciles(matrix.mean(axis=1)) if regime_labels is None
              else regime_labels.reindex(matrix.index).ffill())
    arr = matrix.to_numpy(dtype="float64")

    per_regime: dict[int, dict[str, object]] = {}
    votes: dict[tuple, int] = {}
    coverage: dict[int, float] = {}
    for rid in sorted(int(x) for x in pd.unique(labels.dropna())):
        mask = (labels == rid).to_numpy()
        coverage[rid] = round(float(mask.sum()) / len(labels), 4)
        block = arr[mask]
        if len(block) < min_regime_obs:
            per_regime[rid] = {"skipped": True, "n_obs": int(len(block))}
            continue
        split = int(round(len(block) * train_frac))
        train, oos = block[:split], block[split:]
        if len(oos) < min_oos_obs:
            per_regime[rid] = {"skipped": True, "n_obs": int(len(block))}
            continue
        is_sr = np.array([_sharpe(train[:, j], ppy) for j in range(arr.shape[1])])
        if np.all(np.isnan(is_sr)):
            per_regime[rid] = {"skipped": True, "n_obs": int(len(block))}
            continue
        b = int(np.nanargmax(is_sr))
        per_regime[rid] = {
            "best_params": rows[b]["params"], "train_sharpe": round(float(is_sr[b]), 4),
            "oos_sharpe": round(float(_sharpe(oos[:, b], ppy)), 4),
            "n_obs": int(len(block)), "n_train": int(split), "n_oos": int(len(oos)),
        }
        key = tuple(sorted(rows[b]["params"].items()))
        votes[key] = votes.get(key, 0) + 1

    if not votes:
        return {"per_regime": per_regime, "regime_coverage": coverage, "n_regimes": len(per_regime)}
    best_key = max(votes, key=lambda k: votes[k])
    return {
        "per_regime": per_regime,
        "consensus_params": dict(best_key), "consensus_votes": votes[best_key],
        "regime_coverage": coverage, "n_regimes": len(per_regime),
    }
