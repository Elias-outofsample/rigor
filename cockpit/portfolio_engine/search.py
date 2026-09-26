"""Allocation grid search — find robust weights, not just in-sample-optimal ones.

Samples the long-only simplex (corner allocations, every 50/50 pair, and Dirichlet
draws at mixed concentrations), estimates each candidate's out-of-sample stability
by rolling walk-forward (or combinatorial purged CV, ``use_cpcv=True``):

    stability = sharpe_oos_mean / (1 + sharpe_oos_std)

then re-ranks the leaders by a **composite anti-overfit score** that also rewards
deflated-Sharpe significance and calendar regularity:

    composite = stability · max(DSR, 0.3) · pos_months · pos_years

so an allocation brilliant in one regime but erratic across months/years loses to a
steady, statistically-significant one. Returns the top-K as a DataFrame with weights
+ OOS/full metrics. The search is deterministic (seeded). Metrics come from
``rigor.metrics`` / ``rigor.validation``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from rigor import metrics as _m

__all__ = ["weight_grid_search", "subset_grid_search"]


def _sharpe(arr: np.ndarray, ppy: int) -> float:
    return float(_m.compute_sharpe(arr, ppy)) if len(arr) >= 2 else float("nan")


def _col_sharpes(port: np.ndarray, ppy: float) -> np.ndarray:
    """Annualised Sharpe of each column of a (bars x samples) matrix.

    NaN-aware: a column may carry NaNs in common-window mode (bars where not every
    leg is live), so each column is scored over its own non-NaN bars.
    """
    if port.shape[0] < 2:
        return np.full(port.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(port, axis=0)
        sd = np.nanstd(port, axis=0, ddof=1)
        return np.where(sd > 1e-12, mu / sd * np.sqrt(ppy), np.nan)


def _portfolio_matrix(filled: np.ndarray, mask: np.ndarray, samples: np.ndarray,
                      *, common_window: bool = False) -> np.ndarray:
    """(bars x samples) portfolio returns.

    Default (``common_window=False``): the backtester's active-share rule — each bar
    renormalises over the *active* strategies, preserving target gross, so a leg that
    isn't live yet is carried by the others. Consistent with ``PortfolioBacktester``.

    ``common_window=True``: each candidate is only scored on bars where **every** of
    its weighted legs is live (active share ≈ 1); other bars are ``NaN``. This is the
    honest "only backtest where all strategies overlap" window, computed *per
    candidate* (so a young strategy only shortens the books that actually use it).
    """
    abs_s = np.abs(samples)
    gross = abs_s.sum(axis=1)
    gross_safe = np.where(gross > 1e-12, gross, 1.0)
    num = filled @ samples.T                      # (T, S)
    active = (mask.astype(float) @ abs_s.T) / gross_safe   # (T, S) active share
    if common_window:
        # keep only bars where the whole book is live (active share ≈ 1); elsewhere NaN
        return np.where(active >= 1.0 - 1e-9, num / gross_safe, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        port = num / np.where(active > 1e-12, active, np.nan)
    return np.nan_to_num(port, nan=0.0)


def _samples(n: int, n_samples: int, seed: int) -> np.ndarray:
    """Corner + pairwise + Dirichlet allocations on the long-only simplex."""
    eye = np.eye(n)
    rng = np.random.default_rng(seed)
    blocks = [eye]
    if n > 1:
        i, j = np.triu_indices(n, k=1)
        max_pairs = min(len(i), 200)
        if len(i) > max_pairs:
            pick = rng.choice(len(i), size=max_pairs, replace=False)
            i, j = i[pick], j[pick]
        pairs = np.zeros((len(i), n))
        pairs[np.arange(len(i)), i] = 0.5
        pairs[np.arange(len(j)), j] = 0.5
        blocks.append(pairs)
    n_dir = max(n_samples - sum(b.shape[0] for b in blocks), n_samples // 2)
    n1, n2 = int(n_dir * 0.5), int(n_dir * 0.3)
    n3 = max(n_dir - n1 - n2, 0)
    for conc, cnt in ((0.3, n1), (1.0, n2), (3.0, n3)):
        if cnt > 0:
            blocks.append(rng.dirichlet(np.full(n, conc), size=cnt))
    s = np.concatenate(blocks, axis=0)
    return s / s.sum(axis=1, keepdims=True)


def _folds(T: int, n_folds: int, *, initial_train_frac: float = 0.30,
           min_obs: int = 30) -> list[tuple[int, int]]:
    """Expanding-window rolling walk-forward test slices."""
    train0 = max(int(T * initial_train_frac), min_obs)
    remaining = T - train0
    if n_folds < 2 or remaining < n_folds * min_obs:
        return []
    seg = remaining // n_folds
    return [(train0 + k * seg, train0 + (k + 1) * seg if k < n_folds - 1 else T)
            for k in range(n_folds)]


def _cpcv_oos(port: np.ndarray, ppy: float, *, n_groups: int = 10,
              n_test: int = 2, min_obs: int = 20) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-sample OOS Sharpe mean/std across **combinatorial** purged-CV paths.

    Splits the timeline into ``n_groups`` contiguous blocks and holds out every
    combination of ``n_test`` of them as OOS — C(n_groups, n_test) paths (45 for
    10-choose-2) versus the 10 contiguous folds of walk-forward. Far more robust
    to regime clustering (a single crisis no longer dominates one fold's std).
    Returns ``(oos_mean, oos_std)`` aligned to the sample axis, or ``None``.
    """
    from itertools import combinations
    t = port.shape[0]
    if t < n_groups * min_obs:
        return None
    bounds = np.linspace(0, t, n_groups + 1, dtype=int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]
    paths = []
    for test in combinations(range(n_groups), n_test):
        idx = np.concatenate([groups[g] for g in test])
        if idx.size >= min_obs:
            paths.append(_col_sharpes(port[idx], ppy))
    if len(paths) < 3:
        return None
    arr = np.array(paths)                         # (paths, samples)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _regularity(series: np.ndarray, index: pd.DatetimeIndex) -> tuple[float, float]:
    """Fraction of positive calendar months and positive calendar years —
    rewards an allocation that earns steadily, not one big lucky bar."""
    ser = pd.Series(series, index=index)
    monthly = ser.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    yearly = ser.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)
    pos_m = float((monthly > 0).mean()) if len(monthly) else 0.0
    pos_y = float((yearly > 0).mean()) if len(yearly) else 0.0
    return pos_m, pos_y


def _rank_samples(samples: np.ndarray, filled: np.ndarray, mask: np.ndarray,
                  names: list, index: pd.DatetimeIndex, ppy: float, *,
                  n_folds: int, use_cpcv: bool, top_k: int,
                  common_window: bool = False, min_obs: int = 60) -> pd.DataFrame:
    """Score a batch of weight vectors and return the top-K composite-ranked rows.

    Shared by the dense :func:`weight_grid_search` and the sparse
    :func:`subset_grid_search` so both rank candidates identically. When
    ``common_window`` is set, each candidate is scored only over the bars where its
    whole book is live, and books with fewer than ``min_obs`` such bars are dropped.
    """
    n = len(names)
    index = np.asarray(index)
    port = _portfolio_matrix(filled, mask, samples, common_window=common_window)
    full_sr = _col_sharpes(port, ppy)
    valid_counts = np.sum(~np.isnan(port), axis=0)
    if common_window:
        full_sr = np.where(valid_counts >= min_obs, full_sr, np.nan)
    cpcv = _cpcv_oos(port, ppy) if use_cpcv else None
    if cpcv is not None:
        oos_mean, oos_std = cpcv
    else:
        folds = _folds(port.shape[0], n_folds)
        if folds:
            fold_sr = np.array([_col_sharpes(port[a:b], ppy) for a, b in folds])  # (F, S)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                oos_mean = np.nanmean(fold_sr, axis=0)
                oos_std = np.nanstd(fold_sr, axis=0)
        else:
            oos_mean, oos_std = full_sr, np.zeros_like(full_sr)
    stability = np.where(np.isfinite(oos_std), oos_mean / (1.0 + oos_std), full_sr)
    if common_window:
        stability = np.where(valid_counts >= min_obs, stability, np.nan)

    # Rank by stability, then enrich only the leaders with the heavier composite
    # components (DSR + calendar regularity), and re-rank by composite.
    from rigor.validation.overfit import deflated_sharpe_ratio
    from scipy.stats import kurtosis, skew
    finite = np.flatnonzero(np.isfinite(stability))
    order = finite[np.argsort(stability[finite])[::-1]][: max(top_k * 3, top_k)]
    n_trials = int(samples.shape[0])
    rows = []
    for k in order:
        col = port[:, k]
        idx = index
        if common_window:
            keep = ~np.isnan(col)
            col, idx = col[keep], index[keep]
        if len(col) < 2:
            continue
        pos_m, pos_y = _regularity(col, pd.DatetimeIndex(idx))
        sharpe_pp = full_sr[k] / np.sqrt(ppy) if np.isfinite(full_sr[k]) else 0.0
        try:
            dsr = float(deflated_sharpe_ratio(
                sharpe_pp, n_trials, len(col),
                float(skew(col, bias=False)), float(kurtosis(col, fisher=True, bias=False))))
        except Exception:  # noqa: BLE001
            dsr = 0.0
        composite = float(stability[k]) * max(dsr, 0.3) * max(pos_m, 1e-6) * max(pos_y, 1e-6)
        w = samples[k]
        rows.append({
            "composite_score": composite, "score": float(stability[k]),
            "dsr": dsr, "pos_months_pct": pos_m, "yearly_consistency": pos_y,
            "sharpe_oos_mean": float(oos_mean[k]), "sharpe_oos_std": float(oos_std[k]),
            "sharpe_full": float(full_sr[k]), "n_obs": int(len(col)),
            "cagr": float(_m.compute_cagr(col, ppy)),
            "max_drawdown": float(_m.compute_max_drawdown(col)),
            "weights": {names[j]: round(float(w[j]), 4) for j in range(n) if w[j] > 1e-6},
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def weight_grid_search(
    returns_matrix: pd.DataFrame, *, n_samples: int = 5000, top_k: int = 50,
    n_folds: int = 10, max_weight: float | None = None, seed: int = 42,
    use_cpcv: bool = False,
) -> pd.DataFrame:
    """Dense search over the whole simplex (every selected strategy gets weight);
    return the top-K ranked by the **composite anti-overfit score**::

        composite = stability · max(DSR, 0.3) · pos_months · pos_years

    OOS stability is rolling walk-forward (or CPCV when ``use_cpcv=True``). For a
    *focused* book (a handful of strategies) use :func:`subset_grid_search` instead.
    """
    rm = returns_matrix.sort_index()              # outer-join (union of dates)
    n = rm.shape[1]
    if n == 0 or len(rm) < 60:
        return pd.DataFrame()
    R = rm.to_numpy(dtype="float64")
    mask = ~np.isnan(R)
    filled = np.where(mask, R, 0.0)
    ppy = _m.periods_per_year_of(rm.index)

    samples = _samples(n, n_samples, seed)
    if max_weight and max_weight < 1.0:
        samples = np.minimum(samples, max_weight)
        samples = samples / samples.sum(axis=1, keepdims=True)
    samples = np.unique(np.round(samples, 4), axis=0)
    return _rank_samples(samples, filled, mask, list(rm.columns), rm.index, ppy,
                         n_folds=n_folds, use_cpcv=use_cpcv, top_k=top_k)


def _strategy_sharpes(filled: np.ndarray, mask: np.ndarray, ppy: float) -> np.ndarray:
    """Per-strategy annualised Sharpe over each column's own active bars."""
    n_active = mask.sum(axis=0)
    total = filled.sum(axis=0)
    mean = np.where(n_active > 0, total / np.maximum(n_active, 1), 0.0)
    sq = (filled * filled).sum(axis=0)            # filled is 0 where inactive
    var = np.where(n_active > 1, (sq - n_active * mean * mean) / np.maximum(n_active - 1, 1), 0.0)
    sd = np.sqrt(np.clip(var, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mean / sd * np.sqrt(ppy), 0.0)


def _greedy_chain(filled: np.ndarray, mask: np.ndarray, ppy: float,
                  lo: int, hi: int, *, common_window: bool = False) -> np.ndarray:
    """Greedy forward selection: at each step add the leg that most improves the
    equal-weight Sharpe. Returns one equal-weight vector per cardinality in [lo, hi]
    — strong, deterministic seeds for the random subset search."""
    m = filled.shape[1]
    if m == 0:
        return np.zeros((0, m))
    chosen: list[int] = []
    remaining = list(range(m))
    seeds = []
    for _ in range(hi):
        batch = np.zeros((len(remaining), m))
        for bi, r in enumerate(remaining):
            sub = [*chosen, r]
            batch[bi, sub] = 1.0 / len(sub)
        sr = _col_sharpes(_portfolio_matrix(filled, mask, batch, common_window=common_window), ppy)
        if not np.isfinite(sr).any():
            break
        best = int(np.nanargmax(sr))
        chosen.append(remaining.pop(best))
        if len(chosen) >= lo:
            v = np.zeros(m)
            v[chosen] = 1.0 / len(chosen)
            seeds.append(v)
    return np.array(seeds) if seeds else np.zeros((0, m))


def subset_grid_search(
    returns_matrix: pd.DataFrame, *, min_k: int = 1, max_k: int = 10,
    n_samples: int = 4000, top_k: int = 120, n_folds: int = 10,
    max_weight: float | None = None, use_cpcv: bool = False, seed: int = 42,
    pool: int = 120, common_window: bool = False,
) -> pd.DataFrame:
    """Find the best **k-of-N** combinations: each candidate holds only ``k`` legs
    (``k`` in ``[min_k, max_k]``), not the whole book.

    To stay tractable over a large book, the search runs within a **quality pool**
    of the top ``pool`` strategies by individual Sharpe, then samples subsets biased
    toward the higher-Sharpe legs (plus greedy forward-selection seeds), allocates
    weight within each subset (Dirichlet variety, capped by ``max_weight``), and
    ranks them with the same composite anti-overfit score as the dense search.
    Cardinality is therefore *structural* — every candidate has between ``min_k`` and
    ``max_k`` strategies. With ``common_window=True`` each candidate is scored only on
    the bars where its own legs all overlap (see :func:`_portfolio_matrix`).
    """
    rm = returns_matrix.sort_index()
    n_all = rm.shape[1]
    if n_all == 0 or len(rm) < 60:
        return pd.DataFrame()
    R = rm.to_numpy(dtype="float64")
    mask_all = ~np.isnan(R)
    filled_all = np.where(mask_all, R, 0.0)
    ppy = _m.periods_per_year_of(rm.index)

    # Quality pool: top strategies by their own Sharpe (focuses the search and keeps
    # the matrix small enough to multiply quickly).
    qual = _strategy_sharpes(filled_all, mask_all, ppy)
    pool_n = min(max(pool, max_k or 1), n_all)
    pool_idx = np.argsort(np.nan_to_num(qual, nan=-1e9))[::-1][:pool_n]
    names = [rm.columns[i] for i in pool_idx]
    filled = filled_all[:, pool_idx]
    mask = mask_all[:, pool_idx]
    qpool = qual[pool_idx]
    m = len(pool_idx)

    lo = max(int(min_k or 1), 1)
    hi = min(int(max_k or m), m)
    lo = min(lo, hi)
    rng = np.random.default_rng(seed)
    p = np.clip(qpool - np.nanmin(qpool) + 0.1, 0.01, None)  # quality-biased selection
    p = p / p.sum()

    draws = []
    for _ in range(n_samples):
        k = int(rng.integers(lo, hi + 1))
        idx = rng.choice(m, size=k, replace=False, p=p)
        vec = np.zeros(m)
        vec[idx] = rng.dirichlet(np.full(k, float(rng.choice([0.4, 1.0, 3.0]))))
        draws.append(vec)
    samples = np.array(draws) if draws else np.zeros((0, m))
    seeds = _greedy_chain(filled, mask, ppy, lo, hi, common_window=common_window)
    if seeds.size:
        samples = np.vstack([samples, seeds]) if samples.size else seeds
    if max_weight and max_weight < 1.0:
        samples = np.minimum(samples, max_weight)
    s = samples.sum(axis=1, keepdims=True)
    samples = np.where(s > 0, samples / np.where(s > 0, s, 1.0), samples)
    samples = np.unique(np.round(samples, 4), axis=0)
    if samples.size == 0:
        return pd.DataFrame()
    return _rank_samples(samples, filled, mask, names, rm.index, ppy,
                         n_folds=n_folds, use_cpcv=use_cpcv, top_k=top_k,
                         common_window=common_window)
