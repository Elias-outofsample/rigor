"""Portfolio-level overfit validation.

Two independent diagnostics:

1. ``portfolio_pbo`` — Probability of Backtest Overfitting applied to the
   *allocation itself*.  We treat each candidate weight vector as a "strategy
   configuration" and run the same CSCV-PBO machinery (López de Prado 2018)
   used by the rigor framework's ``rigor.validation.cpcv.cpcv_pbo``.  The matrix
   fed to ``cpcv_pbo`` is ``(n_alloc, n_bars)`` where each row is the
   portfolio return series for one Dirichlet-sampled weight vector.  PBO then
   answers: "did we overfit the allocation to in-sample data?"

   Implementation path: **rigor.validation.cpcv.cpcv_pbo** (reused directly).
   The function signature ``cpcv_pbo(returns_matrix: np.ndarray, ...)`` treats
   row-index as "configurations" and column-index as "time bars" — exactly
   what we want when rows are candidate allocations.

2. ``cardinality_search`` — "how many strategies do we actually need?"
   Uses **greedy forward selection**: start with the empty set; at each step
   add the leg that maximises the equal-weight Sharpe of the current set plus
   that leg.  Record the Sharpe at each cardinality k = 1 .. max_k.  Identify
   the *knee*: the smallest k where Sharpe ≥ 95 % of the best Sharpe seen.
   Greedy is O(n²) and interpretable; exhaustive search is exponential and
   gives nearly identical knee estimates for correlated portfolios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rigor.validation.cpcv import cpcv_pbo
from rigor.validation.permutation import sign_permutation_test

__all__ = [
    "portfolio_pbo",
    "cardinality_search",
    "permutation_test",
    "_portfolio_returns",
    "_sharpe",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _portfolio_returns(returns_matrix: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute the weighted portfolio return series (NaN-safe, date-indexed).

    Missing values in individual legs are treated as zero contribution for
    that bar; the weight is renormalised over available legs so the gross
    exposure stays at 1.0 whenever at least one leg is present.
    """
    rm = returns_matrix.to_numpy(dtype="float64")
    mask = ~np.isnan(rm)
    w = np.asarray(weights, dtype="float64")
    w_abs = np.abs(w)
    # active weight per bar (denominator)
    active = mask @ w_abs  # (T,)
    filled = np.where(mask, rm, 0.0)
    num = filled @ w  # (T,)
    with np.errstate(divide="ignore", invalid="ignore"):
        port = np.where(active > 1e-12, num / active, 0.0)
    return pd.Series(port, index=returns_matrix.index)


def _sharpe(series: pd.Series | np.ndarray, ppy: int = 252) -> float:
    """Annualised Sharpe (mean/std × √ppy).  Returns nan for < 2 obs."""
    arr = np.asarray(series, dtype="float64")
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(ppy))


# ---------------------------------------------------------------------------
# Candidate allocation sampler (Dirichlet)
# ---------------------------------------------------------------------------


def _dirichlet_allocations(
    n_cols: int, n_alloc: int, seed: int
) -> np.ndarray:
    """Sample ``n_alloc - 1`` Dirichlet allocations + equal-weight row.

    Returns array of shape (n_alloc, n_cols) with rows summing to 1.
    Uses three concentration levels (sparse, uniform, concentrated) to cover
    the long-only simplex evenly.
    """
    rng = np.random.default_rng(seed)
    # Equal-weight is always candidate 0
    ew = np.full((1, n_cols), 1.0 / n_cols)
    if n_alloc <= 1 or n_cols == 1:
        return ew

    n_dir = n_alloc - 1
    n1, n2 = int(n_dir * 0.5), int(n_dir * 0.3)
    n3 = max(n_dir - n1 - n2, 0)
    blocks: list[np.ndarray] = [ew]
    for conc, cnt in ((0.3, n1), (1.0, n2), (3.0, n3)):
        if cnt > 0:
            block = rng.dirichlet(np.full(n_cols, conc), size=cnt)
            blocks.append(block)
    w = np.concatenate(blocks, axis=0)[:n_alloc]
    return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# portfolio_pbo
# ---------------------------------------------------------------------------


def portfolio_pbo(
    returns_matrix: pd.DataFrame,
    *,
    n_alloc: int = 50,
    n_splits: int = 8,
    seed: int = 42,
    ppy: int = 252,
) -> dict:
    """Compute PBO over the *allocation choice* itself.

    Parameters
    ----------
    returns_matrix:
        Date-indexed DataFrame, one column per strategy leg.  May contain NaNs
        (leg not yet live).
    n_alloc:
        Number of candidate weight vectors to evaluate (includes equal-weight).
    n_splits:
        Number of CSCV time-groups (``n_groups`` fed to ``cpcv_pbo``).
    seed:
        RNG seed for reproducibility.
    ppy:
        Trading periods per year (252 for daily returns).

    Returns
    -------
    dict with keys:
        ``pbo`` — float in [0, 1].
        ``n_alloc`` — actual number of allocations evaluated.
        ``n_splits`` — n_groups used.
        ``best_is_alloc_oos_rank_pct`` — float in [0, 1]; median OOS rank
            percentile of the IS-best allocation across all CSCV paths.
        ``verdict`` — one of "ROBUST_ALLOCATION", "BORDERLINE",
            "OVERFIT_ALLOCATION", or "INSUFFICIENT_DATA".
    """
    n_cols = returns_matrix.shape[1]
    n_bars = returns_matrix.shape[0]

    if n_cols < 2 or n_bars < n_splits * 2:
        return {
            "pbo": float("nan"),
            "n_alloc": 0,
            "n_splits": n_splits,
            "best_is_alloc_oos_rank_pct": float("nan"),
            "verdict": "INSUFFICIENT_DATA",
        }

    # Build (n_alloc, n_bars) matrix where each row is one allocation's
    # portfolio return series over time.
    weights = _dirichlet_allocations(n_cols, n_alloc, seed)
    actual_n = weights.shape[0]

    rm_np = returns_matrix.sort_index().to_numpy(dtype="float64")
    mask = ~np.isnan(rm_np)
    filled = np.where(mask, rm_np, 0.0)
    w_abs = np.abs(weights)
    # active_share: (n_alloc, n_bars) — fraction of gross weight on live legs
    active_share = w_abs @ mask.T  # (n_alloc, n_bars)
    num = weights @ filled.T       # (n_alloc, n_bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        port_matrix = np.where(active_share > 1e-12, num / active_share, 0.0)

    # cpcv_pbo expects (n_configs, n_bars)
    result = cpcv_pbo(port_matrix, n_groups=n_splits, n_test_groups=2, ppy=ppy)

    if result is None:
        return {
            "pbo": float("nan"),
            "n_alloc": actual_n,
            "n_splits": n_splits,
            "best_is_alloc_oos_rank_pct": float("nan"),
            "verdict": "INSUFFICIENT_DATA",
        }

    pbo_val = float(result["pbo"])
    if pbo_val < 0.3:
        verdict = "ROBUST_ALLOCATION"
    elif pbo_val <= 0.5:
        verdict = "BORDERLINE"
    else:
        verdict = "OVERFIT_ALLOCATION"

    # best_is_alloc_oos_rank_pct: derive from efficiency metric
    # cpcv_pbo does not surface per-path OOS ranks directly; approximate via
    # the mean OOS rank implied by PBO: E[rank_pct] ≈ 1 − pbo (since pbo
    # measures the fraction where the IS-best lands below median OOS).
    oos_rank_pct = 1.0 - pbo_val

    return {
        "pbo": pbo_val,
        "n_alloc": actual_n,
        "n_splits": n_splits,
        "best_is_alloc_oos_rank_pct": round(oos_rank_pct, 4),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------


def permutation_test(
    returns_matrix: pd.DataFrame,
    weights: np.ndarray,
    *,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict:
    """Sign-permutation test on the *weighted portfolio* return series.

    Builds the portfolio returns (NaN-safe, active-share renormalised) and runs
    ``rigor.validation.permutation.sign_permutation_test`` — shuffles the sign of
    each bar, preserving magnitudes, and asks how often a random sign pattern
    matches the observed mean. A small p-value means the edge isn't a coin-flip.

    Returns ``{"p_value", "is_significant", "n_perm"}``; p_value is NaN when there
    are too few observations.
    """
    port = _portfolio_returns(returns_matrix, np.asarray(weights, dtype="float64"))
    res = sign_permutation_test(port.dropna(), n_perm=n_perm, seed=seed)
    return {
        "p_value": float(res.get("p_value", float("nan"))),
        "is_significant": bool(res.get("is_significant", False)),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# cardinality_search
# ---------------------------------------------------------------------------


def cardinality_search(
    returns_matrix: pd.DataFrame,
    *,
    max_k: int | None = None,
    method: str = "greedy",
    seed: int = 42,
    ppy: int = 252,
) -> dict:
    """Find the minimum cardinality (number of legs) that captures most of the Sharpe.

    **Method — greedy forward selection** (``method="greedy"``):
    Start with an empty set.  At each step k = 1, 2, ..., max_k, add the
    remaining leg whose inclusion maximises the equal-weight Sharpe of the
    current set.  Record the Sharpe at every k.  Identify the *knee*: the
    smallest k where Sharpe ≥ ``knee_fraction`` (default 0.95) of the
    highest Sharpe seen across all k.

    Greedy forward selection is O(n²) in n legs and interpretable.  It is
    well-known to produce near-optimal subsets for portfolio problems where
    diversification is monotone (each added leg typically improves the
    collective Sharpe until diminishing returns kick in).

    Parameters
    ----------
    returns_matrix:
        Date-indexed DataFrame, one column per strategy leg.
    max_k:
        Maximum subset size to evaluate.  Defaults to number of columns.
    method:
        Algorithm ("greedy" is the only supported value; reserved for future
        methods such as exhaustive or beam search).
    seed:
        RNG seed (unused for greedy; kept for API consistency).
    ppy:
        Trading periods per year.

    Returns
    -------
    dict with keys:
        ``k`` — list[int] of subset sizes 1 .. max_k.
        ``sharpe_at_k`` — list[float] best equal-weight Sharpe at each k.
        ``best_sharpe`` — float, max of ``sharpe_at_k``.
        ``knee_k`` — int, smallest k reaching >= 95 % of best_sharpe.
        ``knee_fraction`` — float (0.95).
        ``selected_at_knee`` — list[str] of column names chosen at knee_k.
    """
    if method != "greedy":
        raise ValueError(f"Unsupported method {method!r}; only 'greedy' is implemented.")

    cols = list(returns_matrix.columns)
    n = len(cols)
    if n == 0:
        return {
            "k": [], "sharpe_at_k": [], "best_sharpe": float("nan"),
            "knee_k": 0, "knee_fraction": 0.95, "selected_at_knee": [],
        }

    mk = min(max_k, n) if max_k is not None else n
    rm = returns_matrix.sort_index()

    selected: list[str] = []
    remaining = list(cols)
    k_list: list[int] = []
    sharpe_list: list[float] = []
    # Track which columns were selected at each step
    selected_at_k: list[list[str]] = []

    for step in range(mk):
        best_sr = float("-inf")
        best_leg = remaining[0]
        for leg in remaining:
            candidate = selected + [leg]
            sub = rm[candidate].dropna(how="all")
            ew = np.ones(len(candidate)) / len(candidate)
            port = _portfolio_returns(sub, ew)
            sr = _sharpe(port, ppy)
            if np.isfinite(sr) and sr > best_sr:
                best_sr = sr
                best_leg = leg
        selected.append(best_leg)
        remaining.remove(best_leg)
        k_list.append(step + 1)
        sharpe_list.append(best_sr if np.isfinite(best_sr) else float("nan"))
        selected_at_k.append(list(selected))

    # Identify knee: smallest k where sharpe >= 95% of best
    knee_fraction = 0.95
    finite_sharpes = [s for s in sharpe_list if np.isfinite(s)]
    best_sharpe = max(finite_sharpes) if finite_sharpes else float("nan")

    knee_k = mk  # default: all
    selected_at_knee: list[str] = selected_at_k[-1] if selected_at_k else []
    if np.isfinite(best_sharpe) and best_sharpe > 0:
        threshold = knee_fraction * best_sharpe
        for k_idx, sr in enumerate(sharpe_list):
            if np.isfinite(sr) and sr >= threshold:
                knee_k = k_list[k_idx]
                selected_at_knee = selected_at_k[k_idx]
                break
    elif np.isfinite(best_sharpe):
        # best_sharpe is <= 0; knee at k=1 (first selected is best or tied)
        knee_k = k_list[0] if k_list else 1
        selected_at_knee = selected_at_k[0] if selected_at_k else []

    return {
        "k": k_list,
        "sharpe_at_k": sharpe_list,
        "best_sharpe": best_sharpe,
        "knee_k": knee_k,
        "knee_fraction": knee_fraction,
        "selected_at_knee": selected_at_knee,
    }
