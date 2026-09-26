"""Combinatorial Purged Cross-Validation -> PBO (Lopez de Prado 2018).

Deterministic, pure-numpy port (the secondary used numba ``prange``, which is
non-deterministic across machines). Needs a returns matrix across parameter
combos, so it only applies to strategies that expose a multi-value ``param_grid``.

PBO (Probability of Backtest Overfitting): across all combinatorial IS/OOS
splits, take the IS-best config, find its OOS rank, logit-transform it; PBO is
the fraction of splits where the IS-best config lands in the bottom half OOS.

Purge vs embargo (AFML Ch. 7)
-----------------------------
Adjacent IS/OOS bars are serially correlated (overlapping label windows,
autocorrelated returns), which leaks OOS information into the IS fit and biases
PBO *optimistically low*. Two complementary, **train-only** filters remove that
leakage — both drop observations from the TRAIN (IS) side and NEVER touch TEST
(OOS), so the OOS evaluation stays an honest hold-out:

* **PURGE** — drop train bars whose label/feature window overlaps a test block.
  A label realised at bar ``t`` spans roughly ``[t - h, t + h]`` (look-back
  features and a forward holding horizon ``h``), so a train bar within ``h`` of a
  test block on *either* side shares information with it. This is the existing
  symmetric dilation of width ``2*purge_gap + 1`` and removes the pre- and
  post-test overlap caused by the label window itself.

* **EMBARGO** — additionally drop a small one-sided band of train bars
  *immediately AFTER* each test block. Even once the label windows no longer
  overlap, residual serial correlation makes a train bar just after a test block
  partially predictable from it. AFML embargoes a fraction of the sample
  (``embargo_pct``, commonly ~1%) on the trailing edge only. Purge already covers
  the leading (pre-test) edge, so the embargo is deliberately post-test only.

Both filters can only *raise* PBO toward its honest value (they shrink the IS
sample the best config is selected on); ``purge_gap = 0`` and ``embargo_pct = 0``
reproduce the original, un-purged behaviour exactly.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .. import metrics as _m


def _sharpes(returns_matrix: np.ndarray, mask: np.ndarray, ppy: int) -> np.ndarray:
    """Per-config annualised Sharpe over the masked bars."""
    sub = returns_matrix[:, mask]
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mu / sd * np.sqrt(ppy), 0.0)


def _purge_embargo_mask(
    oos_mask: np.ndarray, purge_gap: int, embargo_bars: int
) -> np.ndarray:
    """Train bars to DROP around the OOS blocks (purge + post-test embargo).

    Returns a boolean array (same length as ``oos_mask``) that is ``True`` for the
    train bars to remove. Test bars are *included* in the returned mask (they are
    already excluded from train upstream); the caller AND-NOTs this against the
    train mask, so flagging them is harmless and keeps the logic simple.

    * **purge**   — symmetric dilation of width ``2*purge_gap + 1`` around every
      OOS bar (the label window overlaps on both sides; see module docstring).
    * **embargo** — one-sided: the ``embargo_bars`` train bars immediately AFTER
      the end of each contiguous OOS block.
    """
    drop = np.zeros_like(oos_mask)
    if purge_gap > 0:  # symmetric purge: overlap of the label window, both sides
        drop |= np.convolve(oos_mask, np.ones(2 * purge_gap + 1), mode="same") > 0
    if embargo_bars > 0:  # one-sided post-test embargo: bars just AFTER each block
        # A block ends at index i when oos_mask[i] and not oos_mask[i+1]. Mark the
        # following ``embargo_bars`` indices. Done with an index shift so it is
        # deterministic and vectorised (no Python loop over blocks).
        ends = np.zeros_like(oos_mask)
        ends[:-1] = oos_mask[:-1] & ~oos_mask[1:]
        if oos_mask[-1]:  # a block touching the right edge has no trailing bars
            ends[-1] = False
        emb = np.zeros_like(oos_mask)
        end_idx = np.flatnonzero(ends)
        for k in range(1, embargo_bars + 1):
            tgt = end_idx + k
            tgt = tgt[tgt < oos_mask.shape[0]]
            emb[tgt] = True
        drop |= emb
    return drop


def cpcv_pbo(
    returns_matrix: np.ndarray, *, n_groups: int = 10, n_test_groups: int = 2,
    purge_gap: int = 0, embargo_pct: float = 0.0, embargo_gap: int | None = None,
    ppy: int = _m.TRADING_DAYS,
) -> dict | None:
    """Compute PBO from a (n_combos, n_bars) returns matrix. None if not enough combos.

    Parameters
    ----------
    returns_matrix:
        ``(n_configs, n_bars)`` per-config bar returns (oldest-first columns).
    n_groups, n_test_groups:
        CPCV partition: ``n_groups`` contiguous blocks, ``n_test_groups`` held out
        OOS per split (all combinations enumerated).
    purge_gap:
        Half-width (in bars) of the **symmetric purge** around each OOS block —
        the label/feature horizon ``h``. ``0`` disables purging (legacy behaviour).
    embargo_pct:
        **Post-test embargo** as a fraction of ``n_bars`` (AFML default ~0.01).
        The number of trailing train bars dropped after each OOS block is
        ``ceil(embargo_pct * n_bars)``. Ignored when ``embargo_gap`` is given.
    embargo_gap:
        Explicit post-test embargo width in bars; overrides ``embargo_pct`` when
        not ``None``. ``0`` (or ``embargo_pct == 0``) disables the embargo.
    ppy:
        Annualisation factor for the Sharpe (periods per year).

    Notes
    -----
    Purge and embargo only remove TRAIN bars (never TEST), so OOS evaluation is an
    untouched hold-out. ``purge_gap == 0`` and no embargo reproduce the original
    PBO exactly (backward-compatible). See the module docstring for the rationale.
    """
    rm = np.asarray(returns_matrix, dtype="float64")
    if rm.ndim != 2 or rm.shape[0] < 2 or rm.shape[1] < n_groups:
        return None
    n_cfg, n_bars = rm.shape

    purge_gap = max(int(purge_gap), 0)
    if embargo_gap is not None:
        embargo_bars = max(int(embargo_gap), 0)
    else:
        embargo_bars = math.ceil(max(embargo_pct, 0.0) * n_bars) if embargo_pct > 0 else 0

    bounds = np.linspace(0, n_bars, n_groups + 1, dtype=int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    logits, is_best, oos_best, taus = [], [], [], []
    for test in itertools.combinations(range(n_groups), n_test_groups):
        oos_mask = np.zeros(n_bars, dtype=bool)
        for g in test:
            oos_mask[groups[g]] = True
        is_mask = ~oos_mask
        if purge_gap > 0 or embargo_bars > 0:  # drop leaky IS bars around each OOS block
            is_mask &= ~_purge_embargo_mask(oos_mask, purge_gap, embargo_bars)
        if is_mask.sum() < 2 or oos_mask.sum() < 2:
            continue
        is_sr = _sharpes(rm, is_mask, ppy)
        oos_sr = _sharpes(rm, oos_mask, ppy)
        b = int(np.argmax(is_sr))
        rank = float(np.sum(oos_sr <= oos_sr[b]))  # 1..n_cfg
        rn = np.clip((rank - 0.5) / n_cfg, 1e-6, 1 - 1e-6)
        logits.append(float(np.log(rn / (1 - rn))))
        is_best.append(float(is_sr[b]))
        oos_best.append(float(oos_sr[b]))
        taus.append(_kendall_tau(is_sr, oos_sr))

    if not logits:
        return None
    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0))
    mean_is = float(np.mean(is_best))
    n_paths = len(logits_arr)
    half = 1.96 * float(np.sqrt(pbo * (1.0 - pbo) / n_paths)) if n_paths else 0.0
    finite_taus = [t for t in taus if t == t]
    verdict = ("ROBUST" if pbo < 0.40 else "MODERATE" if pbo < 0.55
               else "FRAGILE" if pbo < 0.70 else "OVERFIT")
    return {
        "pbo": round(pbo, 3),
        "mean_oos_sharpe": round(float(np.mean(oos_best)), 3),
        "efficiency": round(float(np.mean(oos_best) / mean_is), 3) if mean_is != 0 else 0.0,
        "rank_correlation": round(float(np.mean(finite_taus)), 3) if finite_taus else 0.0,
        "pbo_se": round(float(np.sqrt(0.25 / n_paths)), 3) if n_paths else 0.0,
        "pbo_ci": [round(max(pbo - half, 0.0), 3), round(min(pbo + half, 1.0), 3)],
        "n_paths": n_paths,
        "n_configs": n_cfg,
        "purge_gap": purge_gap,
        "embargo_bars": embargo_bars,
        "insufficient_paths": n_paths < 5,
        "verdict": verdict,
    }


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall rank correlation of IS vs OOS per-config Sharpe (NaN if degenerate)."""
    from scipy.stats import kendalltau
    if len(a) < 2:
        return float("nan")
    tau, _ = kendalltau(a, b)
    return float(tau)
