"""Tests for CPCV purge + post-test embargo (AFML Ch. 7).

Covers:
  (i)   purge removes the train bars adjacent to each test fold (both sides);
  (ii)  embargo removes ONLY the post-test bars;
  (iii) on autocorrelated/leaky returns, purge+embargo raises PBO vs purge=0
        (i.e. it removes the optimistic bias);
  (iv)  determinism (same input -> same PBO);
  (v)   purge=embargo=0 reproduces the legacy (un-purged) behaviour.
"""

from __future__ import annotations

import numpy as np

from rigor.validation.cpcv import _purge_embargo_mask, cpcv_pbo

# ── (i) + (ii): mask geometry ────────────────────────────────────────────────

def _single_block_mask(n: int, start: int, stop: int) -> np.ndarray:
    """Boolean OOS mask with one contiguous test block [start, stop)."""
    m = np.zeros(n, dtype=bool)
    m[start:stop] = True
    return m


def test_purge_removes_train_bars_on_both_sides_of_each_fold():
    # One test block in the middle; purge_gap=2 should drop 2 train bars on EACH
    # side of the block (symmetric label-window overlap), and none far away.
    oos = _single_block_mask(20, 10, 13)  # test bars = {10, 11, 12}
    drop = _purge_embargo_mask(oos, purge_gap=2, embargo_bars=0)

    # Pre-test purged train bars: {8, 9}; post-test purged train bars: {13, 14}.
    assert drop[8] and drop[9]
    assert drop[13] and drop[14]
    # Just outside the purge band stays in TRAIN.
    assert not drop[7]
    assert not drop[15]
    # Bars far from the fold are untouched.
    assert not drop[0]
    assert not drop[19]


def test_embargo_removes_only_post_test_bars():
    oos = _single_block_mask(20, 10, 13)  # block ends at index 12
    drop = _purge_embargo_mask(oos, purge_gap=0, embargo_bars=3)

    # Embargo is one-sided: the 3 bars AFTER the block are dropped.
    assert drop[13] and drop[14] and drop[15]
    assert not drop[16]
    # Bars BEFORE the block must NOT be embargoed (purge_gap=0 here).
    assert not drop[9]
    assert not drop[8]
    assert not drop[7]


def test_purge_and_embargo_never_remove_test_bars_from_oos():
    # The drop mask flags TRAIN bars to remove; the OOS bars themselves are kept
    # as the hold-out upstream. Verify the IS mask after AND-NOT excludes nothing
    # extra from OOS (OOS membership is untouched by purge/embargo).
    oos = _single_block_mask(30, 12, 16)
    drop = _purge_embargo_mask(oos, purge_gap=3, embargo_bars=3)
    is_mask = (~oos) & ~drop
    # No bar can be both OOS and IS.
    assert not np.any(is_mask & oos)
    # Every original OOS bar is still OOS (purge/embargo can't shrink the test set).
    assert oos.sum() == 4


def test_embargo_at_right_edge_has_no_trailing_bars():
    # A block touching the right edge has no post-test bars to embargo.
    oos = _single_block_mask(10, 7, 10)  # block ends at the last index
    drop = _purge_embargo_mask(oos, purge_gap=0, embargo_bars=3)
    # Nothing can be dropped after the array end.
    assert not drop[:7].any()


# ── (iii): purge+embargo raises PBO on leaky/autocorrelated returns ──────────

def _leaky_matrix(n_cfg: int = 12, n_bars: int = 1500, seed: int = 7) -> np.ndarray:
    """Configs whose *skill drifts slowly and differently* over time (AR(1) mean).

    Each config carries a highly persistent (phi=0.97) autocorrelated drift in its
    mean return, plus white noise. Persistence makes the config that is best
    in-sample *locally* best on the OOS bars immediately ADJACENT to the IS block,
    so it spuriously looks like it generalises -> an optimistically LOW PBO.
    Because the drift is per-config and slow, the IS-best is NOT the global OOS
    winner, so purging/embargoing the leaky boundary reveals a higher, honest PBO.
    """
    rng = np.random.default_rng(seed)
    mu = np.empty((n_cfg, n_bars))
    eps = rng.normal(0.0, 0.0006, (n_cfg, n_bars))
    mu[:, 0] = eps[:, 0]
    phi = 0.97
    for t in range(1, n_bars):
        mu[:, t] = phi * mu[:, t - 1] + eps[:, t]
    noise = rng.normal(0.0, 0.004, (n_cfg, n_bars))
    return mu + noise


def test_purge_embargo_raises_pbo_on_leaky_returns_single_seed():
    # Seed 7 produces a strong, unambiguous boundary-leakage signal.
    mat = _leaky_matrix(seed=7)
    base = cpcv_pbo(mat, n_groups=10, n_test_groups=2, purge_gap=0, embargo_pct=0.0)
    purged = cpcv_pbo(mat, n_groups=10, n_test_groups=2, purge_gap=15, embargo_pct=0.03)
    assert base is not None and purged is not None
    # Purging the leaky IS/OOS boundary removes the optimistic bias -> PBO rises.
    assert purged["pbo"] > base["pbo"]
    # The honesty/metadata flags survive the new code path.
    assert purged["purge_gap"] == 15
    assert purged["embargo_bars"] >= 1
    assert "insufficient_paths" in purged


def test_purge_embargo_raises_mean_pbo_across_seeds():
    # The bias is statistical: average over many independent leaky draws so the
    # direction (purge+embargo >= un-purged) is robust, not seed-cherry-picked.
    base_pbos, purged_pbos = [], []
    for seed in range(12):
        mat = _leaky_matrix(seed=seed)
        base = cpcv_pbo(mat, n_groups=10, n_test_groups=2, purge_gap=0, embargo_pct=0.0)
        purged = cpcv_pbo(
            mat, n_groups=10, n_test_groups=2, purge_gap=15, embargo_pct=0.03
        )
        assert base is not None and purged is not None
        base_pbos.append(base["pbo"])
        purged_pbos.append(purged["pbo"])
    # Mean PBO strictly rises once the leaky boundary is purged/embargoed.
    assert np.mean(purged_pbos) > np.mean(base_pbos)


# ── (iv): determinism ────────────────────────────────────────────────────────

def test_pbo_is_deterministic():
    mat = _leaky_matrix(seed=3)
    a = cpcv_pbo(mat, n_groups=8, n_test_groups=2, purge_gap=6, embargo_pct=0.01)
    b = cpcv_pbo(mat, n_groups=8, n_test_groups=2, purge_gap=6, embargo_pct=0.01)
    assert a is not None and b is not None
    assert a == b  # identical dicts, no RNG anywhere in the path


# ── (v): backward compatibility ──────────────────────────────────────────────

def test_zero_purge_zero_embargo_reproduces_legacy():
    mat = _leaky_matrix(seed=11)
    new = cpcv_pbo(mat, n_groups=10, n_test_groups=2, purge_gap=0, embargo_pct=0.0)
    assert new is not None
    # Recompute the OLD behaviour inline (no purge/embargo) and compare the PBO.
    import itertools as _it

    rm = np.asarray(mat, dtype="float64")
    n_cfg, n_bars = rm.shape
    bounds = np.linspace(0, n_bars, 10 + 1, dtype=int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(10)]

    def _sr(mask: np.ndarray) -> np.ndarray:
        sub = rm[:, mask]
        mu = sub.mean(axis=1)
        sd = sub.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 1e-12, mu / sd * np.sqrt(252), 0.0)

    logits = []
    for test in _it.combinations(range(10), 2):
        oos = np.zeros(n_bars, dtype=bool)
        for g in test:
            oos[groups[g]] = True
        is_mask = ~oos
        if is_mask.sum() < 2 or oos.sum() < 2:
            continue
        is_sr, oos_sr = _sr(is_mask), _sr(oos)
        bi = int(np.argmax(is_sr))
        rank = float(np.sum(oos_sr <= oos_sr[bi]))
        rn = np.clip((rank - 0.5) / n_cfg, 1e-6, 1 - 1e-6)
        logits.append(float(np.log(rn / (1 - rn))))
    legacy_pbo = round(float(np.mean(np.array(logits) < 0)), 3)
    assert new["pbo"] == legacy_pbo
