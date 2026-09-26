"""Hermetic unit tests for Chatterjee's ξ in signal_quality.py.

Covers ``xi_ic`` (the tie-corrected ξ_n coefficient) and ``xi_ic_significance``
(its one-sided asymptotic p-value).

The headline property — and the whole reason ξ exists alongside the Spearman
rank-IC — is that ξ detects ANY dependence, including the NON-MONOTONIC kind
that Spearman is structurally blind to (e.g. Y = X²). The key test asserts
exactly that: ``rank_ic`` ≈ 0 while ``xi_ic`` is clearly positive.

Strong, seeded assertions:
  * independence -> ξ ≈ 0;
  * a monotonic functional relation -> ξ large;
  * a non-monotonic relation -> Spearman blind (≈ 0) but ξ clearly positive;
  * edge cases (< 5 obs, NaNs, constant Y) are guarded without raising;
  * the same input twice gives an identical result (determinism);
  * the significance p-value is small for a real relation, large under
    independence, and always carries the three documented keys.

Uses only numpy / pandas / pytest — no heavy optional dependencies.
"""
from __future__ import annotations

import math

import numpy as np

from rigor.analysis.signal_quality import rank_ic, xi_ic, xi_ic_significance

# ---------------------------------------------------------------------------
# xi_ic
# ---------------------------------------------------------------------------


def test_xi_independence_is_near_zero() -> None:
    """Two iid random arrays are independent -> ξ ≈ 0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    xi = xi_ic(x, y)
    assert abs(xi) < 0.15


def test_xi_monotonic_functional_is_large() -> None:
    """Y a strictly increasing (near-noiseless) function of X -> ξ large."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = x + rng.normal(scale=0.001, size=500)  # essentially Y = X
    xi = xi_ic(x, y)
    assert xi > 0.7


def test_xi_sees_nonmonotonic_dependence_spearman_misses() -> None:
    """THE KEY TEST: Y = X² is fully dependent on X but Spearman is blind.

    Spearman rank-IC ≈ 0 (a symmetric U-shape has no monotone trend), yet ξ
    sees the dependence clearly. This is exactly the gap ξ closes.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)  # symmetric around 0
    y = x**2                   # non-monotonic, deterministic function of X

    spearman = rank_ic(x, y)
    xi = xi_ic(x, y)

    assert abs(spearman) < 0.15   # Spearman cannot see it
    assert xi > 0.4               # ξ clearly does


def test_xi_short_input_returns_zero() -> None:
    """Fewer than 5 valid pairs -> 0.0 (mirrors rank_ic's guard), no raise."""
    assert xi_ic([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == 0.0
    assert xi_ic(np.arange(4.0), np.arange(4.0)) == 0.0


def test_xi_handles_nan_pairs() -> None:
    """Non-finite pairs are masked out; the finite remainder still scores."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = x + rng.normal(scale=0.001, size=400)
    x[:40] = np.nan
    y[-40:] = np.inf
    xi = xi_ic(x, y)
    assert math.isfinite(xi)
    assert xi > 0.5  # the ~320 finite pairs are still near-functional


def test_xi_constant_y_returns_zero_no_division_error() -> None:
    """A constant Y has a zero denominator -> guarded 0.0, not a crash."""
    x = np.arange(50.0)
    y = np.full(50, 7.0)
    assert xi_ic(x, y) == 0.0


def test_xi_is_deterministic() -> None:
    """Identical input twice -> bit-identical output."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=300)
    y = np.abs(x) + rng.normal(scale=0.01, size=300)
    assert xi_ic(x, y) == xi_ic(x, y)


# ---------------------------------------------------------------------------
# xi_ic_significance
# ---------------------------------------------------------------------------


def test_xi_significance_small_p_for_functional_relation() -> None:
    """A real (here non-monotonic) relation is flagged significant."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=1000)
    y = x**2
    res = xi_ic_significance(x, y)
    assert set(res) == {"xi", "p_value", "n"}
    assert res["n"] == 1000
    assert res["xi"] > 0.4
    assert res["p_value"] < 0.01


def test_xi_significance_large_p_under_independence() -> None:
    """Independent arrays are not flagged significant (large p-value)."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    res = xi_ic_significance(x, y)
    assert set(res) == {"xi", "p_value", "n"}
    assert res["p_value"] > 0.2


def test_xi_significance_short_input_guarded() -> None:
    """n < 5 -> ξ 0.0, p 1.0, and the valid-pair count."""
    res = xi_ic_significance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res == {"xi": 0.0, "p_value": 1.0, "n": 3}
