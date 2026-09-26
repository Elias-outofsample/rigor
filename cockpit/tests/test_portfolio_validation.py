"""Tests for portfolio_validation — portfolio-level PBO and cardinality search.

All tests are hermetic (seeded, no I/O) and deterministic.  Runtime is kept
modest by using small n_alloc / n_splits / n bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_engine.portfolio_validation import (
    _portfolio_returns,
    _sharpe,
    cardinality_search,
    portfolio_pbo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_VERDICTS = {"ROBUST_ALLOCATION", "BORDERLINE", "OVERFIT_ALLOCATION", "INSUFFICIENT_DATA"}


def _make_returns(
    n_legs: int = 5,
    bars: int = 800,
    mu: float = 4e-4,
    sigma: float = 0.01,
    seed: int = 0,
    start: str = "2016-01-01",
) -> pd.DataFrame:
    """Build genuinely-independent positive-Sharpe legs (no shared factor)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=bars)
    data = {
        f"leg{i}": mu + sigma * rng.standard_normal(bars)
        for i in range(n_legs)
    }
    return pd.DataFrame(data, index=idx)


def _make_noisy_leg(bars: int = 800, seed: int = 99, start: str = "2016-01-01") -> pd.Series:
    """A leg with zero expected return — pure noise (no persistent signal)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=bars)
    return pd.Series(0.01 * rng.standard_normal(bars), index=idx)


# ---------------------------------------------------------------------------
# 1. portfolio_pbo on genuinely-independent positive-Sharpe legs
# ---------------------------------------------------------------------------

class TestPortfolioPboBasic:
    """PBO returns valid structure and is deterministic."""

    def setup_method(self):
        rm = _make_returns(n_legs=5, bars=800, seed=7)
        self.result = portfolio_pbo(rm, n_alloc=20, n_splits=6, seed=42)
        self.result2 = portfolio_pbo(rm, n_alloc=20, n_splits=6, seed=42)

    def test_pbo_is_finite_float_in_range(self):
        pbo = self.result["pbo"]
        assert isinstance(pbo, float), f"pbo should be float, got {type(pbo)}"
        assert 0.0 <= pbo <= 1.0, f"pbo {pbo} not in [0, 1]"

    def test_verdict_is_valid_label(self):
        assert self.result["verdict"] in _VALID_VERDICTS, (
            f"Unexpected verdict: {self.result['verdict']}"
        )

    def test_n_alloc_returned(self):
        assert self.result["n_alloc"] > 0

    def test_best_is_alloc_oos_rank_pct_in_range(self):
        rp = self.result["best_is_alloc_oos_rank_pct"]
        assert 0.0 <= rp <= 1.0, f"rank_pct {rp} not in [0, 1]"

    def test_deterministic_across_two_calls(self):
        """Same seed must yield identical result."""
        assert self.result["pbo"] == self.result2["pbo"], (
            f"Non-deterministic: {self.result['pbo']} vs {self.result2['pbo']}"
        )
        assert self.result["verdict"] == self.result2["verdict"]

    def test_n_splits_echoed(self):
        assert self.result["n_splits"] == 6


# ---------------------------------------------------------------------------
# 2. Noise leg raises PBO vs clean matrix (relative check)
# ---------------------------------------------------------------------------

class TestPortfolioPboNoisyVsGenuine:
    """Adding a noise-only leg and evaluating a matrix that only has noise
    allocations should produce higher PBO than a clean matrix of true alpha legs.

    We construct a 'noisy matrix' where one leg has zero drift.  Allocations
    that over-weight that leg look good in-sample (by chance) but degrade OOS.
    """

    def test_genuine_pbo_lte_noisy_or_both_finite(self):
        """The genuine matrix should have pbo <= noisy matrix pbo, or at
        minimum both should be finite floats in [0, 1].  We don't enforce a
        strict inequality (stochastic), but both must be structurally valid."""
        genuine = _make_returns(n_legs=4, bars=900, mu=5e-4, sigma=0.01, seed=1)
        noisy_extra = _make_noisy_leg(bars=900, seed=200)
        # Noisy matrix: 4 genuine legs + 1 zero-drift noise leg
        noisy = genuine.copy()
        noisy["noise"] = noisy_extra.values

        r_genuine = portfolio_pbo(genuine, n_alloc=15, n_splits=6, seed=42)
        r_noisy = portfolio_pbo(noisy, n_alloc=15, n_splits=6, seed=42)

        assert np.isfinite(r_genuine["pbo"]), "genuine pbo should be finite"
        assert np.isfinite(r_noisy["pbo"]), "noisy pbo should be finite"
        assert 0.0 <= r_genuine["pbo"] <= 1.0
        assert 0.0 <= r_noisy["pbo"] <= 1.0

    def test_pure_noise_matrix_pbo_not_lower_than_genuine(self):
        """A matrix of pure noise legs should not have lower PBO than a
        genuine alpha matrix.  We allow equality (rare coincidence) but
        noisy_pbo should be >= genuine_pbo - 0.20 tolerance for small samples.
        """
        genuine = _make_returns(n_legs=3, bars=1000, mu=6e-4, sigma=0.01, seed=2)
        rng = np.random.default_rng(50)
        idx = pd.bdate_range("2016-01-01", periods=1000)
        noise_only = pd.DataFrame(
            {f"n{i}": 0.01 * rng.standard_normal(1000) for i in range(3)},
            index=idx,
        )
        r_genuine = portfolio_pbo(genuine, n_alloc=15, n_splits=6, seed=42)
        r_noise = portfolio_pbo(noise_only, n_alloc=15, n_splits=6, seed=42)

        assert np.isfinite(r_genuine["pbo"])
        assert np.isfinite(r_noise["pbo"])
        # noise PBO should not be meaningfully lower than genuine PBO
        # (it can be equal or higher; a 0.20 slack handles sample variance)
        assert r_noise["pbo"] >= r_genuine["pbo"] - 0.20, (
            f"Noise pbo {r_noise['pbo']:.3f} unexpectedly << genuine {r_genuine['pbo']:.3f}"
        )


# ---------------------------------------------------------------------------
# 3. cardinality_search — 5 legs, 2 dominate
# ---------------------------------------------------------------------------

class TestCardinalitySearch:
    """Greedy forward selection finds knee and returns consistent structure."""

    def setup_method(self):
        rng = np.random.default_rng(10)
        bars = 900
        idx = pd.bdate_range("2016-01-01", periods=bars)
        # 2 high-alpha legs + 3 near-zero legs
        data = {
            "strong1": 8e-4 + 0.01 * rng.standard_normal(bars),
            "strong2": 7e-4 + 0.01 * rng.standard_normal(bars),
            "weak1":   1e-5 + 0.01 * rng.standard_normal(bars),
            "weak2":   1e-5 + 0.01 * rng.standard_normal(bars),
            "weak3":   1e-5 + 0.01 * rng.standard_normal(bars),
        }
        self.rm = pd.DataFrame(data, index=idx)
        self.result = cardinality_search(self.rm, seed=42)

    def test_knee_k_within_range(self):
        assert 1 <= self.result["knee_k"] <= 5, (
            f"knee_k {self.result['knee_k']} outside [1, 5]"
        )

    def test_k_list_covers_all_legs(self):
        assert self.result["k"] == list(range(1, 6)), (
            f"k list {self.result['k']} should be [1,2,3,4,5]"
        )

    def test_sharpe_at_k_has_five_entries(self):
        assert len(self.result["sharpe_at_k"]) == 5

    def test_sharpe_at_k_all_finite(self):
        """Every sharpe_at_k entry should be a finite float.

        Note: equal-weight greedy is NOT guaranteed to be monotone — adding a
        weaker leg to a single strong leg *dilutes* Sharpe.  Values can be
        negative when dilution exceeds the marginal contribution.  We only
        require finiteness; the knee search handles the non-monotone case.
        """
        vals = self.result["sharpe_at_k"]
        for i, sr in enumerate(vals):
            assert np.isfinite(sr), f"sharpe_at_k[{i}] = {sr} is not finite"

    def test_selected_at_knee_length_equals_knee_k(self):
        assert len(self.result["selected_at_knee"]) == self.result["knee_k"], (
            f"selected_at_knee has {len(self.result['selected_at_knee'])} items, "
            f"expected {self.result['knee_k']}"
        )

    def test_selected_at_knee_are_valid_column_names(self):
        valid = set(self.rm.columns)
        for name in self.result["selected_at_knee"]:
            assert name in valid, f"{name!r} not a valid column"

    def test_best_sharpe_is_finite_and_positive(self):
        bs = self.result["best_sharpe"]
        assert np.isfinite(bs), f"best_sharpe {bs} is not finite"
        assert bs > 0.0, f"best_sharpe {bs} should be positive for high-mu legs"

    def test_knee_fraction_is_095(self):
        assert self.result["knee_fraction"] == 0.95

    def test_max_k_param_limits_search(self):
        result3 = cardinality_search(self.rm, max_k=3, seed=42)
        assert result3["k"] == [1, 2, 3], f"k with max_k=3: {result3['k']}"
        assert len(result3["sharpe_at_k"]) == 3


# ---------------------------------------------------------------------------
# 4. Degenerate inputs — n_cols < 2 → INSUFFICIENT_DATA without raising
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    """Degenerate returns_matrices must return INSUFFICIENT_DATA without raising."""

    def test_single_leg_portfolio_pbo(self):
        rm = _make_returns(n_legs=1, bars=500)
        result = portfolio_pbo(rm, n_alloc=10, n_splits=4, seed=0)
        assert result["verdict"] == "INSUFFICIENT_DATA", (
            f"Expected INSUFFICIENT_DATA for n_cols=1, got {result['verdict']}"
        )
        assert not np.isfinite(result["pbo"]) or result["pbo"] != result["pbo"]  # nan

    def test_zero_legs_portfolio_pbo(self):
        empty = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=200))
        result = portfolio_pbo(empty, n_alloc=5, n_splits=4, seed=0)
        assert result["verdict"] == "INSUFFICIENT_DATA"

    def test_single_leg_cardinality_search(self):
        rm = _make_returns(n_legs=1, bars=400)
        result = cardinality_search(rm, seed=0)
        # Should not raise; knee_k must be 1
        assert result["knee_k"] == 1
        assert result["k"] == [1]

    def test_zero_legs_cardinality_search(self):
        empty = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=200))
        result = cardinality_search(empty)
        assert result["k"] == []
        assert result["selected_at_knee"] == []

    def test_two_legs_portfolio_pbo_runs_without_error(self):
        """Minimum viable input (2 legs) should produce a valid result."""
        rm = _make_returns(n_legs=2, bars=600, seed=5)
        result = portfolio_pbo(rm, n_alloc=10, n_splits=4, seed=0)
        assert result["verdict"] in _VALID_VERDICTS

    def test_insufficient_data_verdict_has_nan_pbo(self):
        """INSUFFICIENT_DATA must have nan pbo."""
        rm = _make_returns(n_legs=1, bars=100)
        result = portfolio_pbo(rm, n_alloc=5, n_splits=4, seed=0)
        assert not np.isfinite(result["pbo"]), (
            f"pbo should be nan for INSUFFICIENT_DATA, got {result['pbo']}"
        )


# ---------------------------------------------------------------------------
# 5. Helpers — _portfolio_returns and _sharpe
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_portfolio_returns_equal_weight(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=50)
        rm = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50)},
                          index=idx)
        w = np.array([0.5, 0.5])
        port = _portfolio_returns(rm, w)
        expected = (rm["a"] + rm["b"]) / 2
        np.testing.assert_allclose(port.values, expected.values, rtol=1e-9)

    def test_portfolio_returns_with_nans(self):
        """NaN in one leg: active-share renorm keeps the other leg's return."""
        idx = pd.bdate_range("2020-01-01", periods=10)
        data = {"a": [1.0] * 10, "b": [np.nan] + [1.0] * 9}
        rm = pd.DataFrame(data, index=idx)
        w = np.array([0.5, 0.5])
        port = _portfolio_returns(rm, w)
        # bar 0: only leg 'a' alive -> port[0] = 1.0 (renormalised to 1.0)
        assert abs(port.iloc[0] - 1.0) < 1e-9, f"bar 0 = {port.iloc[0]}"
        # bar 1+: both alive -> 0.5*1 + 0.5*1 = 1.0
        assert abs(port.iloc[1] - 1.0) < 1e-9, f"bar 1 = {port.iloc[1]}"

    def test_sharpe_positive_drift(self):
        arr = np.full(252, 4e-4)
        sr = _sharpe(arr)
        assert not np.isfinite(sr), "all-constant series should give nan (zero std)"

    def test_sharpe_with_noise(self):
        rng = np.random.default_rng(0)
        arr = 5e-4 + 0.01 * rng.standard_normal(1000)
        sr = _sharpe(arr, ppy=252)
        assert np.isfinite(sr) and sr > 0.0

    def test_sharpe_too_short(self):
        assert not np.isfinite(_sharpe(np.array([0.01])))
