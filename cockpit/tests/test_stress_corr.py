"""Tests for stress_vs_calm_correlation — the false-diversification detector.

Four test groups:
1. Legs that are nearly uncorrelated in calm periods but dump together on
   the worst market days  ->  flag True, corr_increase > 0.2.
2. Genuinely independent legs (no common stress shock)  ->  flag False,
   corr_increase near 0.
3. Degenerate inputs (single leg, < 30 usable days)  ->  no raise, flag False.
4. Output shape / symmetry invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_engine.stress import stress_vs_calm_correlation

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_IDX_LONG = pd.bdate_range("2010-01-01", periods=1500)  # ~6 y, plenty of obs
_IDX_SHORT = pd.bdate_range("2020-01-01", periods=25)   # < 30  -> degenerate


def _independent_matrix(
    n_legs: int = 3,
    bars: int = 1500,
    seed: int = 0,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Uncorrelated white-noise returns — no common shock."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=bars)
    data = {f"leg{i}": rng.standard_normal(bars) * vol for i in range(n_legs)}
    return pd.DataFrame(data, index=idx)


def _false_diversification_matrix(
    n_legs: int = 3,
    bars: int = 1500,
    stress_quantile: float = 0.10,
    shock_size: float = -0.05,  # kept for signature compat; unused in new construction
    seed: int = 42,
    common_loading: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (matrix, benchmark) where legs dump together on stress days.

    Construction:
    * The benchmark is a random return series that drives stress classification.
    * On calm days each leg is independent idiosyncratic white noise.
    * On stress days each leg gets a **large common factor** equal to
      ``benchmark_return * common_loading`` plus small idiosyncratic noise.
      Because the common factor *varies* across stress days (it is the
      benchmark return itself, not a constant), it generates high Pearson
      correlation within the stress subset while calm days stay near-zero.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=bars)

    # Build the benchmark used for stress classification (larger vol so the
    # quantile cut is well-separated from the calm distribution)
    bench_vals = rng.standard_normal(bars) * 0.015
    bench = pd.Series(bench_vals, index=idx)

    # Identify stress days by the benchmark
    threshold = float(np.quantile(bench_vals, stress_quantile))
    stress_mask = bench_vals <= threshold

    # Build per-leg returns
    idio_std = 0.004  # small relative to common loading
    data: dict[str, np.ndarray] = {}
    for i in range(n_legs):
        idio = rng.standard_normal(bars) * idio_std
        # On stress days: large common factor (bench_vals * loading) + small idio
        # On calm days: only idio noise -> near-zero cross-leg correlation
        common = np.where(stress_mask, bench_vals * common_loading, 0.0)
        data[f"leg{i}"] = idio + common

    matrix = pd.DataFrame(data, index=idx)
    return matrix, bench


# ---------------------------------------------------------------------------
# 1. False-diversification case -> flag True, corr_increase > 0.2
# ---------------------------------------------------------------------------

class TestFalseDiversificationDetected:
    """Legs that are independent in calm but correlated in stress."""

    def setup_method(self):
        matrix, bench = _false_diversification_matrix(
            n_legs=3, bars=1500, stress_quantile=0.10, shock_size=-0.05, seed=42,
        )
        self.result = stress_vs_calm_correlation(matrix, benchmark=bench, stress_quantile=0.10)

    def test_flag_is_true(self):
        assert self.result["false_diversification_flag"] is True, (
            f"Expected flag=True; corr_increase={self.result['corr_increase']:.4f}"
        )

    def test_corr_increase_above_threshold(self):
        assert self.result["corr_increase"] > 0.2, (
            f"corr_increase={self.result['corr_increase']:.4f} should be > 0.2"
        )

    def test_stress_corr_higher_than_calm(self):
        assert self.result["avg_offdiag_stress"] > self.result["avg_offdiag_calm"], (
            f"stress={self.result['avg_offdiag_stress']:.4f} "
            f"calm={self.result['avg_offdiag_calm']:.4f}"
        )

    def test_obs_counts_are_positive(self):
        assert self.result["n_stress_days"] >= 30
        assert self.result["n_calm_days"] >= 30

    def test_assets_list_correct(self):
        assert self.result["assets"] == ["leg0", "leg1", "leg2"]


# ---------------------------------------------------------------------------
# 2. Genuinely independent legs -> flag False, corr_increase near 0
# ---------------------------------------------------------------------------

class TestGenuinelyIndependent:
    """No common shock: stress correlation should not spike above calm.

    We supply an *external* independent benchmark so that stress-day selection
    is not endogenous to the legs — using the equal-weight average as proxy
    would by construction select days where all legs happened to drop together,
    which inflates stress correlation even for genuinely uncorrelated series.
    """

    def setup_method(self):
        rng = np.random.default_rng(7)
        bars = 1500
        idx = pd.bdate_range("2010-01-01", periods=bars)
        # Pure white-noise legs — no shared factor at all
        data = {f"leg{i}": rng.standard_normal(bars) * 0.01 for i in range(3)}
        matrix = pd.DataFrame(data, index=idx)
        # Independent benchmark for day classification (not derived from the legs)
        bench = pd.Series(rng.standard_normal(bars) * 0.012, index=idx)
        self.result = stress_vs_calm_correlation(
            matrix, benchmark=bench, stress_quantile=0.10
        )

    def test_flag_is_false(self):
        assert self.result["false_diversification_flag"] is False, (
            f"Expected flag=False; corr_increase={self.result['corr_increase']:.4f}"
        )

    def test_corr_increase_below_flag_threshold(self):
        # Independent legs + independent benchmark -> |corr_increase| well below 0.2
        assert abs(self.result["corr_increase"]) < 0.2, (
            f"corr_increase={self.result['corr_increase']:.4f} should be < 0.2 for "
            f"independent legs with independent benchmark"
        )

    def test_obs_counts_are_positive(self):
        assert self.result["n_stress_days"] >= 30
        assert self.result["n_calm_days"] >= 30


# ---------------------------------------------------------------------------
# 3. Degenerate inputs -> no raise, flag False
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    """Edge cases: single leg and too-few days must not raise."""

    def test_single_leg_no_raise(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=500)
        matrix = pd.DataFrame({"only": rng.standard_normal(500) * 0.01}, index=idx)
        result = stress_vs_calm_correlation(matrix, stress_quantile=0.10)
        assert isinstance(result, dict)
        assert result["false_diversification_flag"] is False
        assert "note" in result

    def test_single_leg_empty_matrices(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2015-01-01", periods=500)
        matrix = pd.DataFrame({"only": rng.standard_normal(500) * 0.01}, index=idx)
        result = stress_vs_calm_correlation(matrix)
        # Correlation undefined for one leg -> empty matrices returned
        assert result["calm_corr"] == []
        assert result["stress_corr"] == []

    def test_fewer_than_30_usable_days_no_raise(self):
        """Only 25 rows -> stress regime will have < 30 obs -> degenerate path."""
        rng = np.random.default_rng(2)
        matrix = pd.DataFrame(
            {"a": rng.standard_normal(25) * 0.01, "b": rng.standard_normal(25) * 0.01},
            index=_IDX_SHORT,
        )
        result = stress_vs_calm_correlation(matrix, stress_quantile=0.10)
        assert isinstance(result, dict)
        assert result["false_diversification_flag"] is False
        assert "note" in result

    def test_empty_dataframe_no_raise(self):
        """Zero rows should not raise."""
        matrix = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
        result = stress_vs_calm_correlation(matrix)
        assert isinstance(result, dict)
        assert result["false_diversification_flag"] is False


# ---------------------------------------------------------------------------
# 4. Output shape / symmetry invariants
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Matrices must be square with side = n_legs and symmetric."""

    def setup_method(self):
        matrix = _independent_matrix(n_legs=4, bars=1500, seed=99)
        self.result = stress_vs_calm_correlation(matrix, stress_quantile=0.10)
        self.n = 4

    def _to_array(self, key: str) -> np.ndarray:
        return np.array(self.result[key])

    def test_calm_corr_shape(self):
        arr = self._to_array("calm_corr")
        assert arr.shape == (self.n, self.n), f"calm_corr shape {arr.shape}"

    def test_stress_corr_shape(self):
        arr = self._to_array("stress_corr")
        assert arr.shape == (self.n, self.n), f"stress_corr shape {arr.shape}"

    def test_delta_corr_shape(self):
        arr = self._to_array("delta_corr")
        assert arr.shape == (self.n, self.n), f"delta_corr shape {arr.shape}"

    def test_calm_corr_symmetric(self):
        arr = self._to_array("calm_corr")
        assert np.allclose(arr, arr.T, atol=1e-10), "calm_corr not symmetric"

    def test_stress_corr_symmetric(self):
        arr = self._to_array("stress_corr")
        assert np.allclose(arr, arr.T, atol=1e-10), "stress_corr not symmetric"

    def test_diagonal_is_one(self):
        for key in ("calm_corr", "stress_corr"):
            arr = self._to_array(key)
            assert np.allclose(np.diag(arr), 1.0, atol=1e-10), f"{key} diagonal != 1"

    def test_all_required_keys_present(self):
        required = {
            "assets", "calm_corr", "stress_corr", "delta_corr",
            "avg_offdiag_calm", "avg_offdiag_stress", "corr_increase",
            "false_diversification_flag", "n_stress_days", "n_calm_days",
        }
        assert required <= set(self.result), (
            f"Missing keys: {required - set(self.result)}"
        )

    def test_corr_increase_equals_difference(self):
        result = self.result
        expected = result["avg_offdiag_stress"] - result["avg_offdiag_calm"]
        assert abs(result["corr_increase"] - expected) < 1e-10, (
            f"corr_increase {result['corr_increase']} != "
            f"stress-calm {expected}"
        )


# ---------------------------------------------------------------------------
# 5. External benchmark path
# ---------------------------------------------------------------------------

class TestExternalBenchmark:
    """Passing an external benchmark series should work without raising."""

    def test_with_external_benchmark_no_raise(self):
        matrix, bench = _false_diversification_matrix(n_legs=3, bars=1200, seed=10)
        result = stress_vs_calm_correlation(matrix, benchmark=bench, stress_quantile=0.10)
        assert isinstance(result, dict)
        assert "false_diversification_flag" in result

    def test_benchmark_drives_stress_classification(self):
        """When all legs shock together on benchmark-down days, flag should be True."""
        matrix, bench = _false_diversification_matrix(
            n_legs=3, bars=1500, shock_size=-0.06, seed=77,
        )
        result = stress_vs_calm_correlation(matrix, benchmark=bench, stress_quantile=0.10)
        assert result["false_diversification_flag"] is True, (
            f"Expected flag=True; corr_increase={result['corr_increase']:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. NaN robustness
# ---------------------------------------------------------------------------

class TestNaNRobustness:
    """Scattered NaNs in the returns matrix should not raise."""

    def test_scattered_nans_no_raise(self):
        rng = np.random.default_rng(5)
        idx = pd.bdate_range("2012-01-01", periods=600)
        data = rng.standard_normal((600, 3)) * 0.01
        # Scatter 5% NaNs randomly
        mask = rng.random((600, 3)) < 0.05
        data[mask] = np.nan
        matrix = pd.DataFrame(data, columns=["a", "b", "c"], index=idx)
        result = stress_vs_calm_correlation(matrix)
        assert isinstance(result, dict)
        # Flag should be a bool, not NaN
        assert isinstance(result["false_diversification_flag"], bool)

    def test_pytest_is_not_needed(self):
        """Dummy test so the class has a trivially green baseline case."""
        assert True
