"""Capture-ratio analytics (#capture): Morningstar up/down capture + stress tails."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.analysis import capture


def _bench(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(0.0003 + 0.012 * rng.standard_normal(n),
                     index=pd.bdate_range("2016-01-01", periods=n))


# --- compute_capture_ratios ------------------------------------------------

def test_beta_one_gives_unit_capture():
    """A strategy equal to the benchmark captures ~100% up and down."""
    b = _bench(seed=1)
    out = capture.compute_capture_ratios(b.copy(), b)
    assert abs(out["up_capture"] - 1.0) < 1e-9
    assert abs(out["down_capture"] - 1.0) < 1e-9
    assert abs(out["capture_ratio"] - 1.0) < 1e-9


def test_half_beta_gives_half_capture():
    """r = 0.5*b ⇒ both captures collapse to 0.5 (mean ratio is scale-exact)."""
    b = _bench(seed=2)
    out = capture.compute_capture_ratios(0.5 * b, b)
    assert abs(out["up_capture"] - 0.5) < 1e-9
    assert abs(out["down_capture"] - 0.5) < 1e-9


def test_convex_series_has_more_up_than_down_capture():
    """Amplify up bars, damp down bars ⇒ up_capture > down_capture, ratio > 1."""
    b = _bench(seed=3)
    r = b.where(b <= 0, b * 1.5)        # 1.5x on up days
    r = r.where(b >= 0, b * 0.5)        # 0.5x on down days
    out = capture.compute_capture_ratios(r, b)
    assert out["up_capture"] > out["down_capture"]
    assert out["capture_ratio"] > 1.0
    assert abs(out["up_capture"] - 1.5) < 1e-9
    assert abs(out["down_capture"] - 0.5) < 1e-9


def test_no_overlap_returns_neutral():
    """Disjoint indices ⇒ zero aligned obs ⇒ neutral dict, no raise."""
    a = pd.Series(np.ones(50), index=pd.bdate_range("2016-01-01", periods=50))
    z = pd.Series(np.ones(50), index=pd.bdate_range("2030-01-01", periods=50))
    out = capture.compute_capture_ratios(a, z)
    assert out == {"up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0}


def test_too_few_observations_returns_neutral():
    """Below the 20-obs floor ⇒ neutral dict, no raise / NaN."""
    b = _bench(n=10, seed=4)
    out = capture.compute_capture_ratios(b.copy(), b)
    assert out == {"up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0}


def test_empty_benchmark_returns_neutral():
    out = capture.compute_capture_ratios(_bench(seed=5), pd.Series(dtype="float64"))
    assert out == {"up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0}
    assert all(np.isfinite(v) for v in out.values())


# --- compute_stress_capture ------------------------------------------------

def test_stress_capture_keys_and_tail_sizes():
    b = _bench(seed=6)
    out = capture.compute_stress_capture(b.copy(), b, stress_quantile=0.10)
    assert {"up_capture", "down_capture", "capture_ratio",
            "crash_convex", "n_up", "n_down"} <= set(out)
    # ~10% tails on each side of a 600-bar sample.
    assert 40 <= out["n_up"] <= 80
    assert 40 <= out["n_down"] <= 80


def test_stress_capture_flags_crash_convexity():
    """Damp the crash tail, keep the upside ⇒ convex flag True, down < up."""
    b = _bench(seed=7)
    lo = b.quantile(0.10)
    r = b.where(b > lo, b * 0.3)        # only soften the worst-decile crash bars
    out = capture.compute_stress_capture(r, b, stress_quantile=0.10)
    assert out["crash_convex"] is True
    assert out["down_capture"] < out["up_capture"]
    assert abs(out["up_capture"] - 1.0) < 1e-9


def test_stress_capture_not_convex_when_crash_amplified():
    """Amplify the crash tail ⇒ down_capture > up_capture ⇒ flag False."""
    b = _bench(seed=8)
    lo = b.quantile(0.10)
    r = b.where(b > lo, b * 2.0)        # take MORE of the crash than the market
    out = capture.compute_stress_capture(r, b, stress_quantile=0.10)
    assert out["crash_convex"] is False
    assert out["down_capture"] > out["up_capture"]


def test_stress_capture_degenerate_returns_neutral():
    """Too few obs / thin tails ⇒ neutral, crash_convex False, no raise."""
    b = _bench(n=12, seed=9)
    out = capture.compute_stress_capture(b.copy(), b)
    assert out["crash_convex"] is False
    assert out["up_capture"] == 0.0 and out["down_capture"] == 0.0
    assert out["n_up"] == 0 and out["n_down"] == 0


def test_stress_capture_handles_inf_and_nan_without_raising():
    b = _bench(seed=10)
    r = b.copy()
    r.iloc[:5] = np.nan
    r.iloc[5] = np.inf
    out = capture.compute_stress_capture(r, b)
    assert all(np.isfinite(v) for v in (out["up_capture"], out["down_capture"],
                                        out["capture_ratio"]))
