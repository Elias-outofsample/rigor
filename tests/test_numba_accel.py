"""Tests for the optional Numba acceleration of the optimiser's composite-score
hot path (``rigor.optimize.numba_accel``).

There is no Numba in CI, so the *default* path here is always the pure-NumPy
fallback. These tests assert:

  * the NumPy fallback matches an independent reference geomean (correctness);
  * IF Numba is importable, the Numba kernel equals the NumPy fallback
    *bit-for-bit* (parity proof) — otherwise the case is skipped;
  * the default dispatch picks the fallback when the flag is off / Numba absent;
  * a tiny end-to-end ``score_configs`` still works and is unchanged whether or
    not acceleration is enabled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.optimize import numba_accel as na
from rigor.optimize.scoring import score_configs


def _reference_geomean(pillars: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Independent weighted-geomean reference (``prod(x**w) ** (1/sum w)``)."""
    wsum = weights.sum()
    return np.prod(pillars ** weights, axis=1) ** (1.0 / wsum)


def _rand_pillars(seed: int = 0, n: int = 257, k: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # strictly positive, in the score_configs domain ([eps, 10+eps]); weights >= 1
    pillars = rng.uniform(0.1, 10.1, size=(n, k))
    weights = np.ones(k)
    weights[0], weights[3], weights[6] = 3.0, 1.5, 1.2  # mirror the calibration weights
    return pillars, weights


def test_numpy_fallback_matches_reference() -> None:
    """(1) The NumPy fallback computes the weighted geomean correctly."""
    pillars, weights = _rand_pillars()
    out = na.weighted_geomean(pillars, weights, enable=False)
    assert out.shape == (pillars.shape[0],)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, _reference_geomean(pillars, weights), rtol=1e-12, atol=0.0)


def test_default_dispatch_uses_fallback_when_flag_off(monkeypatch) -> None:
    """(3) With the env flag off, dispatch resolves to the NumPy fallback even if
    Numba is installed — so the default (and CI) path is the fallback."""
    monkeypatch.delenv("RIGOR_OPTIMIZE_ACCEL", raising=False)
    assert na.accel_enabled() is False
    assert na.active_impl() == "numpy"
    assert na.active_impl(enable=None) == "numpy"


def test_dispatch_uses_fallback_when_numba_absent(monkeypatch) -> None:
    """(3) Even with the flag ON, no importable Numba -> NumPy fallback. This is
    exactly the CI situation (numba not installed)."""
    monkeypatch.setattr(na, "_HAS_NUMBA", False)
    monkeypatch.setenv("RIGOR_OPTIMIZE_ACCEL", "1")
    assert na.accel_enabled() is True
    assert na.numba_available() is False
    assert na.active_impl() == "numpy"
    # ...and it still produces the reference result via the fallback.
    pillars, weights = _rand_pillars(seed=3)
    np.testing.assert_allclose(
        na.weighted_geomean(pillars, weights),
        _reference_geomean(pillars, weights), rtol=1e-12, atol=0.0,
    )


def test_env_flag_truthy_parsing(monkeypatch) -> None:
    """The env flag accepts the documented truthy spellings; anything else is off."""
    for val in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("RIGOR_OPTIMIZE_ACCEL", val)
        assert na.accel_enabled() is True
    for val in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("RIGOR_OPTIMIZE_ACCEL", val)
        assert na.accel_enabled() is False


@pytest.mark.skipif(not na.numba_available(), reason="numba not installed (CI fallback path)")
def test_numba_path_is_bit_identical_to_numpy() -> None:
    """(2) When Numba is importable, the kernel equals the NumPy fallback to the
    last bit — the parity proof. Skipped on CI (no numba)."""
    for seed in range(5):
        pillars, weights = _rand_pillars(seed=seed)
        np_out = na.weighted_geomean(pillars, weights, enable=False)
        nb_out = na.weighted_geomean(pillars, weights, enable=True)
        assert na.active_impl(enable=True) == "numba"
        # Bit-for-bit equality: identical bytes, not merely "close".
        assert np_out.tobytes() == nb_out.tobytes()
        np.testing.assert_array_equal(np_out, nb_out)


@pytest.mark.skipif(not na.numba_available(), reason="numba not installed (CI fallback path)")
def test_numba_kernel_failure_falls_back(monkeypatch) -> None:
    """A kernel that raises falls back to the identical NumPy result (never crashes)."""
    def _boom(*_a, **_k):
        raise RuntimeError("simulated JIT failure")

    monkeypatch.setattr(na, "_geomean_numba", _boom)
    pillars, weights = _rand_pillars(seed=9)
    out = na.weighted_geomean(pillars, weights, enable=True)
    np.testing.assert_array_equal(out, na.weighted_geomean(pillars, weights, enable=False))


# --- (4) tiny end-to-end optimise scoring is unchanged by the acceleration flag ---

def _toy_rows(n: int = 6) -> list[dict]:
    """A handful of configs with param-dependent daily returns over ~1.4y."""
    idx = pd.bdate_range("2021-01-01", periods=360)
    rng = np.random.default_rng(123)
    rows = []
    for a in range(n):
        mu = 0.0004 + 0.00005 * a
        r = pd.Series(mu + 0.01 * rng.standard_normal(len(idx)), index=idx)
        rows.append({"params": {"a": a}, "returns": r})
    return rows


def test_end_to_end_score_configs_runs_and_is_flag_invariant(monkeypatch) -> None:
    """(4) ``score_configs`` (which uses the hot path) runs and gives an identical
    composite ranking with the accel flag off vs on — results never drift."""
    rows = _toy_rows()

    monkeypatch.delenv("RIGOR_OPTIMIZE_ACCEL", raising=False)  # fallback
    res_off = score_configs(rows)

    monkeypatch.setenv("RIGOR_OPTIMIZE_ACCEL", "1")  # numba if present, else fallback
    res_on = score_configs(rows)

    assert "best_params" in res_off and res_off["best_params"] is not None
    assert res_off["best_params"] == res_on["best_params"]
    assert res_off["best_composite"] == res_on["best_composite"]
    assert [r["params"] for r in res_off["ranking"]] == [r["params"] for r in res_on["ranking"]]
