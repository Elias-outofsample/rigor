"""Hermetic unit tests for signal-decay / OU half-life in signal_quality.py.

Covers ``compute_signal_decay`` (exponential IC-decay fit -> half-life + R^2)
and ``compute_ou_half_life`` (Ornstein-Uhlenbeck mean-reversion half-life).

Strong assertions:
  * a fast-decaying synthetic IC series yields a SHORT half-life;
  * a persistent IC series yields a LONG half-life;
  * a strongly mean-reverting OU series yields a SHORT half-life and is
    flagged mean-reverting, while a random-walk-ish series yields a much
    LONGER (or infinite) half-life and is not flagged;
  * short / degenerate input returns NaN + a clear status WITHOUT raising.

Uses only numpy / pandas / pytest — no heavy optional dependencies.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rigor.analysis.signal_quality import (
    compute_ou_half_life,
    compute_signal_decay,
)

# ---------------------------------------------------------------------------
# compute_signal_decay
# ---------------------------------------------------------------------------


def test_signal_decay_fast_vs_persistent_half_life() -> None:
    """A fast-decaying IC series has a much shorter half-life than a slow one."""
    lags = np.arange(1, 13, dtype=float)
    ic0 = 0.10

    fast = compute_signal_decay(ic0 * np.exp(-0.6 * lags))
    slow = compute_signal_decay(ic0 * np.exp(-0.03 * lags))

    assert fast["status"] == "ok"
    assert slow["status"] == "ok"
    # Both are finite, positive half-lives and fast decays in far fewer bars.
    assert math.isfinite(fast["half_life"])
    assert math.isfinite(slow["half_life"])
    assert fast["half_life"] < slow["half_life"]
    assert fast["half_life"] < 3.0          # ln(2)/0.6 ~= 1.16 bars
    assert slow["half_life"] > 15.0         # ln(2)/0.03 ~= 23 bars
    # A clean exponential fits essentially perfectly in log space.
    assert fast["r_squared"] > 0.99
    assert slow["r_squared"] > 0.99
    # Recovered decay rate matches the generator closely.
    assert fast["decay_rate"] == pytest.approx(0.6, abs=1e-6)


def test_signal_decay_recovers_half_life_value() -> None:
    """ln(2)/lambda is recovered to high precision for a clean exponential."""
    lags = np.arange(1, 21, dtype=float)
    lam = 0.25
    res = compute_signal_decay({int(t): 0.2 * math.exp(-lam * t) for t in lags})
    assert res["status"] == "ok"
    assert res["half_life"] == pytest.approx(math.log(2) / lam, rel=1e-6)
    assert res["peak_ic"] == pytest.approx(0.2 * math.exp(-lam), rel=1e-6)
    assert res["optimal_horizon"] == 1
    assert res["n_obs"] == 20


def test_signal_decay_no_decay_returns_inf() -> None:
    """A flat IC profile yields an effectively-infinite half-life, not a crash."""
    res = compute_signal_decay([0.05, 0.05, 0.05, 0.05, 0.05])
    # A perfectly flat series has slope ~0; the half-life is huge (inf or a vast
    # finite number from float rounding). Either way it signals "no decay".
    assert math.isinf(res["half_life"]) or res["half_life"] > 1e6
    assert res["status"] in {"ok", "no decay (rate <= 0)"}


def test_signal_decay_rising_ic_is_not_decay() -> None:
    """A rising |IC| profile gives a non-positive decay rate -> inf half-life."""
    res = compute_signal_decay([0.02, 0.04, 0.06, 0.08, 0.10])
    assert math.isinf(res["half_life"])
    assert res["decay_rate"] < 0
    assert res["status"] == "no decay (rate <= 0)"


def test_signal_decay_short_input_returns_nan_status() -> None:
    """Too few usable IC points -> NaN half-life + explanatory status (no raise)."""
    res = compute_signal_decay([0.1, 0.05])
    assert math.isnan(res["half_life"])
    assert math.isnan(res["decay_rate"])
    assert res["n_obs"] == 2
    assert "insufficient" in res["status"]


def test_signal_decay_empty_input_returns_nan_status() -> None:
    res = compute_signal_decay([])
    assert math.isnan(res["half_life"])
    assert res["status"] == "empty input"


def test_signal_decay_drops_nonfinite_and_zero_ics() -> None:
    """Non-finite / zero ICs are dropped before the log; the fit still succeeds."""
    lags = np.arange(1, 9, dtype=float)
    ics = 0.1 * np.exp(-0.4 * lags)
    ics[2] = np.nan
    ics[5] = 0.0
    res = compute_signal_decay(ics)
    assert res["status"] == "ok"
    assert res["n_obs"] == 6
    assert math.isfinite(res["half_life"])


# ---------------------------------------------------------------------------
# compute_ou_half_life
# ---------------------------------------------------------------------------


def _ou_path(kappa: float, n: int, sigma: float, seed: int) -> np.ndarray:
    """Simulate a discrete OU / AR(1) path: x_t = (1-kappa) x_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = (1.0 - kappa) * x[i - 1] + sigma * rng.standard_normal()
    return x


def test_ou_mean_reverting_vs_random_walk() -> None:
    """A strongly mean-reverting series reverts fast; a random walk does not."""
    mr = compute_ou_half_life(_ou_path(kappa=0.5, n=2000, sigma=1.0, seed=1))

    rng = np.random.default_rng(2)
    rw = compute_ou_half_life(np.cumsum(rng.standard_normal(2000)))

    assert mr["status"] == "ok"
    assert rw["status"] == "ok"

    # Mean-reverting: short, finite half-life and flagged significant.
    assert math.isfinite(mr["half_life"])
    assert mr["half_life"] < 5.0
    assert mr["kappa"] > 0
    assert mr["is_mean_reverting"] is True
    assert mr["p_value"] < 0.05
    assert mr["half_life"] == pytest.approx(math.log(2) / 0.5, rel=0.25)

    # Random walk: far slower reversion (huge or infinite half-life), not flagged.
    assert rw["is_mean_reverting"] is False
    assert (not math.isfinite(rw["half_life"])) or rw["half_life"] > mr["half_life"] * 5


def test_ou_half_life_annualized_and_mu() -> None:
    res = compute_ou_half_life(
        _ou_path(kappa=0.3, n=1500, sigma=1.0, seed=7), periods_per_year=252
    )
    assert res["status"] == "ok"
    assert math.isfinite(res["half_life"])
    assert res["half_life_annualized"] == pytest.approx(res["half_life"] / 252, rel=1e-9)
    assert math.isfinite(res["mu"])
    assert res["n_obs"] == 1500


def test_ou_accepts_pandas_series() -> None:
    ser = pd.Series(_ou_path(kappa=0.4, n=1000, sigma=1.0, seed=3))
    res = compute_ou_half_life(ser)
    assert res["status"] == "ok"
    assert res["is_mean_reverting"] is True


def test_ou_short_input_returns_nan_status() -> None:
    """Fewer than the minimum observations -> NaN + status, no exception."""
    res = compute_ou_half_life([0.0, 1.0, 0.5, 0.2, 0.7])
    assert math.isnan(res["half_life"])
    assert math.isnan(res["kappa"])
    assert res["is_mean_reverting"] is False
    assert "insufficient" in res["status"]


def test_ou_constant_series_returns_status() -> None:
    """A constant series has no variation to fit -> guarded NaN + status."""
    res = compute_ou_half_life(np.full(100, 3.14))
    assert math.isnan(res["half_life"])
    assert res["status"] == "constant series (no variation)"
