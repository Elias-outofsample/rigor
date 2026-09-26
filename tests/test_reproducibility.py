"""Framework-correctness regression: deterministic offline reproduction.

The research workflow runs on *live* data (close but not byte-identical across
machines — expected). Reproduction matters only to prove the framework itself is
deterministic: given **fixed input data** + **fixed code**, the data layer ->
engine -> metrics path must produce the same output.

One strategy (``sma_trend``, single instrument SPY) runs ``--offline`` against a
committed snapshot (``fixtures/repro/`` — EOD/div/splits for one symbol). The snapshot
is *synthetic* (``fixtures/repro/make_synthetic_spy.py``, fixed seed): vendor data is
not redistributable, and determinism does not need real prices. No key, no network.

**On cross-platform byte-exactness.** Bit-identical output is *platform-locked*:
floating-point rounding and comparison boundaries (e.g. the SMA crossover) make the
exact bytes depend on the OS/CPU. So there are two checks:

  * a **portable** check (every platform): the run produces the right number of
    bars and sane, finite, stable metrics — catches crashes, NaNs, wrong windows,
    or a grossly broken engine;
  * a **strict byte-exact** check pinned to the **Linux CI runner** (skipped
    elsewhere): the daily-returns hash must equal a stored golden — the precise
    regression guard that runs on every push.

Refresh the strict golden intentionally (after a deliberate change) by reading the
hash printed at the bottom of a CI run, or run this file on Linux.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from rigor.data import DataConfig, DataLoader
from rigor.project.run import load_config, load_strategy_module

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "repro"
AS_OF = "2026-06-01"
STRATEGY = REPO / "strategies" / "trend_following" / "sma_trend"

GOLDEN_BARS = 5584
GOLDEN_SHARPE = 0.396916          # reference (Linux); portable check is tolerant
# sha256 of sma_trend's daily-returns CSV on the Linux CI runner (the canonical platform).
GOLDEN_LINUX_RETURNS_SHA = "a60615c221f9d318278bc57198434303776317f48681c44f06fdeeea6c8a352e"


def _run_offline():
    config, _ = load_config(STRATEGY)
    data = DataLoader(DataConfig(api_key="OFFLINE", cache_dir=FIXTURE, as_of=AS_OF, offline=True))
    return load_strategy_module(STRATEGY, "sma_trend").build(config, data).backtest()


def test_offline_runs_with_stable_output():
    """Portable on every platform: the offline path produces the expected window
    and sane, stable headline metrics."""
    r = _run_offline()
    assert len(r.returns) == GOLDEN_BARS                       # exact (date-driven)
    assert np.isfinite(r.metrics["sharpe"])
    assert abs(r.metrics["sharpe"] - GOLDEN_SHARPE) < 0.05     # tolerant: cross-platform FP
    assert float(r.returns.abs().sum()) > 0                    # not a degenerate all-zero run


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="byte-exact golden is pinned to the Linux CI runner")
def test_byte_identical_on_ci_platform():
    """Strict regression guard on the canonical CI platform: byte-identical returns."""
    r = _run_offline()
    sha = hashlib.sha256(r.returns.to_csv().encode()).hexdigest()
    assert sha == GOLDEN_LINUX_RETURNS_SHA, (
        f"sma_trend offline returns changed on Linux: {sha} != golden "
        f"{GOLDEN_LINUX_RETURNS_SHA}. If intentional, update GOLDEN_LINUX_RETURNS_SHA.")


if __name__ == "__main__":  # print the current hash (run on Linux to refresh the golden)
    res = _run_offline()
    print(f"platform = {sys.platform}")
    print(f"bars     = {len(res.returns)}  sharpe = {res.metrics['sharpe']:.6f}")
    print(f"sha256   = {hashlib.sha256(res.returns.to_csv().encode()).hexdigest()}")
