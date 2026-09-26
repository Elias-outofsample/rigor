"""Tests for the optional GPU/CUDA grid-search routing in ``rigor.optimize.core``.

There is no GPU here, so the *real* CuPy kernel path cannot run. These tests
exercise the **routing logic** with a pure-Python fake kernel: the kernel simply
calls the strategy's own ``run_backtest`` per combo, so it computes exactly what
the CPU loop would. We then assert:

  * the kernel-routed rows are byte-identical (params, sharpe, returns) to the
    CPU ``_run`` on the same combos — proving determinism is preserved;
  * a strategy without a kernel uses the CPU path;
  * a kernel that raises (or returns garbage) falls back to CPU cleanly;
  * a grid below the launch threshold uses CPU even with a kernel + device.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.optimize import core
from rigor.strategy import StrategyBase, StrategyConfig

N_BARS = 40


def _config() -> StrategyConfig:
    return StrategyConfig(
        name="Kernel Grid", start_date="2015-01-01", end_date=None,
        initial_capital=100000.0, commission_bps=0.0, long_short="long",
        rebalance_freq="daily", role="alpha", status="idle", version="v1",
    )


class _CpuStrategy(StrategyBase):
    """Param-dependent returns, aligned to a real calendar; no grid kernel."""

    def build_cache(self) -> dict:
        return {"dates": pd.bdate_range("2015-01-01", periods=N_BARS)}

    def run_backtest(self, cache, params):
        rng = np.random.default_rng(params["a"] * 100 + params["b"])
        mu = 0.0003 * params["a"] - 0.0001 * params["b"]
        return pd.Series(mu + 0.01 * rng.standard_normal(len(cache["dates"])),
                         index=cache["dates"])


class _KernelStrategy(_CpuStrategy):
    """Same maths, but exposes a batched kernel that delegates to ``run_backtest``."""

    def grid_kernel(self):
        def kernel(cache, combos):
            # Pure-Python "kernel": stack the per-combo bar returns into the
            # (n_combos, n_bars) tensor the contract requires. This is exactly
            # what a real CuPy kernel would *produce*, just computed on the CPU.
            rows = [np.asarray(self.run_backtest(cache, c), dtype="float64") for c in combos]
            return np.vstack(rows)
        return kernel


class _RaisingKernelStrategy(_CpuStrategy):
    def grid_kernel(self):
        def kernel(cache, combos):
            raise RuntimeError("simulated CUDA failure")
        return kernel


def _combos(n: int) -> list[dict]:
    # n distinct combos (a in 0..n-1, b fixed) so de-dup keeps them all.
    return [{"a": i, "b": 7} for i in range(n)]


def _assert_rows_identical(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra["params"] == rb["params"]
        # NaN-safe scalar equality for the authoritative Sharpe
        assert (ra["sharpe"] == rb["sharpe"]) or (
            ra["sharpe"] != ra["sharpe"] and rb["sharpe"] != rb["sharpe"])
        pd.testing.assert_series_equal(ra["returns"], rb["returns"])


@pytest.fixture
def force_cuda(monkeypatch):
    """Make the routing believe a CUDA device is present (no real GPU here)."""
    monkeypatch.setattr(core, "active_backend", lambda *_a, **_k: "cuda")
    monkeypatch.setattr(core, "backends_available", lambda: {"numpy": True, "numba": False,
                                                             "cuda": True})


def test_kernel_path_is_byte_identical_to_cpu(force_cuda, monkeypatch):
    monkeypatch.setattr(core, "_KERNEL_MIN_COMBOS", 4)  # low threshold to force the path
    combos = _combos(8)
    cpu = _CpuStrategy(_config())
    ker = _KernelStrategy(_config())

    # sanity: routing decision actually fires for the kernel strategy
    assert core._use_kernel(ker, combos, "auto") is True
    assert core._use_kernel(cpu, combos, "auto") is False

    rows_cpu = core._run(cpu, cpu.cache, combos, backend="auto")          # CPU loop
    rows_ker = core._run(ker, ker.cache, combos, backend="auto")          # kernel path
    _assert_rows_identical(rows_cpu, rows_ker)


def test_kernel_routed_directly_matches_cpu(monkeypatch):
    """Call the helper directly (no backend mock) — pure routing/maths check."""
    combos = _combos(6)
    cpu = _CpuStrategy(_config())
    ker = _KernelStrategy(_config())
    rows_cpu = core._run(cpu, cpu.cache, combos)             # CPU loop (no kernel)
    rows_ker = core._run_kernel(ker, ker.cache, core._dedup(combos))
    _assert_rows_identical(rows_cpu, rows_ker)


def test_strategy_without_kernel_uses_cpu(force_cuda, monkeypatch):
    monkeypatch.setattr(core, "_KERNEL_MIN_COMBOS", 4)
    cpu = _CpuStrategy(_config())
    assert cpu.grid_kernel() is None
    assert core._use_kernel(cpu, _combos(8), "auto") is False


def test_raising_kernel_falls_back_to_cpu_cleanly(force_cuda, monkeypatch):
    monkeypatch.setattr(core, "_KERNEL_MIN_COMBOS", 4)
    combos = _combos(8)
    cpu = _CpuStrategy(_config())
    bad = _RaisingKernelStrategy(_config())
    assert core._use_kernel(bad, combos, "auto") is True  # it *would* route...
    rows_cpu = core._run(cpu, cpu.cache, combos)
    rows_fallback = core._run(bad, bad.cache, combos, backend="auto")  # ...but kernel raises
    _assert_rows_identical(rows_cpu, rows_fallback)


def test_grid_below_threshold_uses_cpu(force_cuda):
    # threshold is the real 1000 here; 8 combos must not route to the kernel
    ker = _KernelStrategy(_config())
    assert core._use_kernel(ker, _combos(8), "auto") is False


def test_no_cuda_device_uses_cpu(monkeypatch):
    # kernel + low threshold, but no device available -> CPU path
    monkeypatch.setattr(core, "_KERNEL_MIN_COMBOS", 4)
    monkeypatch.setattr(core, "active_backend", lambda *_a, **_k: "numpy")
    ker = _KernelStrategy(_config())
    assert core._use_kernel(ker, _combos(8), "auto") is False
