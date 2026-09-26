"""The Rigor strategy contract — how every strategy in the book is written.

Adopted from an earlier in-house library's ``StrategyBase`` (the industrialised
build_cache / run_backtest / param_grid pattern), hardened for a standardised,
reproducible pipeline:

  * ``equity_fn`` is no longer abstract — it defaults to ``cumprod(1+returns)``,
    so a strategy only must implement ``build_cache`` and ``run_backtest``.
  * No coupling to the heavy optimisation pipeline (CUDA / two-pass / CPCV).
    ``backtest()`` runs the deterministic core path and returns a BacktestResult.
    Optimisation is a separate, opt-in layer added later.

Contract:
  * ``build_cache()``               -> dict (data + indicators, built once)
  * ``run_backtest(cache, params)`` -> np.ndarray | pd.Series of bar NET returns
  * ``param_grid()``                -> dict[str, list] (optional)
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..engine import BacktestResult, TradeLog, result_from_returns
from .config import StrategyConfig


class StrategyBase(ABC):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._cache: dict[str, Any] | None = None

    @property
    def cache(self) -> dict[str, Any]:
        """Lazily built, memoised data/indicator cache (shared across params)."""
        if self._cache is None:
            self._cache = self.build_cache()
        return self._cache

    @abstractmethod
    def build_cache(self) -> dict[str, Any]:
        """Build the shared data/indicator cache once. Should include a ``dates``
        key (DatetimeIndex) so results carry a real calendar."""

    @abstractmethod
    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]):
        """Run one parameter combo -> array/Series of bar NET returns.

        Use the ``cache`` argument (not ``self.cache``) so future eval-window
        restrictions propagate. Most strategies build target weights and call
        ``rigor.engine.simulate_weights`` to get standardised cost accounting.
        """

    def equity_fn(self, cache: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
        """Equity curve for enrichment. Default = cumprod(1+returns)."""
        out = self.run_backtest(cache, params)
        # run_backtest may hand back a full BacktestResult (carrying weights) or a
        # bare returns array/Series — normalise to the returns here.
        out = out.returns if isinstance(out, BacktestResult) else out
        r = np.asarray(out, dtype="float64")
        r = r[np.isfinite(r)]
        return np.cumprod(1.0 + r)

    def param_grid(self) -> dict[str, list]:
        """Default parameter grid (override per strategy). Empty = single run."""
        return {}

    def grid_kernel(self) -> Callable[[dict[str, Any], list[dict]], np.ndarray] | None:
        """Optional batched grid-search kernel — returns ``None`` by default.

        Override to return a callable that evaluates a *whole* combo list at once
        (e.g. a CuPy/CUDA batched backtest), so the optimiser can route a large
        grid through a single GPU launch instead of the per-combo CPU loop.

        Kernel contract::

            kernel(cache, combos: list[dict]) -> np.ndarray   # shape (n_combos, n_bars)

        It is handed the same shared ``cache`` the CPU path uses and the *exact*
        de-duplicated combo list, and must return the per-combo bar NET returns as
        a 2-D tensor whose row ``i`` corresponds to ``combos[i]`` and whose columns
        are bars (aligned to ``cache['dates']`` when present, oldest-first).

        Determinism guarantee: the kernel only *produces the returns faster* — the
        optimiser still computes every authoritative Sharpe and the final ranking
        from ``rigor.metrics`` on those returns, exactly as for the CPU path. A kernel
        that raises, or returns ``None`` / a wrong-shaped tensor, makes the optimiser
        fall back to the CPU loop with byte-identical results (never a crash).
        """
        return None

    def default_params(self) -> dict[str, Any]:
        """Canonical single-run parameters: first grid cell, but ``config.extra`` wins.

        ``config.json`` is the declarative source of truth for a strategy. When an axis
        name also appears in ``config.extra``, the configured value takes precedence over
        the grid's first cell — otherwise ``backtest()`` silently runs a DIFFERENT
        strategy than the one the config declares (the trap the ORB strategies work
        around by hand-ordering their grid).
        """
        extra = getattr(self.config, "extra", None) or {}
        return {k: (extra[k] if k in extra else v[0])
                for k, v in self.param_grid().items() if v}

    def backtest(self, params: dict[str, Any] | None = None) -> BacktestResult:
        """Deterministic core-path run -> BacktestResult (returns, equity, metrics)."""
        params = self.default_params() if params is None else params
        cache = self.cache
        # Event-driven strategies opt into a position ledger by adding
        # ``trade_log=None`` to run_backtest and recording trades into it. We only
        # pass it when the signature accepts it, so existing strategies (and the
        # weights-based path) are completely unaffected.
        log = TradeLog()
        # The concrete ``run_backtest`` override may add ``trade_log=None`` beyond
        # the abstract contract; dispatch through ``Callable[..., Any]`` so the
        # opt-in keyword type-checks against whatever signature the strategy declares.
        run: Callable[..., Any] = self.run_backtest
        if "trade_log" in inspect.signature(self.run_backtest).parameters:
            returns = run(cache, params, trade_log=log)
        else:
            returns = run(cache, params)
        # A strategy may return a full BacktestResult (carrying target weights, so a
        # position ledger can be reconstructed) instead of a bare returns series.
        if isinstance(returns, BacktestResult):
            return returns
        index = None
        if not isinstance(returns, pd.Series):
            dates = cache.get("dates") if isinstance(cache, dict) else None
            if dates is not None:
                idx = pd.DatetimeIndex(np.asarray(dates))
                n = len(np.asarray(returns))
                index = idx[-n:] if len(idx) >= n else None
        return result_from_returns(returns, index=index, trade_log=(log if log.trades else None))
