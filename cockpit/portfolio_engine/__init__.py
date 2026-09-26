"""Compute core for the Portfolio Builder.

Ported from the prior research library's ``portfolio_builder`` engine, rewired so
every statistic comes from ``rigor.validation`` / ``rigor.metrics`` (one validation
stack, shared with the rest of the framework) and every strategy is discovered
from the Rigor ``strategies/`` book.
"""
from __future__ import annotations

from . import analysis, stress
from .allocator import AllocationConfig, PortfolioAllocator
from .backtester import BacktestResult, PortfolioBacktester
from .candidates import StrategyCandidate, scan_strategies

# cockpit imports the modules above, so bind it last to keep import order clean.
from .cockpit import build_from_weights, build_portfolio, search_portfolios  # noqa: E402
from .data import build_factor_proxies, load_benchmark_returns, strategy_sector
from .portfolio_io import list_saved, load_portfolio, save_portfolio
from .search import weight_grid_search
from .selector import Constraints, PortfolioSelector
from .weight_methods import METHODS, allocate_weights

__all__ = [
    "StrategyCandidate", "scan_strategies",
    "Constraints", "PortfolioSelector",
    "AllocationConfig", "PortfolioAllocator",
    "BacktestResult", "PortfolioBacktester",
    "weight_grid_search", "allocate_weights", "METHODS",
    "load_benchmark_returns", "strategy_sector", "build_factor_proxies",
    "save_portfolio", "load_portfolio", "list_saved",
    "build_portfolio", "build_from_weights", "search_portfolios",
    "analysis", "stress",
]
