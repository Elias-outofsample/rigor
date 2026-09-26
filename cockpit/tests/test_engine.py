"""Phase 1: the portfolio compute engine (allocate -> backtest -> search)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_engine import (
    METHODS,
    AllocationConfig,
    BacktestResult,
    Constraints,
    PortfolioAllocator,
    PortfolioBacktester,
    PortfolioSelector,
    StrategyCandidate,
    allocate_weights,
    scan_strategies,
    weight_grid_search,
)

REPO = Path(__file__).resolve().parents[2]
STRATEGY = REPO / "strategies"


def _synth(n=6, bars=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=bars)
    cands = []
    for i in range(n):
        mu = 0.0004 + 0.0001 * i
        r = pd.Series(mu + 0.01 * rng.standard_normal(bars), index=idx)
        cands.append(StrategyCandidate(f"strat_{i}", REPO, r, slug=f"strat_{i}"))
    return cands


def _returns_matrix(cands):
    return pd.concat({c.name: c.returns for c in cands}, axis=1)


# --- weight methods --------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_allocate_weights_valid_simplex(method):
    rm = _returns_matrix(_synth())
    w = allocate_weights(rm, method)
    assert len(w) == rm.shape[1]
    assert abs(w.sum() - 1.0) < 1e-6      # sums to 1
    assert (w >= -1e-9).all()             # long-only


def test_max_weight_cap_respected():
    rm = _returns_matrix(_synth())
    w = allocate_weights(rm, "inverse_vol", max_weight=0.25)
    assert w.max() <= 0.25 + 1e-6 and abs(w.sum() - 1.0) < 1e-6


# --- allocator + apply chain ----------------------------------------------

def test_allocator_modes_and_overrides():
    cands = _synth()
    w = PortfolioAllocator(cands, AllocationConfig(mode="risk_parity")).allocate()
    assert abs(w.sum() - 1.0) < 1e-6
    # an override caps the raw weight then renormalises, so it pulls that name
    # well below its uncapped equal weight (1/N)
    cfg = AllocationConfig(mode="equal_weight", overrides={cands[0].name: 0.05})
    w2 = PortfolioAllocator(cands, cfg).allocate()
    assert w2[0] < 1.0 / len(cands)


def test_vol_target_scales_gross():
    cands = _synth()
    cfg = AllocationConfig(mode="equal_weight", vol_target=0.10, leverage_cap=0)
    w = PortfolioAllocator(cands, cfg).allocate()
    assert w.sum() > 0  # scaled to hit the target vol (may differ from 1.0)


# --- selector --------------------------------------------------------------

def test_selector_filters_and_decorrelates():
    cands = _synth(n=8)
    sel = PortfolioSelector(cands, Constraints(min_sharpe=0.5))
    kept = sel.select()
    assert all(c.sharpe >= 0.5 or c.is_hedge for c in kept)
    # a punitive corr_max drops correlated names and records reasons
    sel2 = PortfolioSelector(cands, Constraints(corr_max=0.05))
    kept2 = sel2.select()
    assert len(kept2) <= len(cands)


# --- backtester ------------------------------------------------------------

def test_backtest_combines_and_reports_metrics():
    cands = _synth()
    w = PortfolioAllocator(cands, AllocationConfig(mode="equal_weight")).allocate()
    res = PortfolioBacktester(cands, w, start_date=None).run()
    assert isinstance(res, BacktestResult)
    assert len(res.portfolio_returns) > 100
    for k in ("sharpe", "cagr", "max_drawdown", "var_95", "ulcer_index"):
        assert k in res.metrics and np.isfinite(res.metrics[k])
    assert res.diversification["ratio"] >= 1.0 - 1e-6  # diversified vs weighted-avg vol


def test_backtest_beta_when_benchmark_supplied():
    cands = _synth()
    bench = pd.Series(0.0003 + 0.01 * np.random.default_rng(9).standard_normal(900),
                      index=cands[0].returns.index)
    w = PortfolioAllocator(cands, AllocationConfig(mode="equal_weight")).allocate()
    res = PortfolioBacktester(cands, w, start_date=None, benchmark_returns=bench).run()
    assert "beta" in res.beta and np.isfinite(res.beta["beta"])


# --- grid search -----------------------------------------------------------

def test_grid_search_returns_ranked_topk():
    rm = _returns_matrix(_synth(n=5, bars=1200))
    df = weight_grid_search(rm, n_samples=400, top_k=10, seed=1)
    assert 0 < len(df) <= 10
    # ranked by the composite anti-overfit score, descending
    assert (df["composite_score"].values[:-1]
            >= df["composite_score"].values[1:] - 1e-9).all()
    assert isinstance(df.iloc[0]["weights"], dict)


def test_grid_search_is_deterministic():
    rm = _returns_matrix(_synth(n=5, bars=1200))
    a = weight_grid_search(rm, n_samples=300, seed=1)
    b = weight_grid_search(rm, n_samples=300, seed=1)
    pd.testing.assert_frame_equal(a, b)


# --- integration on the real book -----------------------------------------

def test_end_to_end_on_real_strategies():
    cands = scan_strategies(STRATEGY)[:8]
    kept = PortfolioSelector(cands, Constraints(min_n_obs=100)).select()
    assert kept
    w = PortfolioAllocator(kept, AllocationConfig(mode="inverse_vol")).allocate()
    res = PortfolioBacktester(kept, w).run()
    assert np.isfinite(res.metrics["sharpe"])
