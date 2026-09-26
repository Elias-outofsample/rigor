"""Wave A: portfolio analytics (beta family, significance, regimes, MC, frontier, stress)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_engine import (
    AllocationConfig,
    PortfolioAllocator,
    PortfolioBacktester,
    StrategyCandidate,
    analysis,
    stress,
)

REPO = Path(__file__).resolve().parents[2]


def _ret(mu=0.0004, bars=1200, seed=0, beta=0.0, bench=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=bars)
    noise = 0.01 * rng.standard_normal(bars)
    vals = mu + noise + (beta * bench.to_numpy() if bench is not None else 0.0)
    return pd.Series(vals, index=idx)


def _matrix(n=5, bars=1200):
    return pd.concat({f"s{i}": _ret(0.0004 + 0.0001 * i, bars, seed=i) for i in range(n)}, axis=1)


def test_beta_family():
    bench = _ret(0.0003, seed=99)
    port = _ret(0.0005, seed=1, beta=0.8, bench=bench)
    bd = analysis.beta_decomposition(port, bench)
    assert 0.4 < bd["beta"] < 1.2          # recovers the ~0.8 beta
    assert {"up_beta", "down_beta", "alpha"} <= set(bd)
    assert "up_capture" in analysis.capture_ratios(port, bench)
    assert np.isfinite(analysis.drawdown_beta(port, bench))
    assert "asymmetry" in analysis.conditional_correlation(port, bench)
    assert len(analysis.rolling_beta(port, bench, window=252)) > 0
    assert len(analysis.rolling_correlation(port, bench, window=252)) > 0


def test_significance_reuses_rigor():
    sig = analysis.significance(_ret(0.0008, seed=2))
    assert "bootstrap_ci" in sig and 0.0 <= sig["psr"] <= 1.0 and 0.0 <= sig["dsr"] <= 1.0


def test_regimes_and_performance():
    port = _ret(0.0005, seed=3)
    regimes = analysis.market_regimes(port)
    assert regimes.notna().all() and regimes.str.contains("-").all()
    perf = analysis.regime_performance(port, regimes)
    assert perf and all("sharpe" in v and "pct_time" in v for v in perf.values())


def test_monte_carlo_and_ruin():
    port = _ret(0.0006, seed=4)
    mc = analysis.monte_carlo(port, n_sims=300)
    assert 0.0 <= mc["prob_loss"] <= 1.0 and mc["worst_max_dd"] <= 0.0
    ru = analysis.ruin_probability(port, target_drawdown=-0.15, horizon=252, n_sims=500)
    assert 0.0 <= ru["ruin_prob"] <= 1.0


def test_efficient_frontier_and_rebalance_sweep():
    rm = _matrix()
    front = analysis.efficient_frontier(rm, n_samples=1500)
    assert len(front) >= 3 and (front["vol"].values[:-1] <= front["vol"].values[1:] + 1e-9).all()
    sweep = analysis.rebalance_sweep(rm, np.ones(rm.shape[1]) / rm.shape[1])
    assert set(sweep.index) >= {"Never", "Monthly"} and "turnover" in sweep.columns


def test_stress_scenarios():
    # a long daily series spanning several crises
    long = _ret(0.0004, bars=5200, seed=7)
    long.index = pd.bdate_range("2005-01-01", periods=5200)
    results = stress.run_all_scenarios(long)
    assert results and all("max_dd" in r and r["covered"] for r in results)


def test_sector_aggregation():
    cands = [StrategyCandidate(f"S{i}", REPO, _ret(seed=i), slug=f"s{i}",
                               sector_exposure={"Energy" if i % 2 else "Crypto": 1.0})
             for i in range(4)]
    expo = analysis.portfolio_sector_exposure(cands, {f"S{i}": 0.25 for i in range(4)})
    assert abs(sum(expo.values()) - 1.0) < 1e-6 and set(expo) == {"Energy", "Crypto"}


def test_regime_dependent_alloc_and_dd_throttle():
    cands = [StrategyCandidate(f"S{i}", REPO, _ret(0.0004 + 0.0001 * i, seed=i),
                               slug=f"s{i}") for i in range(5)]
    w = PortfolioAllocator(cands, AllocationConfig(mode="regime_dependent")).allocate()
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-9).all()
    # dd-throttle reduces realised volatility vs no throttle
    base = PortfolioBacktester(cands, w, start_date=None).run()
    throttled = PortfolioBacktester(cands, w, start_date=None,
                                    dd_throttle_trigger=0.05).run()
    assert throttled.metrics["volatility"] <= base.metrics["volatility"] + 1e-9


def test_regime_dependent_fallback_contract():
    """Below the ~1y minimum, regime_dependent is exactly plain inverse-vol (the
    documented fallback); with full history the per-regime blend runs to a valid
    simplex (so the mode is not a silent inverse-vol stub)."""
    short = [StrategyCandidate(f"S{i}", REPO, _ret(0.0004, bars=120, seed=i), slug=f"s{i}")
             for i in range(4)]
    w_fb = PortfolioAllocator(short, AllocationConfig(mode="regime_dependent")).allocate()
    w_iv = PortfolioAllocator(short, AllocationConfig(mode="inverse_vol")).allocate()
    np.testing.assert_allclose(w_fb, w_iv, atol=1e-9)

    full = [StrategyCandidate(f"S{i}", REPO, _ret(0.0004, bars=800, seed=i), slug=f"s{i}")
            for i in range(4)]
    w_full = PortfolioAllocator(full, AllocationConfig(mode="regime_dependent")).allocate()
    assert abs(w_full.sum() - 1.0) < 1e-6 and (w_full >= -1e-9).all()
