"""Phase 0: the Rigor strategy-discovery adapter reads the real strategy book."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_engine import StrategyCandidate, scan_strategies
from portfolio_engine.candidates import read_returns_csv

REPO = Path(__file__).resolve().parents[2]
STRATEGY = REPO / "strategies"


def test_discovers_production_strategies():
    cands = scan_strategies(STRATEGY)
    assert len(cands) >= 10  # the example book ships 16 strategies
    slugs = {c.slug for c in cands}
    assert "sma_trend" in slugs
    # production scan excludes baselines
    assert all(c.category != "Baseline" for c in cands)


def test_baselines_included_when_requested(tmp_path):
    import shutil
    src = STRATEGY / "trend_following" / "sma_trend"
    shutil.copytree(src, tmp_path / "trend_following" / "sma_trend")
    shutil.copytree(src, tmp_path / "Baseline" / "trend_following" / "sma_trend")
    prod = scan_strategies(tmp_path)
    allc = scan_strategies(tmp_path, include_baseline=True)
    assert len(prod) == 1
    assert len(allc) > len(prod)


def test_candidate_metrics_match_rigor():
    cands = {c.slug: c for c in scan_strategies(STRATEGY)}
    c = cands["sma_trend"]
    assert isinstance(c.returns, pd.Series) and len(c.returns) > 1000
    assert c.sharpe > 0 and np.isfinite(c.sharpe)
    assert np.isfinite(c.cagr) and c.max_drawdown <= 0
    assert c.verdict in {"PROMOTE", "CONDITIONAL", "REJECT", "UNKNOWN"}
    # oos_sharpe falls back to full Sharpe when there's no CPCV/walk-forward
    assert np.isfinite(c.oos_sharpe)


def test_read_returns_csv_rejects_short_series(tmp_path):
    p = tmp_path / "x_returns.csv"
    pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10),
                  "returns": [0.001] * 10}).to_csv(p, index=False)
    assert read_returns_csv(p) is None  # < 30 obs


def test_candidate_is_hedge_heuristic():
    idx = pd.bdate_range("2015-01-01", periods=300)
    flat = pd.Series(np.zeros(300) + 1e-4, index=idx)
    assert StrategyCandidate("My Hedge Overlay", REPO, flat).is_hedge
    assert not StrategyCandidate("Carbon Credits", REPO, flat).is_hedge


@pytest.mark.parametrize("include_baseline", [False, True])
def test_scan_is_deterministic(include_baseline):
    a = scan_strategies(STRATEGY, include_baseline=include_baseline)
    b = scan_strategies(STRATEGY, include_baseline=include_baseline)
    assert [c.slug for c in a] == [c.slug for c in b]
