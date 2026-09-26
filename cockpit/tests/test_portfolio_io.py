"""Phase 3: saving a portfolio writes the portfolios/<name>/ standard and round-trips."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_engine import (
    AllocationConfig,
    PortfolioAllocator,
    PortfolioBacktester,
    StrategyCandidate,
    list_saved,
    load_portfolio,
    save_portfolio,
)
from portfolio_engine.portfolio_io import slugify

REPO = Path(__file__).resolve().parents[2]


def _synth(n=5, bars=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=bars)
    return [StrategyCandidate(f"Strat {i}", REPO, slug=f"strat_{i}", category="x",
                              returns=pd.Series(0.0004 + 0.0001 * i
                                                + 0.01 * rng.standard_normal(bars), index=idx))
            for i in range(n)]


def test_slugify():
    assert slugify("My Best Portfolio!") == "my_best_portfolio"
    assert slugify("  ") == "portfolio"


def test_save_writes_standard_and_round_trips(tmp_path):
    cands = _synth()
    w = PortfolioAllocator(cands, AllocationConfig(mode="inverse_vol")).allocate()
    res = PortfolioBacktester(cands, w, start_date=None).run()
    folder = save_portfolio("Core Diversified", cands, res.weights,
                            mode="inverse_vol", result=res, portfolio_root=tmp_path)

    assert folder == tmp_path / "core_diversified"
    assert (folder / "portfolio.json").exists()
    assert (folder / "artifacts" / "core_diversified_returns.csv").exists()
    assert (folder / "artifacts" / "core_diversified_summary.json").exists()
    assert (folder / "artifacts" / "core_diversified_report.html").exists()

    definition = json.loads((folder / "portfolio.json").read_text())
    assert definition["allocation_mode"] == "inverse_vol"
    assert {s["slug"] for s in definition["strategies"]} <= {f"strat_{i}" for i in range(5)}
    assert abs(sum(s["weight"] for s in definition["strategies"]) - 1.0) < 1e-3
    assert "sharpe" in definition["metrics"]

    # returns artifact reloads with the right shape
    rl = pd.read_csv(folder / "artifacts" / "core_diversified_returns.csv")
    assert list(rl.columns) == ["date", "returns"] and len(rl) > 100

    assert list_saved(tmp_path) == ["core_diversified"]
    assert load_portfolio("Core Diversified", tmp_path)["slug"] == "core_diversified"


def test_list_saved_empty(tmp_path):
    assert list_saved(tmp_path) == []
    assert list_saved(tmp_path / "nope") == []
