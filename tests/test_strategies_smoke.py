"""Smoke test: every strategy in the catalog must RUN (not just be shaped right).

For each ``strategies/**`` folder we build the strategy against a deterministic fake
DataLoader (no key, no network, no snapshot) and run its backtest, asserting it
produces a non-empty returns series without raising. A representative subset also
generates its thesis PDF (the slow, reportlab path).

This catches the class of bug that lint + unit tests miss: a strategy whose folder
is perfectly shaped but whose code crashes when actually run (e.g. a config/engine
mismatch, or a thesis that chokes on a non-ASCII name). It does NOT judge whether a
strategy is any *good* — only that it isn't *broken*.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fake_data import FakeDataLoader  # tests/ is on sys.path under pytest

from rigor.project.run import load_config, load_strategy_module

REPO = Path(__file__).resolve().parents[1]
STRAT_ROOT = REPO / "strategies"
STRATEGY_DIRS = sorted(p.parent for p in STRAT_ROOT.glob("**/config.json"))


def _sid(p: Path) -> str:
    return str(p.relative_to(STRAT_ROOT)).replace(os.sep, "/")


# strategies the smoke test should also push through the (slow) thesis generator;
# one per distinct data path (equity universe, multi-asset, intraday futures, crypto, ETF).
THESIS_SAMPLE = {
    "mean_reversion/rsi2_mr", "trend_following/global_tsmom", "intraday/vwap_regime_route_es",
    "crypto/dual_ema_crypto", "carry/credit_carry_rotation",
}


@pytest.fixture(autouse=True)
def _chdir_repo_root():
    """Run from the repo root so strategies that read committed files
    (universe.json, membership.json, ...) resolve their relative paths."""
    prev = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(prev)


def test_catalog_is_discovered():
    # Guard against a path/glob mistake silently testing nothing.
    assert len(STRATEGY_DIRS) >= 16, f"expected the example book, found {len(STRATEGY_DIRS)}"


@pytest.mark.parametrize("sdir", STRATEGY_DIRS, ids=[_sid(p) for p in STRATEGY_DIRS])
def test_strategy_runs(sdir: Path):
    config, raw = load_config(sdir)
    module = load_strategy_module(sdir, raw.get("slug") or sdir.name)
    strategy = module.build(config, FakeDataLoader())
    result = strategy.backtest()
    returns = result.returns
    assert isinstance(returns, pd.Series), f"{_sid(sdir)} did not return a returns Series"
    assert len(returns) > 0, f"{_sid(sdir)} produced an empty returns series"
    assert returns.notna().any(), f"{_sid(sdir)} returns are all NaN"


@pytest.mark.parametrize("sid", sorted(THESIS_SAMPLE))
def test_thesis_generates(sid: str, tmp_path: Path):
    sdir = STRAT_ROOT / sid
    if not (sdir / "config.json").exists():
        pytest.skip(f"{sid} not present")
    from rigor import thesis_card
    config, raw = load_config(sdir)
    module = load_strategy_module(sdir, raw["slug"])
    strategy = module.build(config, FakeDataLoader())
    result = strategy.backtest()
    out = thesis_card.write_thesis(result, tmp_path, raw["slug"],
                                   name=config.name, category=raw.get("category"),
                                   as_of="2026-06-09")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_thesis_is_deterministic(tmp_path):
    """The same result generates a byte-identical thesis PDF twice (no git churn)."""
    sdir = STRAT_ROOT / "mean_reversion" / "rsi2_mr"
    config, raw = load_config(sdir)
    module = load_strategy_module(sdir, raw["slug"])
    result = module.build(config, FakeDataLoader()).backtest()

    from rigor import thesis_card
    kw = {"name": config.name, "category": raw.get("category"), "as_of": "2026-06-09"}
    p1 = thesis_card.write_thesis(result, tmp_path / "a", raw["slug"], **kw)
    p2 = thesis_card.write_thesis(result, tmp_path / "b", raw["slug"], **kw)
    assert Path(p1).read_bytes() == Path(p2).read_bytes()
