"""Runtime param_grid contract: every non-exempt strategy must be RUN-ABLE through
the optimiser, not merely statically gate-compliant.

The static optimize_gate parses source; it cannot see a grid that (a) is inherited
from a shared engine, (b) is computed at runtime, or (c) is shadowed by a duplicate
method. This test instantiates every strategy on the deterministic FakeDataLoader
(no key, no network) and asserts ``param_grid()`` actually returns a sweepable
``dict[str, list]`` — at least one axis with >= 2 type-consistent values — so the
optimiser has something real to sweep. Strategies carrying ``optimize_exempt_reason``
are skipped (they are deliberately not optimisable).

This is the runtime companion to ``rigor.project.optimize_gate`` and closes the
static-vs-runtime gap: a strategy that passes the gate but whose grid is empty /
single-valued / errors at runtime fails HERE.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fake_data import FakeDataLoader

from rigor.project.run import load_config, load_strategy_module

REPO = Path(__file__).resolve().parents[1]
STRAT_ROOT = REPO / "strategies"
STRATEGY_DIRS = sorted(p.parent for p in STRAT_ROOT.glob("**/config.json")
                       if "Baseline" not in p.parts)


def _sid(p: Path) -> str:
    return str(p.relative_to(STRAT_ROOT)).replace(os.sep, "/")


def _is_exempt(sdir: Path) -> bool:
    try:
        raw = json.loads((sdir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    reason = raw.get("optimize_exempt_reason")
    return isinstance(reason, str) and bool(reason.strip())


@pytest.fixture(autouse=True)
def _chdir_repo_root():
    prev = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(prev)


def test_catalog_discovered():
    assert len(STRATEGY_DIRS) >= 16, f"expected the example book, found {len(STRATEGY_DIRS)}"


@pytest.mark.parametrize("sdir", STRATEGY_DIRS, ids=[_sid(p) for p in STRATEGY_DIRS])
def test_param_grid_is_runtime_sweepable(sdir: Path):
    if _is_exempt(sdir):
        pytest.skip("exempt (optimize_exempt_reason set)")
    config, raw = load_config(sdir)
    module = load_strategy_module(sdir, raw.get("slug") or sdir.name)
    strategy = module.build(config, FakeDataLoader())
    grid = strategy.param_grid()
    sid = _sid(sdir)
    assert isinstance(grid, dict) and grid, \
        f"{sid}: param_grid() must return a non-empty dict, got {grid!r}"
    multi = 0
    for axis, values in grid.items():
        assert isinstance(axis, str) and axis, f"{sid}: bad axis name {axis!r}"
        assert isinstance(values, (list, tuple)) and values, \
            f"{sid}: axis {axis!r} must be a non-empty list"
        vals = list(values)
        if len(vals) >= 2:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
            assert numeric or len({type(v) for v in vals}) == 1, \
                f"{sid}: axis {axis!r} has inconsistent value types"
            multi += 1
    assert multi >= 1, \
        f"{sid}: param_grid() has no axis with >=2 values — nothing to sweep at runtime"
