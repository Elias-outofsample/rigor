"""Tests for the unified optimisation pipeline + calibration profiles.

Covers the three contracts the orchestrator must hold:
  * the pipeline runs the existing stages end-to-end, deterministically;
  * ``discovery`` admits at least as much as ``deployment`` (profile floors
    genuinely drive selection / the deployment gate);
  * the legacy ``optimize_strategy`` path is untouched (backward-compat).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fake_data import FakeDataLoader  # tests/ is on sys.path under pytest

from rigor.optimize import optimize_strategy, run_pipeline
from rigor.optimize.calibration import (
    DEPLOYMENT,
    DISCOVERY,
    CalibrationProfile,
    get_profile,
)
from rigor.optimize.pipeline import PipelineResult

# A synthetic strategy whose returns depend on its params (so ranking is real)
# and whose best config has a *borderline* edge: the per-config Sharpe lands
# between the discovery floor (0.30) and the deployment floor (0.80), so the
# permissive discovery profile selects a winner while the strict deployment
# profile rejects the grid — the differentiation the profiles exist to produce.
_GRID_STRATEGY = '''
from __future__ import annotations
import numpy as np
import pandas as pd
from rigor.strategy import StrategyBase, StrategyConfig

class Strategy(StrategyBase):
    def __init__(self, config, data):
        super().__init__(config); self.data = data
    def build_cache(self):
        px = self.data.prices("SPY", start=self.config.start_date, end=self.config.end_date)
        return {"dates": pd.DatetimeIndex(px["date"])}
    def param_grid(self):
        return {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}
    def default_params(self):
        return {"a": 1, "b": 1}
    def run_backtest(self, cache, params):
        dates = cache["dates"]; n = len(dates)
        # A shared market factor (same seed for every config) makes the configs
        # correlated, so the in-sample ranking is stable out-of-sample (low PBO)
        # and the grid is *selectable*. A small per-param drift on top gives the
        # best config a borderline Sharpe ~0.5 — above the discovery floor (0.30)
        # but below the deployment floor (0.80).
        market = np.random.default_rng(0).standard_normal(n)
        idio = np.random.default_rng(params["a"] * 100 + params["b"]).standard_normal(n)
        mu = 0.00009 * params["a"] - 0.00002 * params["b"]
        r = mu + 0.008 * market + 0.003 * idio
        return pd.Series(r, index=dates)

def build(config, data):
    return Strategy(config, data)
'''

_CONFIG = {
    "name": "Grid", "slug": "grid_strat", "category": "other",
    "start_date": "2010-01-01", "end_date": None, "initial_capital": 100000.0,
    "commission_bps": 0.0, "long_short": "long", "rebalance_freq": "daily",
    "role": "alpha", "status": "idle", "version": "v1", "extra": {},
}


@pytest.fixture
def grid_strategy(tmp_path):
    d = tmp_path / "grid_strat"
    d.mkdir()
    (d / "strategy.py").write_text(_GRID_STRATEGY, encoding="utf-8")
    (d / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    return d


# --- calibration profiles --------------------------------------------------

def test_profiles_resolve_by_name_and_default_to_discovery():
    assert get_profile("discovery") is DISCOVERY
    assert get_profile("deployment") is DEPLOYMENT
    assert get_profile("DEPLOYMENT") is DEPLOYMENT      # case-insensitive
    assert get_profile(None) is DISCOVERY               # None -> permissive default
    assert get_profile("nonsense") is DISCOVERY         # unknown -> permissive default


def test_deployment_floors_are_strictly_tighter_than_discovery():
    d, p = DISCOVERY, DEPLOYMENT
    # every selection-ladder floor is at least as strict under deployment
    assert p.sharpe_floor >= d.sharpe_floor
    assert p.psr_floor >= d.psr_floor
    assert p.cagr_floor >= d.cagr_floor
    assert p.max_dd_floor >= d.max_dd_floor      # less negative = stricter wipe-out floor
    assert p.pbo_max <= d.pbo_max                # lower PBO ceiling = stricter
    # and the deployment-readiness gate is stricter too
    assert p.dsr_min >= d.dsr_min
    assert p.min_active_bars >= d.min_active_bars
    assert d.blocking_verdicts <= p.blocking_verdicts   # deployment blocks a superset


def test_profile_projects_onto_robust_floors():
    floors = DEPLOYMENT.robust_floors()
    assert floors.sharpe == DEPLOYMENT.sharpe_floor
    assert floors.psr == DEPLOYMENT.psr_floor
    assert floors.pbo == DEPLOYMENT.pbo_max
    assert isinstance(DEPLOYMENT.as_dict(), dict)


def test_profile_with_overrides_is_a_copy():
    tweaked = DISCOVERY.with_overrides(sharpe_floor=0.9)
    assert tweaked.sharpe_floor == 0.9
    assert DISCOVERY.sharpe_floor == 0.30      # original is frozen / unchanged
    assert isinstance(tweaked, CalibrationProfile)


# --- pipeline end-to-end ---------------------------------------------------

def test_pipeline_runs_every_stage(grid_strategy):
    res = run_pipeline(grid_strategy, profile="discovery", data=FakeDataLoader(), write=False)
    assert isinstance(res, PipelineResult)
    s = res.stages
    assert s["configs_evaluated"] >= 2
    # the existing stages are all present in the structured result
    for stage in ("overfit", "cpcv", "verdict", "holdout", "falsification", "ranking"):
        assert stage in s, f"missing stage {stage}"
    assert s["best_sharpe"] == s["ranking"][0]["sharpe"]
    assert res.profile == "discovery"
    assert res.outcome in {"ADMIT", "WITHHOLD"}
    # the gate ties to the shared verdict
    assert s["verdict"]["verdict"] in {"PROMOTE", "CONDITIONAL", "REJECT"}
    assert {"selection", "deflated_sharpe", "activity", "verdict"} == {
        c["name"] for c in res.gate["checks"]}


def test_pipeline_deployment_outcome_is_promotion_language(grid_strategy):
    res = run_pipeline(grid_strategy, profile="deployment", data=FakeDataLoader(), write=False)
    assert res.profile == "deployment"
    assert res.outcome in {"PROMOTE", "REJECT"}
    assert res.deploy_ready == (res.outcome == "PROMOTE")


def test_discovery_admits_more_than_deployment(grid_strategy):
    """The whole point of the profiles: on a *borderline* winner the permissive
    discovery profile selects + admits, while the strict deployment profile
    rejects the grid under its tighter Sharpe / PBO floors."""
    disc = run_pipeline(grid_strategy, profile="discovery", data=FakeDataLoader(), write=False)
    dep = run_pipeline(grid_strategy, profile="deployment", data=FakeDataLoader(), write=False)
    # discovery keeps the borderline winner; deployment's stricter floors reject it
    assert disc.selection_priority == "ROBUST_PERF"
    assert disc.deploy_ready is True and disc.outcome == "ADMIT"
    assert dep.selection_priority == "REJECTED"
    assert dep.deploy_ready is False and dep.outcome == "REJECT"
    # discovery is never stricter than deployment
    assert disc.gate["n_failed"] <= dep.gate["n_failed"]
    # and deployment never deploys something discovery withheld
    if not disc.deploy_ready:
        assert not dep.deploy_ready


def test_discovery_strictly_admits_a_borderline_winner():
    """A hand-built borderline winner: clears the discovery floors, fails the
    deployment ones — proving the profiles actually differentiate."""
    from rigor.optimize.pipeline import _evaluate_gate

    # A report whose winner is a modest edge: DSR 0.55, CONDITIONAL verdict,
    # 40 active bars. Discovery (dsr>=0.5, bars>=20, blocks only REJECT) passes;
    # deployment (dsr>=0.6, bars>=100, blocks CONDITIONAL) fails on all three.
    report = {
        "selection": {"priority": "ROBUST_PERF", "selected_metrics": {"active_bars": 40}},
        "overfit": {"dsr": 0.55},
        "verdict": {"verdict": "CONDITIONAL"},
    }
    disc = _evaluate_gate(report, DISCOVERY)
    dep = _evaluate_gate(report, DEPLOYMENT)
    assert disc["passed"] is True
    assert dep["passed"] is False
    assert dep["n_failed"] >= disc["n_failed"]


def test_pipeline_is_deterministic(grid_strategy):
    a = run_pipeline(grid_strategy, profile="deployment", data=FakeDataLoader(), write=False)
    b = run_pipeline(grid_strategy, profile="deployment", data=FakeDataLoader(), write=False)
    assert a.outcome == b.outcome
    assert a.selected_params == b.selected_params
    assert a.stages["ranking"] == b.stages["ranking"]
    assert a.gate == b.gate


def test_pipeline_rejects_a_grid_with_no_robust_edge(grid_strategy, monkeypatch):
    """When the profile floors reject the whole grid, the pipeline reports a
    REJECTED selection and an un-deployable outcome (never crashes)."""
    # An impossible Sharpe floor rejects every config.
    harsh = DEPLOYMENT.with_overrides(sharpe_floor=99.0)
    res = run_pipeline(grid_strategy, profile=harsh, data=FakeDataLoader(), write=False)
    assert res.selection_priority == "REJECTED"
    assert res.selected_params is None
    assert res.deploy_ready is False
    assert res.outcome == "REJECT"


def test_pipeline_writes_report_when_requested(grid_strategy, tmp_path):
    res = run_pipeline(grid_strategy, profile="discovery", data=FakeDataLoader(), write=True)
    art = grid_strategy / "artifacts" / f"{res.slug}_pipeline_discovery.json"
    assert art.is_file()
    payload = json.loads(art.read_text(encoding="utf-8"))
    assert payload["profile"] == "discovery"
    assert "pipeline_gate" in payload


def test_pipeline_raises_without_a_multivalue_grid(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    src = _GRID_STRATEGY.replace(
        'return {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}', 'return {"a": [1]}'
    ).replace('return {"a": 1, "b": 1}', 'return {"a": 1}')
    (single / "strategy.py").write_text(src, encoding="utf-8")
    cfg = dict(_CONFIG, slug="single")
    (single / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="nothing to optimise"):
        run_pipeline(single, profile="discovery", data=FakeDataLoader(), write=False)


# --- backward-compatibility: the legacy path is untouched ------------------

def test_legacy_optimize_strategy_unchanged(grid_strategy):
    """The pipeline is additive: the old optimize_strategy entry point behaves
    exactly as before (no profile, default select_mode)."""
    r = optimize_strategy(grid_strategy, data=FakeDataLoader(), write=False)
    assert r["select_mode"] == "sharpe"            # legacy default
    assert "profile" not in r                       # the legacy report has no profile key
    assert r["selection"]["priority"] in {"TOP_SHARPE", "ROBUST_PERF", "REJECTED"}
    assert r["best_sharpe"] == r["ranking"][0]["sharpe"]


def test_pipeline_winner_matches_legacy_search(grid_strategy):
    """The pipeline's search/ranking is the same authoritative ordering the
    legacy optimiser uses — same top config by Sharpe."""
    legacy = optimize_strategy(grid_strategy, data=FakeDataLoader(), write=False)
    res = run_pipeline(grid_strategy, profile="discovery", data=FakeDataLoader(), write=False)
    # both rank by Sharpe desc on the same deterministic backtests
    assert res.stages["ranking"][0]["params"] == legacy["ranking"][0]["params"]
    assert np.isclose(res.stages["ranking"][0]["sharpe"], legacy["ranking"][0]["sharpe"])
