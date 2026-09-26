"""Tests for the opt-in optimiser (rigor.optimize) and its data-snooping tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fake_data import FakeDataLoader  # tests/ is on sys.path under pytest

from rigor.optimize import optimize_strategy
from rigor.optimize.clustering import cluster_parameters
from rigor.optimize.enrichment import enrich_equity
from rigor.optimize.ensemble import build_ensemble
from rigor.optimize.falsification import noise_injection_test, phase_randomize
from rigor.optimize.grid import decimate, expand, refine_around, sample
from rigor.optimize.holdout import holdout_validate
from rigor.optimize.regime import label_vol_terciles, optimize_per_regime
from rigor.optimize.scoring import score_configs
from rigor.optimize.select import RobustFloors, config_metrics, select
from rigor.optimize.walkforward import walk_forward_opt
from rigor.validation import (
    hansen_spa_test,
    multiple_testing_adjustment,
    stepm_test,
    white_reality_check,
)

REPO = Path(__file__).resolve().parents[1]
SMA_TREND = REPO / "strategies" / "trend_following" / "sma_trend"

# A synthetic strategy with a *large* grid (25 configs) so the two-pass search and
# the ensemble/robust modes have something to chew on — the real book only has the
# 2-config sma_trend opted in. Its returns depend on the params so ranking is real.
_BIG_STRATEGY = '''
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
        rng = np.random.default_rng(params["a"] * 100 + params["b"])
        mu = 0.0003 * params["a"] - 0.0001 * params["b"]
        return pd.Series(mu + 0.01 * rng.standard_normal(len(cache["dates"])), index=cache["dates"])

def build(config, data):
    return Strategy(config, data)
'''

_BIG_CONFIG = {
    "name": "Big Grid", "slug": "big_grid", "category": "other",
    "start_date": "2010-01-01", "end_date": None, "initial_capital": 100000.0,
    "commission_bps": 0.0, "long_short": "long", "rebalance_freq": "daily",
    "role": "alpha", "status": "idle", "version": "v1", "extra": {},
}


@pytest.fixture
def big_grid_strategy(tmp_path):
    import json
    d = tmp_path / "big_grid"
    d.mkdir()
    (d / "strategy.py").write_text(_BIG_STRATEGY, encoding="utf-8")
    (d / "config.json").write_text(json.dumps(_BIG_CONFIG), encoding="utf-8")
    return d


# --- grid ------------------------------------------------------------------

def test_expand_is_cartesian_product():
    combos = expand({"a": [1, 2], "b": [10, 20, 30]})
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos and {"a": 2, "b": 30} in combos


def test_sample_caps_keeps_default_and_is_deterministic():
    combos = expand({"a": list(range(50)), "b": [1, 2, 3, 4]})  # 200 combos
    default = {"a": 0, "b": 1}
    s1 = sample(combos, 20, default=default)
    assert len(s1) == 20
    assert default in s1                       # the default config is always kept
    assert s1 == sample(combos, 20, default=default)   # deterministic (seeded)
    assert sample(combos[:10], 20, default=default) == combos[:10]  # no-op under the cap


def test_decimate_caps_grid_and_keeps_endpoints():
    grid = {"a": list(range(20)), "b": list(range(20))}  # 400 combos
    dec = decimate(grid, 64)
    assert len(expand(dec)) <= 64
    for axis in ("a", "b"):
        assert dec[axis][0] == grid[axis][0] and dec[axis][-1] == grid[axis][-1]  # span intact


def test_refine_around_keeps_neighbourhood():
    grid = {"a": [1, 2, 3, 4, 5]}
    fine = refine_around(grid, {"a": 3}, span=1)
    assert fine["a"] == [2, 3, 4]


# --- data-snooping tests ---------------------------------------------------

def _ar_series(mu, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + 0.01 * rng.standard_normal(n))


def test_white_and_hansen_flag_a_real_edge():
    # family of mostly-zero configs plus one clear winner vs a cash benchmark
    family = [_ar_series(0.004, seed=1)] + [_ar_series(0.0, seed=i) for i in range(2, 12)]
    wrc = white_reality_check(family, None, n_bootstrap=400)
    spa = hansen_spa_test(family, None, n_bootstrap=400)
    assert wrc["p_value"] < 0.05 and spa["p_consistent"] < 0.05
    assert wrc["is_significant"]


def test_snooping_is_not_significant_for_a_pure_noise_grid():
    # every config is zero-mean noise: the in-sample max must NOT look significant
    # once the snooping correction accounts for having searched the whole family
    family = [_ar_series(0.0, seed=i) for i in range(12)]
    wrc = white_reality_check(family, None, n_bootstrap=400)
    assert wrc["p_value"] > 0.10


def test_stepm_counts_rejections_and_handles_singleton():
    family = [_ar_series(0.004, seed=1)] + [_ar_series(0.0, seed=i) for i in range(2, 8)]
    rw = stepm_test(family, None, n_bootstrap=400)
    assert 0 <= rw["n_rejected"] <= rw["n_total"] == 7
    empty = white_reality_check([_ar_series(0.0)], None)["p_value"]
    assert empty != empty  # nan when the family has < 2 configs


def test_multiple_testing_adjustment_monotone_and_bounded():
    p = [0.001, 0.01, 0.04, 0.5]
    for method in ("bonferroni", "holm", "bh", "by"):
        adj = multiple_testing_adjustment(p, method=method)
        assert np.all(adj >= np.array(p) - 1e-9) and np.all(adj <= 1.0)
    with pytest.raises(ValueError):
        multiple_testing_adjustment(p, method="nope")


# --- selection -------------------------------------------------------------

def test_config_metrics_basic():
    m = config_metrics(_ar_series(0.001))
    assert {"sharpe", "cagr", "max_dd", "psr", "active_bars"} <= set(m)
    assert m["active_bars"] > 0


def test_select_sharpe_picks_top_and_robust_can_reject():
    rows = [{"params": {"a": i}, "sharpe": 2.0 - i, "returns": _ar_series(0.001 * (3 - i), seed=i)}
            for i in range(4)]
    assert select(rows, mode="sharpe")["index"] == 0
    # an overfit-prone grid (high PBO) is rejected outright in robust mode
    rej = select(rows, mode="robust", pbo=0.9)
    assert rej["priority"] == "REJECTED" and rej["index"] is None
    with pytest.raises(ValueError):
        select(rows, mode="bogus")


def test_robust_floors_reject_a_weak_grid():
    rows = [{"params": {"a": i}, "sharpe": -0.5, "returns": _ar_series(-0.001, seed=i)}
            for i in range(5)]
    out = select(rows, mode="robust", floors=RobustFloors(sharpe=0.3), pbo=0.1)
    assert out["priority"] == "REJECTED"


# --- ensemble --------------------------------------------------------------

def test_ensemble_combines_members():
    rows = [{"params": {"a": i}, "sharpe": 1.5 - 0.1 * i, "returns": _ar_series(0.0005, seed=i)}
            for i in range(8)]
    e = build_ensemble(rows, n_members=4)
    assert e["n_members"] == 4 and len(e["members"]) == 4
    assert "ensemble_sharpe" in e and "ensemble_vs_best" in e


# --- end-to-end orchestrator ----------------------------------------------

def test_optimize_reports_best_overfit_snooping_and_walk_forward():
    report = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False)
    assert report["configs_evaluated"] >= 2
    assert report["search"] == "one_pass"
    assert report["best_sharpe"] == report["ranking"][0]["sharpe"]
    assert "dsr" in report["overfit"]
    assert report["overfit"]["n_trials"] == report["configs_evaluated"]
    assert "data_snooping" in report and "walk_forward_opt" in report
    assert report["verdict"]["verdict"] in {"PROMOTE", "CONDITIONAL", "REJECT"}
    assert report["selection"]["priority"] in {"TOP_SHARPE", "ROBUST_PERF", "REJECTED"}


def test_optimize_two_pass_and_modes_run(big_grid_strategy):
    report = optimize_strategy(
        big_grid_strategy, data=FakeDataLoader(), write=False, two_pass=True, coarse_combos=8,
        wf_mode="rolling", select_mode="robust", ensemble=3, snoop_bootstrap=200)
    assert report["search"] == "two_pass"
    assert report["grid_size"] == 25
    assert report["configs_evaluated"] < 25  # coarse + a refined neighbourhood, not the full grid
    assert report["walk_forward_opt"].get("mode") in {"rolling", None}
    assert report["ensemble"]["n_members"] == 3
    assert report["selection"]["priority"] in {"ROBUST_PERF", "REJECTED"}


def test_optimize_is_deterministic():
    a = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False)
    b = optimize_strategy(SMA_TREND, data=FakeDataLoader(), write=False)
    assert a["best_params"] == b["best_params"]
    assert a["ranking"] == b["ranking"]
    assert a["data_snooping"] == b["data_snooping"]


# --- the full research pipeline stages -------------------------------------

def _trend_returns(n=600, mu=0.0006, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + 0.01 * rng.standard_normal(n),
                     index=pd.bdate_range("2015-01-01", periods=n))


def test_enrich_equity_keys_and_neutral():
    m = enrich_equity(_trend_returns())
    assert {"roc310", "stability", "k_ratio", "eta", "psr", "upi",
            "rolling_consistency", "sortino", "recency"} <= set(m)
    assert enrich_equity(_trend_returns(n=10))["roc310"] == 5.0  # neutral on short series


def test_holdout_split_and_degraded_flag():
    h = holdout_validate(_trend_returns())
    assert {"is_sharpe", "oos_sharpe", "holdout_sharpe", "degraded"} <= set(h)
    assert h["is_n"] + h["oos_n"] + h["holdout_n"] == len(_trend_returns())
    assert isinstance(h["degraded"], bool)


def test_phase_randomize_preserves_mean_var():
    x = _trend_returns().to_numpy()
    s = phase_randomize(x, np.random.default_rng(0))
    assert abs(s.mean() - x.mean()) < 1e-9
    assert abs(s.std() - x.std()) < 1e-6  # power spectrum preserved


def test_noise_injection_fragility_increases_with_noise():
    ni = noise_injection_test(_trend_returns(seed=1), ppy=252, n_trials=50)
    assert ni["base_sharpe"] > 0
    # decay is monotone: more noise -> lower mean Sharpe
    d = ni["decay_curve"]
    assert d[0.25] >= d[1.0]
    assert ni["fragility"] >= 0.0


def test_score_configs_ranks_by_composite():
    rows = [{"params": {"a": i}, "sharpe": 1.2 - 0.1 * i,
             "returns": _trend_returns(mu=0.0008 - 0.0001 * i, seed=i)}
            for i in range(8)]
    sc = score_configs(rows)
    assert "best_composite" in sc and sc["pillars_used"]
    assert sc["ranking"][0]["composite"] >= sc["ranking"][-1]["composite"]


def test_regime_terciles_and_per_regime():
    lab = label_vol_terciles(_trend_returns())
    assert set(lab.unique()) <= {0, 1, 2}
    rows = [{"params": {"a": i}, "sharpe": 1.0, "returns": _trend_returns(mu=0.0006, seed=i)}
            for i in range(5)]
    rg = optimize_per_regime(rows)
    assert "regime_coverage" in rg and rg["n_regimes"] >= 1


def test_clustering_returns_centers():
    rows = [{"params": {"a": i, "b": i % 3}, "sharpe": 1.5 - 0.05 * i,
             "returns": _trend_returns(seed=i)} for i in range(20)]
    cl = cluster_parameters(rows)
    assert cl["n_clusters"] >= 2 and len(cl["cluster_centers"]) == cl["n_clusters"]


def test_sobol_pick_covers_and_dedups():
    from rigor.validation.accel import sobol_pick
    idx = sobol_pick(1000, 50, seed=0)
    assert len(idx) == 50 and len(set(idx.tolist())) == 50
    assert idx.min() >= 0 and idx.max() < 1000
    assert np.array_equal(idx, sobol_pick(1000, 50, seed=0))  # deterministic
    assert np.array_equal(sobol_pick(10, 50), np.arange(10))  # no-op above the cap


def test_bootstrap_index_batch_valid_and_deterministic():
    from rigor.validation.accel import bootstrap_index_batch
    a = bootstrap_index_batch(200, 64, 10.0, seed=42)
    assert a.shape == (64, 200) and a.min() >= 0 and a.max() < 200
    assert np.array_equal(a, bootstrap_index_batch(200, 64, 10.0, seed=42))


def test_backends_identical_results():
    # the backend must be a speed knob, never a results knob
    from rigor.validation.accel import backends_available
    fam = [_ar_series(0.004, seed=1)] + [_ar_series(0.0, seed=i) for i in range(2, 12)]
    base = white_reality_check(fam, None, n_bootstrap=300, backend="numpy")["p_value"]
    for be in ("auto", "numba", "cuda"):
        if be in ("numba", "cuda") and not backends_available()[be]:
            continue
        assert white_reality_check(fam, None, n_bootstrap=300, backend=be)["p_value"] == base


def test_full_pipeline_runs_every_stage(big_grid_strategy):
    r = optimize_strategy(big_grid_strategy, data=FakeDataLoader(), write=False, full=True,
                          snoop_bootstrap=100)
    for stage in ("enrichment", "holdout", "falsification", "selection_score_5pillar",
                  "regime", "clustering", "ensemble"):
        assert stage in r, f"missing stage {stage}"
    assert r["default_in_grid"] is True


def test_walk_forward_purge_gap():
    """purge_gap drops the in-sample bars adjacent to OOS: a no-op at 0, shrinks the
    usable in-sample window when large, and rejects negatives (mirrors cpcv's purge)."""
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(1200, 8)),
        index=pd.bdate_range("2018-01-01", periods=1200),
    )
    base = walk_forward_opt(matrix, n_folds=5, mode="expanding")
    assert base["n_folds"] == 5
    # default purge is byte-identical to no purge
    assert walk_forward_opt(matrix, n_folds=5, mode="expanding", purge_gap=0) == base
    # a large gap shrinks an early fold's in-sample window below the usable floor
    purged = walk_forward_opt(matrix, n_folds=5, mode="expanding", purge_gap=190)
    assert purged["n_folds"] < base["n_folds"]
    with pytest.raises(ValueError):
        walk_forward_opt(matrix, n_folds=5, purge_gap=-1)
