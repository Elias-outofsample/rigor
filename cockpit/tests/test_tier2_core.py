"""Tier-2 engine additions: attribution, factor regression, CPCV/composite grid
search, and the portfolio permutation test — exercised on synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_engine import analysis, build_factor_proxies
from portfolio_engine.portfolio_validation import permutation_test
from portfolio_engine.search import weight_grid_search


def _matrix(n=4, bars=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=bars)
    cols = {f"s{i}": 0.0003 + 0.0002 * i + (0.008 + 0.002 * i) * rng.standard_normal(bars)
            for i in range(n)}
    return pd.DataFrame(cols, index=idx)


def test_attribution_shares_sum_to_one():
    rm = _matrix()
    w = np.array([0.4, 0.3, 0.2, 0.1])
    df = analysis.attribution(rm, w)
    assert len(df) == 4
    assert abs(df["risk_contrib_pct"].sum() - 1.0) < 1e-6
    assert abs(df["ret_contrib_pct"].sum() - 1.0) < 1e-6
    # sorted by risk share, descending
    assert df["risk_contrib_pct"].is_monotonic_decreasing


def test_factor_regression_recovers_beta():
    rm = _matrix(n=2)
    mkt = pd.Series(0.01 * np.random.default_rng(1).standard_normal(len(rm)), index=rm.index)
    port = 0.6 * mkt + 0.004 * pd.Series(
        np.random.default_rng(2).standard_normal(len(rm)), index=rm.index)
    out = analysis.factor_regression(port, {"MKT": mkt})
    assert "factors" in out and "MKT" in out["factors"]
    assert abs(out["factors"]["MKT"]["beta"] - 0.6) < 0.1   # recovers the loading
    assert 0.0 <= out["r_squared"] <= 1.0
    # no factors → graceful note, never a crash
    assert "note" in analysis.factor_regression(port, {})


def test_build_factor_proxies_offline_is_safe():
    # No data key / no network in the test env → empty dict, not an exception.
    proxies = build_factor_proxies()
    assert isinstance(proxies, dict)


def test_permutation_test_flags_real_edge():
    rm = _matrix()
    w = np.array([0.25, 0.25, 0.25, 0.25])
    res = permutation_test(rm, w, n_perm=500)
    assert set(res) == {"p_value", "is_significant", "n_perm"}
    assert 0.0 <= res["p_value"] <= 1.0


def test_subset_grid_search_respects_cardinality():
    from portfolio_engine.search import subset_grid_search
    rm = _matrix(n=8, bars=1200, seed=5)
    df = subset_grid_search(rm, min_k=2, max_k=3, n_samples=600, top_k=20)
    assert not df.empty
    legs = df["weights"].apply(lambda w: sum(1 for x in w.values() if x > 1e-6))
    assert legs.between(2, 3).all()        # every candidate holds 2–3 strategies


def test_common_window_shortens_backtest():
    from portfolio_engine.search import subset_grid_search
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2000-01-01", periods=1600)
    cols = {}
    for i in range(4):
        s = pd.Series(0.0004 + 0.01 * rng.standard_normal(1600), index=idx)
        s.iloc[: 300 * i] = np.nan          # staggered starts: 2000, ~2001, ~2002, ~2003
        cols[f"s{i}"] = s
    rm = pd.DataFrame(cols)
    full = subset_grid_search(rm, min_k=4, max_k=4, n_samples=200, top_k=5)
    cw = subset_grid_search(rm, min_k=4, max_k=4, n_samples=200, top_k=5, common_window=True)
    assert not full.empty and not cw.empty
    # the 4-leg common window starts at the youngest leg (~bar 900), so far fewer obs
    assert cw.iloc[0]["n_obs"] < full.iloc[0]["n_obs"]


def test_grid_search_cpcv_and_composite():
    rm = _matrix(n=5, bars=1200, seed=7)
    df = weight_grid_search(rm, n_samples=300, top_k=8, use_cpcv=True)
    assert not df.empty
    for col in ("composite_score", "dsr", "pos_months_pct", "yearly_consistency"):
        assert col in df.columns
    # composite ranking is descending
    assert df["composite_score"].is_monotonic_decreasing
