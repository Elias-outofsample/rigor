"""Tests for rigor.analysis.scenario_response — market-regime response curves."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.analysis import scenario_response as sr


@pytest.fixture
def rng():
    return np.random.default_rng(99)


def _bench(rng, n=1600):
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


# ---------------------------------------------------------------------------
# scenario_response — known behavioural cases
# ---------------------------------------------------------------------------


def test_defensive_strategy_verdict(rng):
    b = _bench(rng)
    defensive = pd.Series(-0.5 * b.to_numpy() + rng.normal(0.0003, 0.004, len(b)),
                          index=b.index)
    resp = sr.scenario_response(defensive, b)
    # Outperforms when the market is down.
    assert resp["market_down"]["ann_return"] > resp["market_up"]["ann_return"]
    assert resp["market_up"]["beta"] < 0  # inverse exposure
    assert "defensive" in resp["verdict"]


def test_directional_strategy_verdict(rng):
    b = _bench(rng)
    direc = pd.Series(1.2 * b.to_numpy() + rng.normal(0.0, 0.003, len(b)),
                      index=b.index)
    resp = sr.scenario_response(direc, b)
    assert resp["market_up"]["ann_return"] > resp["market_down"]["ann_return"]
    assert resp["market_up"]["beta"] > 0.5
    assert "directional" in resp["verdict"] or "trend-following" in resp["verdict"]


def test_response_structure(rng):
    b = _bench(rng)
    strat = pd.Series(rng.normal(0.0003, 0.008, len(b)), index=b.index)
    resp = sr.scenario_response(strat, b)
    for key in ("market_up", "market_down", "trend_up", "trend_down",
                "vol_low", "vol_high", "drawdown_response", "return_buckets",
                "verdict", "periods_per_year"):
        assert key in resp
    assert resp["vol_proxy"] == "benchmark_rolling_std"
    assert resp["periods_per_year"] == 252


def test_response_with_vix_proxy(rng):
    b = _bench(rng)
    strat = pd.Series(rng.normal(0.0003, 0.008, len(b)), index=b.index)
    vix = pd.Series(20.0 + 5.0 * np.abs(rng.normal(0.0, 1.0, len(b))), index=b.index)
    resp = sr.scenario_response(strat, b, vix=vix)
    assert resp["vol_proxy"] == "VIX"
    assert resp["vol_high"]["n_obs"] > 0


def test_sector_rotation_optional(rng):
    b = _bench(rng)
    strat = pd.Series(0.5 * b.to_numpy() + rng.normal(0.0, 0.004, len(b)),
                      index=b.index)
    sectors = pd.DataFrame(
        {"tech": b.to_numpy() * 1.5, "util": b.to_numpy() * 0.3}, index=b.index
    )
    resp = sr.scenario_response(strat, b, sector_returns=sectors)
    assert "sector_response" in resp
    assert resp["best_sector"] in {"tech", "util"}
    assert resp["worst_sector"] in {"tech", "util"}


# ---------------------------------------------------------------------------
# market_drawdown_response
# ---------------------------------------------------------------------------


def test_market_drawdown_response_buckets(rng):
    b = _bench(rng, 2000)
    # Construct a benchmark close with a clear deep drawdown.
    close = (1.0 + b).cumprod()
    close.iloc[800:1000] *= 0.7  # force a >20% drawdown window
    strat = pd.Series(rng.normal(0.0, 0.01, len(b)), index=b.index)
    out = sr.market_drawdown_response(strat, close)
    assert set(out) == {"mkt_dd_ge_5pct", "mkt_dd_ge_10pct", "mkt_dd_ge_20pct"}
    deep = out["mkt_dd_ge_20pct"]
    assert deep["n_days"] >= 5
    assert "strat_mean" in deep and "strat_hit_rate" in deep


def test_market_drawdown_response_thin_bucket(rng):
    b = _bench(rng, 400)
    close = (1.0 + b).cumprod()  # mild, unlikely to hit -20%
    strat = pd.Series(rng.normal(0.0, 0.01, len(b)), index=b.index)
    out = sr.market_drawdown_response(strat, close)
    # The deepest bucket likely has too few days → only n_days reported.
    assert "n_days" in out["mkt_dd_ge_20pct"]


# ---------------------------------------------------------------------------
# market_return_buckets
# ---------------------------------------------------------------------------


def test_return_buckets_curve(rng):
    b = _bench(rng, 1500)
    strat = pd.Series(b.to_numpy() + rng.normal(0.0, 0.003, len(b)), index=b.index)
    out = sr.market_return_buckets(strat, b, n_buckets=5)
    assert out["n_buckets"] == 5
    means = [v["bench_mean"] for v in out["buckets"].values()]
    assert means == sorted(means)  # buckets ordered by benchmark return


def test_return_buckets_too_small_returns_empty(rng):
    b = _bench(rng, 20)
    strat = pd.Series(rng.normal(0.0, 0.01, 20), index=b.index)
    assert sr.market_return_buckets(strat, b, n_buckets=7) == {}


# ---------------------------------------------------------------------------
# realized_vol
# ---------------------------------------------------------------------------


def test_realized_vol(rng):
    b = _bench(rng)
    rv = sr.realized_vol(b)
    assert isinstance(rv, pd.Series)
    assert rv.notna().sum() > 0
    assert (rv.dropna() >= 0).all()
    # un-annualised is smaller than annualised
    rv_raw = sr.realized_vol(b, annualise=False)
    assert rv_raw.dropna().mean() < rv.dropna().mean()


# ---------------------------------------------------------------------------
# scenario_verdict & degenerate inputs
# ---------------------------------------------------------------------------


def test_verdict_neutral_on_empty_dict():
    assert sr.scenario_verdict({}) == "regime-neutral"


def test_verdict_handles_nan():
    resp = {"market_up": {"ann_return": float("nan")},
            "market_down": {"ann_return": float("nan")}}
    assert sr.scenario_verdict(resp) == "regime-neutral"


def test_short_series_no_raise(rng):
    b = _bench(rng, 4)
    strat = pd.Series([0.01, -0.02, 0.015, -0.01], index=b.index)
    resp = sr.scenario_response(strat, b)
    assert np.isnan(resp["market_up"]["ann_return"])
    assert resp["verdict"] == "regime-neutral"


def test_array_inputs_without_datetime_index(rng):
    """Non-DatetimeIndex input falls back to the default trading-day scale."""
    b = pd.Series(rng.normal(0.0004, 0.01, 600))
    strat = pd.Series(rng.normal(0.0, 0.008, 600))
    resp = sr.scenario_response(strat, b)
    assert resp["periods_per_year"] == 252
    assert resp["n_obs"] == 600
