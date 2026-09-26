"""Wave 3 analysis: edge classifier (#3), unified verdict (#1), tail-risk (#5),
regimes (#12), cost/capacity (#4), factor attribution + alpha robustness (#2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.analysis import (
    alpha_robustness,
    cost,
    edge_classifier,
    factor,
    regime,
    tail_risk,
    verdict,
)


def _ret(mu=0.0005, vol=0.01, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + vol * rng.standard_normal(n),
                     index=pd.bdate_range("2015-01-01", periods=n))


# --- edge classifier (#3) --------------------------------------------------

def test_edge_classifier_mean_reversion():
    diag = {"autocorr_lag1": -0.30, "ic_half_life": 3.0, "ou_half_life": 10.0,
            "ou_pvalue": 0.01, "n_valid_obs": 500}
    out = edge_classifier.classify_edge_source(diag, universe="equity_us")
    assert out["primary"] == "MEAN_REVERSION" and out["confidence"] in {"high", "medium"}


def test_edge_classifier_degenerate():
    out = edge_classifier.classify_edge_source({"autocorr_lag1": float("nan")})
    assert out["primary"] == "DEGENERATE_INPUT"


# --- unified verdict (#1) --------------------------------------------------

def test_verdict_full_vs_sparse():
    metrics = {"sharpe": 1.6, "n_obs": 2520, "periods_per_year": 252,
               "net_sharpe": 1.4, "total_trades": 300}
    ao = {"pbo": 0.2, "dsr_p_skill": 0.7, "haircut_pct": 0.3, "psr": 0.99,
          "permutation_p": 0.01}
    rob = {"has_cliff": False, "sensitivity_cv": 0.1, "wf_efficiency": 0.7,
           "sharpe_cv": 0.3, "has_decay": False, "min_period_sharpe": 0.2}
    v = verdict.compute_strategy_verdict(metrics, ao, rob)
    assert v.verdict in {"ROBUST", "MODERATE"} and 0 <= v.score <= 100
    # sparse report: only metrics → neutral gates dropped, still scored on what's present
    v2 = verdict.compute_strategy_verdict(metrics)
    assert any(g.neutral for g in v2.gates) and 0 <= v2.score <= 100


# --- tail risk (#5) --------------------------------------------------------

def test_tail_risk_evt_and_var_backtest():
    r = _ret(mu=0.0, vol=0.013, n=1500, seed=1)
    gpd = tail_risk.fit_gpd_tail(r)
    assert "var_99" in gpd and gpd["var_99"] <= 0
    kup = tail_risk.kupiec_pof_test(r.to_numpy(), var_quantile=0.05)
    assert kup["verdict"] in {"PASS", "FAIL_HIGH_BREACHES", "FAIL_LOW_BREACHES"}
    cc = tail_risk.christoffersen_cc_test(r.to_numpy(), var_quantile=0.05)
    assert "p_value_cc" in cc


def test_tail_risk_garch_falls_back_gracefully():
    # arch may be absent in CI → must not raise, must return a result dict.
    out = tail_risk.gjr_garch_fhs_cvar(_ret(n=400, seed=2), n_simulations=500)
    assert "es" in out and "fallback_used" in out
    if out["fallback_used"]:
        assert out["method"] == "historical_fallback"


# --- regimes (#12) ---------------------------------------------------------

def test_regime_rule_based_and_performance():
    r = _ret(seed=3)
    reg = regime.detect_market_vol_regime(r)
    assert set(reg.unique()) <= {0, 1, 2}
    perf = regime.compute_regime_performance(r, reg)
    assert perf and all("sharpe" in v for v in perf.values())
    tl = regime.compute_regime_timeline(reg)
    assert tl and all("duration" in seg for seg in tl)


def test_regime_hmm_optional():
    out = regime.detect_regimes_hmm(_ret(seed=4))
    # either a fit (hmmlearn present) or a clear error string (absent)
    assert "regimes" in out and ("n_regimes" in out or "error" in out)


# --- cost / capacity (#4) --------------------------------------------------

def test_cost_breakdown_and_capacity():
    c = cost.compute_realistic_costs(trade_value=1e6, adv=5e7, returns=_ret(seed=5),
                                     short_fraction=0.5, leverage=1.5)
    assert c["net_sharpe"] <= c["gross_sharpe"] and c["total_round_trip_bps"] > 0
    cap = cost.compute_strategy_capacity(10, universe="equity_us")
    assert cap["capacity_usd"] > 0
    agg = cost.aggregate_portfolio_capacity([1e6, 2e6, 3e6])
    assert agg >= 3e6


def test_cost_sensitivity_and_scaling():
    table = cost.compute_cost_sensitivity_table(1.5, 0.12, 200, 10)
    assert len(table) == 8 and table[0]["cost_bps"] == 0
    scal = cost.compute_capital_scaling(_ret(seed=6))
    assert scal["base_sharpe"] != 0 and scal["scaling_curve"]


# --- factor attribution + alpha robustness (#2) ----------------------------

def _factor_panel(returns, seed=7):
    rng = np.random.default_rng(seed)
    n = len(returns)
    mkt = 0.0004 + 0.009 * rng.standard_normal(n)
    return pd.DataFrame(
        {"Mkt-RF": mkt, "SMB": 0.005 * rng.standard_normal(n),
         "HML": 0.005 * rng.standard_normal(n), "RMW": 0.004 * rng.standard_normal(n),
         "CMA": 0.004 * rng.standard_normal(n), "Mom": 0.006 * rng.standard_normal(n),
         "RF": np.full(n, 0.00005)},
        index=returns.index)


def test_factor_regression_recovers_beta():
    base = _ret(mu=0.0003, n=1000, seed=8)
    fac = _factor_panel(base, seed=108)          # independent stream from base
    strat = base + 0.8 * fac["Mkt-RF"]           # beta ~0.8 to the market
    res = factor.compute_factor_regression(strat, fac, use_hac=True)
    assert "betas" in res and abs(res["betas"]["Mkt-RF"] - 0.8) < 0.15
    assert res["se_type"] == "HAC"
    vif = factor.compute_vif(fac.drop(columns=["RF"]))
    assert all(np.isfinite(v) or np.isnan(v) for v in vif.values())


def test_progressive_alpha_and_verdict():
    base = _ret(mu=0.0006, n=1000, seed=9)
    fac = _factor_panel(base, seed=109)          # independent stream from base
    strat = base + 0.5 * fac["Mkt-RF"]
    prog = factor.compute_progressive_alpha(strat, fac)
    assert "CAPM" in prog and "FF6" in prog
    v = alpha_robustness.alpha_vs_risk_premium_verdict(
        {k: {"alpha": p["alpha_cond"], "alpha_t_stat": p["t"]} for k, p in prog.items()})
    assert v["verdict"] in {"GENUINE_ALPHA", "RISK_PREMIUM", "MOMENTUM_FACTOR",
                            "MIXED", "NEGATIVE_ALPHA", "NO_EDGE", "INSUFFICIENT_DATA"}
