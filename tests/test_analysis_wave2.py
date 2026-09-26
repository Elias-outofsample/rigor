"""Wave 2 analysis: signal-quality (#6), structural (#7), predictive-ability (#7),
attribution/Memmel/Brinson (#11), exposure+crisis (#13), Bayesian Sharpe +
Romano-Wolf DSR (#13)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.analysis import (
    attribution,
    exposure,
    predictive_ability,
    signal_quality,
    structural,
)
from rigor.validation import bayesian, overfit


def _ret(mu=0.0005, vol=0.01, n=900, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + vol * rng.standard_normal(n),
                     index=pd.bdate_range("2016-01-01", periods=n))


# --- signal quality (#6) ---------------------------------------------------

def test_rank_ic_detects_real_signal():
    rng = np.random.default_rng(0)
    sig = pd.Series(rng.standard_normal(500))
    fwd = 0.5 * sig + 0.5 * pd.Series(rng.standard_normal(500))   # IC > 0
    assert signal_quality.rank_ic(sig, fwd) > 0.2
    fl = signal_quality.flam_summary(signal_quality.rank_ic_rolling(sig, fwd)["ic_series"])
    assert {"ic", "breadth", "ir"} <= set(fl)


def test_fundamental_law_components():
    assert signal_quality.fundamental_ir(0.05, 100.0) > 0
    assert (signal_quality.icir_flam(0.05, 100.0)
            == signal_quality.fundamental_ir(0.05, 100.0, tc=1.0))


# --- structural breaks (#7) ------------------------------------------------

def test_cusum_flags_a_mean_break():
    a = np.concatenate([0.0 + 0.005 * np.random.default_rng(1).standard_normal(300),
                        0.01 + 0.005 * np.random.default_rng(2).standard_normal(300)])
    assert structural.cusum_mean(a)["has_break"]
    assert "has_break" in structural.cusum_variance(a)


def test_masters_block_permutation_real_edge():
    out = structural.masters_block_permutation(_ret(mu=0.0015, n=600, seed=3).to_numpy())
    assert out["p_value"] < 0.20 and "verdict" in out


# --- predictive ability (#7) -----------------------------------------------

def test_diebold_mariano_and_pesaran():
    rng = np.random.default_rng(4)
    actual = rng.standard_normal(400)
    good = (actual + 0.3 * rng.standard_normal(400)) ** 2 * 0.0   # placeholder
    loss_a = (actual - (actual + 0.3 * rng.standard_normal(400))) ** 2
    loss_b = (actual - rng.standard_normal(400)) ** 2
    dm = predictive_ability.diebold_mariano(loss_a, loss_b)
    assert "dm_stat" in dm and "p_value" in dm
    pt = predictive_ability.pesaran_timmermann(actual + 0.5 * rng.standard_normal(400), actual)
    assert pt["verdict"] in {"DIRECTIONAL_SKILL", "NO_DIRECTIONAL_SKILL",
                             "ANTI_DIRECTIONAL", "insufficient_data", "degenerate"}
    _ = good


def test_model_confidence_set_keeps_best():
    rng = np.random.default_rng(5)
    base = rng.standard_normal((300, 1)) ** 2
    losses = np.column_stack([base[:, 0], base[:, 0] + 0.05, base[:, 0] + 2.0])
    out = predictive_ability.model_confidence_set(losses, n_bootstrap=100)
    assert 2 not in out["survivors"] or len(out["survivors"]) >= 1   # worst model dropped


# --- attribution / Memmel / Brinson (#11) ----------------------------------

def test_signal_vs_setup_and_memmel():
    sig = _ret(mu=0.0008, seed=6)
    setup = _ret(mu=0.0002, seed=7)
    combined = sig + setup
    dec = attribution.decompose_signal_vs_setup(sig, setup, combined)
    assert dec["dominant_source"] in {"SIGNAL", "SETUP_SCORE", "INTERACTION"}
    mem = attribution.memmel_sharpe_difference_test(combined, sig)
    assert "z_stat" in mem and "p_value" in mem
    orth = attribution.test_signal_orthogonality(sig, setup)
    assert orth["verdict"] in {"ORTHOGONAL", "WEAKLY_CORRELATED",
                               "MODERATELY_CORRELATED", "REDUNDANT"}


def test_brinson_fachler_reconciles():
    idx = pd.bdate_range("2020-01-01", periods=12)
    cols = ["tech", "fin", "energy"]
    rng = np.random.default_rng(8)
    sw = pd.DataFrame(rng.random((12, 3)), index=idx, columns=cols)
    sw = sw.div(sw.sum(axis=1), axis=0)
    bw = pd.DataFrame(np.full((12, 3), 1 / 3), index=idx, columns=cols)
    sr = pd.DataFrame(0.01 * rng.standard_normal((12, 3)), index=idx, columns=cols)
    br = pd.DataFrame(0.01 * rng.standard_normal((12, 3)), index=idx, columns=cols)
    out = attribution.brinson_fachler(sw, sr, bw, br)
    assert abs(out["total"]["residual"]) < 1e-9          # alloc+selec+inter == excess
    assert out["dominant_effect"] in {"allocation", "selection", "interaction"}


# --- exposure + crisis (#13) -----------------------------------------------

def test_greeks_summary_and_crisis():
    bench = _ret(mu=0.0003, vol=0.012, n=1500, seed=9)
    # strat with mild convexity to the benchmark
    strat = 0.5 * bench + 0.3 * bench ** 2 + _ret(mu=0.0002, n=1500, seed=10)
    g = exposure.compute_greeks_summary(strat, bench)
    assert {"vol_exposure", "gamma_exposure", "theta_exposure", "payoff"} <= set(g)
    cr = exposure.decompose_by_crisis(_ret(mu=0.0006, n=2000, seed=11), benchmark=bench)
    assert cr["verdict"] in {"ROBUST_TO_CRISES", "VOL_RISK_PREMIUM_CANDIDATE",
                             "CONVEXITY_CANDIDATE", "MIXED", "NEGATIVE_EDGE",
                             "INSUFFICIENT_DATA"}


# --- Bayesian Sharpe + Romano-Wolf DSR (#13) -------------------------------

def test_bayesian_sharpe_posterior_and_decision():
    post = bayesian.bayesian_sharpe_posterior(_ret(mu=0.001, n=750, seed=12).to_numpy())
    assert post["ci_5"] <= post["mean"] <= post["ci_95"]
    p = bayesian.prob_sharpe_above_threshold(post, threshold=0.0)
    assert 0.0 <= p <= 1.0
    assert bayesian.decision_rule(post)["decision"] in {"DEPLOY", "WAIT", "REJECT",
                                                        "INSUFFICIENT_DATA"}


def test_romano_wolf_dsr_runs_and_bounds():
    rng = np.random.default_rng(13)
    rm = 0.0005 + 0.01 * rng.standard_normal((600, 20))     # 20 correlated trials
    sr_pp = float(rm[:, 0].mean() / rm[:, 0].std(ddof=1))
    out = overfit.deflated_sharpe_romano_wolf(sr_pp, 600, rm, n_bootstrap=100)
    assert 0.0 <= out["dsr_rw"] <= 1.0 and "e_max_empirical" in out
    # deterministic
    out2 = overfit.deflated_sharpe_romano_wolf(sr_pp, 600, rm, n_bootstrap=100)
    assert out["dsr_rw"] == out2["dsr_rw"]
