"""Wave 1 analysis: risk battery (#10), labeling (#8), look-ahead detector (#9)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.analysis import labeling, lookahead, risk


def _ret(mu=0.0005, n=900, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + 0.01 * rng.standard_normal(n),
                     index=pd.bdate_range("2016-01-01", periods=n))


# --- risk battery ----------------------------------------------------------

def test_risk_summary_keys_and_finite():
    s = risk.risk_summary(_ret())
    expected = {"var_95", "cvar_95", "cornish_fisher_var_95", "omega", "tail_ratio",
                "rachev", "gain_to_pain", "ulcer_index", "upi", "cdar_95",
                "pain_index", "sterling", "burke"}
    assert expected <= set(s)
    assert all(np.isfinite(v) for v in s.values())


def test_var_ordering_and_signs():
    r = _ret(mu=0.0, seed=1)
    assert risk.conditional_var(r) <= risk.value_at_risk(r) <= 0   # ES <= VaR <= 0
    assert risk.ulcer_index(r) >= 0 and risk.pain_index(r) >= 0
    assert risk.cdar(r) >= 0


def test_omega_above_one_for_positive_drift():
    assert risk.omega(_ret(mu=0.002, seed=2)) > 1.0   # gains dominate losses


def test_drawdown_events_and_underwater():
    r = _ret(seed=3)
    events = risk.top_drawdowns(r, n=5)
    assert events and all(e.depth <= 0 for e in events)
    assert events == sorted(events, key=lambda e: e.depth)        # deepest first
    tuw = risk.time_under_water(r)
    assert tuw["count"] > 0 and tuw["max"] >= tuw["median"]
    assert "is_underwater" in risk.current_underwater(r)


def test_regime_conditional_risk():
    r = _ret(seed=4)
    reg = pd.Series(np.where(np.arange(len(r)) % 2, "hi", "lo"), index=r.index)
    out = risk.regime_conditional_risk(r, reg)
    assert {"hi", "lo"} <= set(out) and "sharpe" in out["hi"]


# --- labeling (#8) ---------------------------------------------------------

def test_triple_barrier_labels():
    close = (1 + _ret(mu=0.0008, seed=5)).cumprod() * 100
    events = close.index[::20]
    df = labeling.triple_barrier_labels(close, events, pt_sl=(2.0, 1.0), num_days=20)
    assert set(df["label"].unique()) <= {-1, 0, 1}
    assert list(df.columns) == ["label", "ret", "t_exit"]


def test_meta_labels_and_weights():
    sig = pd.Series(np.sign(_ret(seed=6)).astype(int))
    sig.index = _ret(seed=6).index
    meta = labeling.meta_labels(sig, _ret(seed=7))
    assert set(meta.unique()) <= {0, 1}
    close = (1 + _ret(seed=8)).cumprod()
    w = labeling.sample_weights_uniqueness(close.index[::10], close, num_days=20)
    assert abs(w.sum() - len(w)) < 1e-6        # normalised to n_events
    idx = labeling.sequential_bootstrap_indices(w, seed=1)
    assert len(idx) == len(w) and idx.min() >= 0 and idx.max() < len(w)
    assert np.array_equal(idx, labeling.sequential_bootstrap_indices(w, seed=1))  # deterministic


# --- look-ahead detector (#9) ----------------------------------------------

def test_lookahead_flags_a_leaking_strategy():
    # a strategy whose return IS the (unshifted) future signal; shifting kills it
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(600)

    def bt(cache, params):
        shift = cache.get("_shift", 0)
        sig = np.roll(signal, shift) if shift else signal
        return 0.01 * sig * signal          # baseline: perfectly aligned -> high Sharpe

    out = lookahead.detect_lookahead(bt, {"signal": signal}, {}, n_random_trials=3)
    assert out["shift_supported"] and out["suspect"] and out["confidence"] == "high"


def test_lookahead_clean_strategy_not_flagged():
    rng = np.random.default_rng(1)
    base = 0.0015 + 0.01 * rng.standard_normal(600)   # a clear, real edge (Sharpe ~2)

    def bt(cache, params):
        return base                          # ignores _shift -> identical, no drop

    out = lookahead.detect_lookahead(bt, {}, {}, n_random_trials=3)
    assert not out["suspect"] and out["sharpe_drop_pct"] == 0.0
