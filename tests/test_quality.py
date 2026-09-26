"""Tests for the compact quality badge (rigor.analysis.quality).

The badge grades already-computed gates into ``WEAK · FRAGILE · MODERATE ·
ROBUST``. These cover known-answer cases (a clearly-robust gate set scores
ROBUST; a clearly-overfit one scores WEAK/FRAGILE), the hard-cap distinctions
that are the additive value over the full verdict (deep drawdown caps a real
edge to MODERATE, near-ruin to FRAGILE, poor consistency to FRAGILE), and
degenerate / missing-key inputs that must return a neutral label without raising.
"""
from __future__ import annotations

from rigor.analysis.quality import (
    BADGE,
    LABELS,
    BadgeThresholds,
    badge_from_gates,
    strategy_badge,
)

# ---------------------------------------------------------------------------
# Known-answer gate sets
# ---------------------------------------------------------------------------

def test_robust_gate_set_scores_robust():
    """Low PBO, strong Sharpe, high consistency, passing holdout -> ROBUST."""
    label = badge_from_gates(
        pbo=0.05, full_sharpe=1.4, consistency=0.92, holdout_ok=True,
        max_drawdown=-0.20,
    )
    assert label == "ROBUST"


def test_overfit_gate_set_scores_weak_or_fragile():
    """High PBO, near-zero Sharpe, low consistency, failing holdout -> bottom band."""
    label = badge_from_gates(
        pbo=0.80, full_sharpe=0.05, consistency=0.30, holdout_ok=False,
        max_drawdown=-0.50,
    )
    assert label in {"WEAK", "FRAGILE"}
    assert LABELS.index(label) <= 1


# ---------------------------------------------------------------------------
# Hard caps — the additive value over the full verdict
# ---------------------------------------------------------------------------

def test_deep_drawdown_caps_real_edge_to_moderate():
    """A real edge (3 gates) with a deep but not-ruinous DD is risky-not-fragile."""
    no_dd = badge_from_gates(
        pbo=0.10, full_sharpe=1.1, consistency=0.80, holdout_ok=True,
        max_drawdown=-0.30,
    )
    deep_dd = badge_from_gates(
        pbo=0.10, full_sharpe=1.1, consistency=0.80, holdout_ok=True,
        max_drawdown=-0.70,  # between cap_drawdown and cap_drawdown_ruin
    )
    assert no_dd == "ROBUST"
    assert deep_dd == "MODERATE"


def test_near_ruin_drawdown_caps_to_fragile():
    """A near-ruin drawdown forces FRAGILE even with otherwise-strong gates."""
    label = badge_from_gates(
        pbo=0.10, full_sharpe=1.1, consistency=0.80, holdout_ok=True,
        max_drawdown=-0.85,
    )
    assert label == "FRAGILE"


def test_near_zero_sharpe_caps_to_fragile():
    """A near-zero full Sharpe cannot exceed FRAGILE regardless of gate count."""
    label = badge_from_gates(
        pbo=0.05, full_sharpe=0.10, consistency=0.95, holdout_ok=True,
    )
    assert label == "FRAGILE"


def test_poor_consistency_caps_to_fragile():
    """Consistency below cap_consistency forces FRAGILE."""
    label = badge_from_gates(
        pbo=0.05, full_sharpe=1.5, consistency=0.20, holdout_ok=True,
        max_drawdown=-0.20,
    )
    assert label == "FRAGILE"


# ---------------------------------------------------------------------------
# Fallback gates
# ---------------------------------------------------------------------------

def test_wf_eff_fallback_when_consistency_absent():
    """When consistency is None, gate 2 falls back to wf_eff."""
    passing = badge_from_gates(
        pbo=0.05, full_sharpe=1.2, consistency=None, holdout_ok=True,
        wf_eff=0.80, max_drawdown=-0.20,
    )
    failing = badge_from_gates(
        pbo=0.05, full_sharpe=1.2, consistency=None, holdout_ok=True,
        wf_eff=0.10, max_drawdown=-0.20,
    )
    assert LABELS.index(passing) > LABELS.index(failing)


def test_oos_sharpe_fallback_when_holdout_absent():
    """When holdout_ok is None, gate 3 falls back to oos_sharpe."""
    passing = badge_from_gates(
        pbo=0.05, full_sharpe=1.2, consistency=0.80, holdout_ok=None,
        oos_sharpe=1.0, max_drawdown=-0.20,
    )
    failing = badge_from_gates(
        pbo=0.05, full_sharpe=1.2, consistency=0.80, holdout_ok=None,
        oos_sharpe=0.10, max_drawdown=-0.20,
    )
    assert LABELS.index(passing) > LABELS.index(failing)


# ---------------------------------------------------------------------------
# Degenerate / dict wrapper
# ---------------------------------------------------------------------------

def test_all_none_gates_yield_weak():
    """Every input absent -> no gates earned, no caps -> WEAK (worst)."""
    assert badge_from_gates(None, None, None, None) == "WEAK"


def test_nonfinite_inputs_do_not_raise():
    """NaN/inf gate values are coerced to absent, not propagated."""
    label = badge_from_gates(
        pbo=float("nan"), full_sharpe=float("inf"), consistency=float("nan"),
        holdout_ok=None, max_drawdown=float("-inf"),
    )
    assert label in LABELS


def test_strategy_badge_dict_in_dict_out():
    """The dict wrapper returns a label + matching rank from already-computed keys."""
    out = strategy_badge({
        "pbo": 0.05, "sharpe": 1.4, "consistency": 0.92,
        "holdout_ok": True, "max_drawdown": -0.20,
    })
    assert out["badge"] == "ROBUST"
    assert out["rank"] == LABELS.index("ROBUST")


def test_strategy_badge_empty_and_none_are_neutral():
    """An empty dict or None yields the neutral worst label without raising."""
    assert strategy_badge({})["badge"] == "WEAK"
    assert strategy_badge(None)["badge"] == "WEAK"
    assert strategy_badge({})["rank"] == 0


def test_strategy_badge_alias_keys():
    """full_sharpe/max_dd/wf_efficiency aliases are honoured."""
    out = strategy_badge({
        "pbo": 0.05, "full_sharpe": 1.3, "consistency": None,
        "wf_efficiency": 0.80, "holdout_ok": True, "max_dd": -0.25,
    })
    assert out["badge"] == "ROBUST"


def test_thresholds_are_configurable():
    """A stricter sharpe_min flips a borderline edge below ROBUST."""
    strict = BadgeThresholds(sharpe_min=2.0)
    label = badge_from_gates(
        pbo=0.05, full_sharpe=1.0, consistency=0.95, holdout_ok=True,
        max_drawdown=-0.20, cfg=strict,
    )
    # full_sharpe 1.0 < sharpe_min 2.0 -> gate 1 fails -> not ROBUST.
    assert label != "ROBUST"
    # Sanity: the default config DOES grade it ROBUST.
    assert badge_from_gates(
        pbo=0.05, full_sharpe=1.0, consistency=0.95, holdout_ok=True,
        max_drawdown=-0.20, cfg=BADGE,
    ) == "ROBUST"
