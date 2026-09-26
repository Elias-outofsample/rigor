"""Tests for archetype threshold presets in compute_strategy_verdict.

Covers:
  1. Backward-compat: omitting preset vs preset="default" yields identical results.
  2. Stricter preset bites: "institutional" fails gates that "default" passes.
  3. Lenient preset lifts score: "exploratory" passes gates that "default" fails.
  4. Score in [0, 100] and verdict in valid labels for every preset.
  5. Unknown preset raises ValueError.
  6. Neutral-gate rescaling still holds under non-default presets.
"""
from __future__ import annotations

import pytest

from rigor.analysis.verdict import VERDICT_PRESETS, compute_strategy_verdict

# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------

_VALID_VERDICTS = {"ROBUST", "MODERATE", "FRAGILE", "OVERFIT"}


def _good_metrics() -> dict:
    """Metrics for a well-behaved strategy with high trade count & positive SR."""
    return {
        "sharpe": 1.5,
        "net_sharpe": 1.2,
        "n_obs": 2520,            # ~10 years at 252 obs/yr
        "periods_per_year": 252,
        "total_trades": 300,
    }


def _good_ao() -> dict:
    """Anti-overfit dict that passes all default overfitting gates comfortably."""
    return {
        "pbo": 0.20,
        "dsr_p_skill": 0.80,
        "haircut_pct": 0.25,
        "psr": 0.99,
        "permutation_p": 0.01,
    }


def _good_rob() -> dict:
    """Robustness dict that passes all default stability/temporal/viability gates."""
    return {
        "has_cliff": False,
        "sensitivity_cv": 0.15,
        "wf_efficiency": 0.75,
        "sharpe_cv": 0.30,
        "has_decay": False,
        "min_period_sharpe": 0.20,
    }


# ---------------------------------------------------------------------------
# 1. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_no_preset_equals_default_preset(self) -> None:
        """Omitting preset and passing preset="default" must be byte-identical."""
        m, a, r = _good_metrics(), _good_ao(), _good_rob()
        v_no_preset = compute_strategy_verdict(m, a, r)
        v_default = compute_strategy_verdict(m, a, r, preset="default")

        assert v_no_preset.score == v_default.score
        assert v_no_preset.verdict == v_default.verdict
        assert len(v_no_preset.gates) == len(v_default.gates)
        assert [g.passed for g in v_no_preset.gates] == [g.passed for g in v_default.gates]

    def test_no_preset_equals_default_with_n_trials(self) -> None:
        """n_trials path still consistent when preset omitted."""
        m, a, r = _good_metrics(), _good_ao(), _good_rob()
        v1 = compute_strategy_verdict(m, a, r, n_trials=5)
        v2 = compute_strategy_verdict(m, a, r, n_trials=5, preset="default")
        assert v1.score == v2.score
        assert [g.passed for g in v1.gates] == [g.passed for g in v2.gates]

    def test_default_preset_field(self) -> None:
        """StrategyVerdict.preset is 'default' when preset omitted."""
        v = compute_strategy_verdict(_good_metrics(), _good_ao(), _good_rob())
        assert v.preset == "default"

    def test_default_summary_no_preset_tag(self) -> None:
        """Summary must NOT include 'Preset:' for default."""
        v = compute_strategy_verdict(_good_metrics(), _good_ao(), _good_rob())
        assert "Preset:" not in v.summary

    def test_non_default_summary_includes_preset_tag(self) -> None:
        """Summary MUST include 'Preset: <name>' for non-default presets."""
        v = compute_strategy_verdict(_good_metrics(), _good_ao(), _good_rob(),
                                     preset="institutional")
        assert "Preset: institutional" in v.summary


# ---------------------------------------------------------------------------
# 2. Stricter preset bites — "institutional" vs "default"
# ---------------------------------------------------------------------------

class TestInstitutionalStricter:
    """Construct a borderline input that passes default but fails institutional."""

    @staticmethod
    def _borderline_metrics() -> dict:
        """125 trades: passes default (>=100) but fails institutional (>=150)."""
        m = _good_metrics()
        m["total_trades"] = 125
        return m

    @staticmethod
    def _borderline_ao() -> dict:
        """PBO=0.35: passes default (<0.40) but fails institutional (<0.30)."""
        a = _good_ao()
        a["pbo"] = 0.35
        return a

    def test_institutional_score_lte_default_score(self) -> None:
        m, a, r = self._borderline_metrics(), self._borderline_ao(), _good_rob()
        v_default = compute_strategy_verdict(m, a, r, preset="default")
        v_inst = compute_strategy_verdict(m, a, r, preset="institutional")
        assert v_inst.score <= v_default.score

    def test_pbo_gate_flips(self) -> None:
        """PBO=0.35 → passes default (<0.40), fails institutional (<0.30)."""
        m, a, r = _good_metrics(), self._borderline_ao(), _good_rob()
        v_def = compute_strategy_verdict(m, a, r, preset="default")
        v_inst = compute_strategy_verdict(m, a, r, preset="institutional")

        pbo_gate_def = next(g for g in v_def.gates if "PBO" in g.name)
        pbo_gate_inst = next(g for g in v_inst.gates if "PBO" in g.name)
        assert pbo_gate_def.passed is True
        assert pbo_gate_inst.passed is False

    def test_trades_gate_flips(self) -> None:
        """125 trades → passes default (>=100), fails institutional (>=150)."""
        m, a, r = self._borderline_metrics(), _good_ao(), _good_rob()
        v_def = compute_strategy_verdict(m, a, r, preset="default")
        v_inst = compute_strategy_verdict(m, a, r, preset="institutional")

        trades_gate_def = next(g for g in v_def.gates if "Trades" in g.name)
        trades_gate_inst = next(g for g in v_inst.gates if "Trades" in g.name)
        assert trades_gate_def.passed is True
        assert trades_gate_inst.passed is False

    def test_institutional_preset_field(self) -> None:
        m, a, r = _good_metrics(), _good_ao(), _good_rob()
        v = compute_strategy_verdict(m, a, r, preset="institutional")
        assert v.preset == "institutional"


# ---------------------------------------------------------------------------
# 3. Lenient preset lifts score — "exploratory" vs "default"
# ---------------------------------------------------------------------------

class TestExploratoryLenient:
    """An input that fails gates under default but passes them under exploratory."""

    @staticmethod
    def _sparse_metrics() -> dict:
        """Only 60 trades: fails default (>=100), passes exploratory (>=50)."""
        m = _good_metrics()
        m["total_trades"] = 60
        return m

    @staticmethod
    def _borderline_ao() -> dict:
        """PBO=0.45: fails default (<0.40), passes exploratory (<0.50)."""
        a = _good_ao()
        a["pbo"] = 0.45
        return a

    def test_exploratory_score_gte_default_score(self) -> None:
        m, a, r = self._sparse_metrics(), self._borderline_ao(), _good_rob()
        v_default = compute_strategy_verdict(m, a, r, preset="default")
        v_exp = compute_strategy_verdict(m, a, r, preset="exploratory")
        assert v_exp.score >= v_default.score

    def test_pbo_gate_lifts(self) -> None:
        """PBO=0.45 → fails default (<0.40), passes exploratory (<0.50)."""
        m, a, r = _good_metrics(), self._borderline_ao(), _good_rob()
        v_def = compute_strategy_verdict(m, a, r, preset="default")
        v_exp = compute_strategy_verdict(m, a, r, preset="exploratory")

        pbo_def = next(g for g in v_def.gates if "PBO" in g.name)
        pbo_exp = next(g for g in v_exp.gates if "PBO" in g.name)
        assert pbo_def.passed is False
        assert pbo_exp.passed is True

    def test_trades_gate_lifts(self) -> None:
        """60 trades → fails default (>=100), passes exploratory (>=50)."""
        m, a, r = self._sparse_metrics(), _good_ao(), _good_rob()
        v_def = compute_strategy_verdict(m, a, r, preset="default")
        v_exp = compute_strategy_verdict(m, a, r, preset="exploratory")

        tr_def = next(g for g in v_def.gates if "Trades" in g.name)
        tr_exp = next(g for g in v_exp.gates if "Trades" in g.name)
        assert tr_def.passed is False
        assert tr_exp.passed is True

    def test_exploratory_preset_field(self) -> None:
        m, a, r = _good_metrics(), _good_ao(), _good_rob()
        v = compute_strategy_verdict(m, a, r, preset="exploratory")
        assert v.preset == "exploratory"


# ---------------------------------------------------------------------------
# 4. Score always in [0, 100]; verdict always a valid label
# ---------------------------------------------------------------------------

class TestScoreAndVerdictBounds:
    @pytest.mark.parametrize("preset", list(VERDICT_PRESETS))
    def test_score_in_range_good_input(self, preset: str) -> None:
        v = compute_strategy_verdict(_good_metrics(), _good_ao(), _good_rob(), preset=preset)
        assert 0.0 <= v.score <= 100.0
        assert v.verdict in _VALID_VERDICTS

    @pytest.mark.parametrize("preset", list(VERDICT_PRESETS))
    def test_score_in_range_bad_input(self, preset: str) -> None:
        """Adversarial (all-failing) input still yields a valid score/verdict."""
        m = {"sharpe": -1.0, "net_sharpe": -1.0, "n_obs": 100,
             "periods_per_year": 252, "total_trades": 5}
        a = {"pbo": 0.99, "dsr_p_skill": 0.01, "haircut_pct": 0.99, "psr": 0.10}
        r = {"has_cliff": True, "sensitivity_cv": 2.0, "wf_efficiency": 0.10,
             "sharpe_cv": 2.0, "has_decay": True, "min_period_sharpe": -5.0}
        v = compute_strategy_verdict(m, a, r, preset=preset)
        assert 0.0 <= v.score <= 100.0
        assert v.verdict in _VALID_VERDICTS

    @pytest.mark.parametrize("preset", list(VERDICT_PRESETS))
    def test_score_in_range_empty_input(self, preset: str) -> None:
        """Completely empty dicts → all neutral gates → score 0 or 100 but valid."""
        v = compute_strategy_verdict({}, preset=preset)
        assert 0.0 <= v.score <= 100.0
        assert v.verdict in _VALID_VERDICTS


# ---------------------------------------------------------------------------
# 5. Unknown preset raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownPreset:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            compute_strategy_verdict(_good_metrics(), preset="nonexistent_xyz")

    def test_error_lists_valid_presets(self) -> None:
        with pytest.raises(ValueError, match="default"):
            compute_strategy_verdict(_good_metrics(), preset="bogus")

    def test_case_sensitive(self) -> None:
        """Preset names are case-sensitive — 'Default' is not 'default'."""
        with pytest.raises(ValueError):
            compute_strategy_verdict(_good_metrics(), preset="Default")


# ---------------------------------------------------------------------------
# 6. Neutral-gate rescaling under non-default presets
# ---------------------------------------------------------------------------

class TestNeutralGateRescaling:
    @pytest.mark.parametrize("preset", list(VERDICT_PRESETS))
    def test_sparse_input_neutral_gates(self, preset: str) -> None:
        """Only metrics provided; antioverfit/robustness missing → many neutral gates."""
        m = {"sharpe": 1.0, "net_sharpe": 0.8, "n_obs": 2520, "periods_per_year": 252}
        v = compute_strategy_verdict(m, preset=preset)

        neutral_count = sum(1 for g in v.gates if g.neutral)
        assert neutral_count > 0, "Expected at least some neutral gates with sparse input"
        assert 0.0 <= v.score <= 100.0
        assert v.verdict in _VALID_VERDICTS

    def test_neutral_gates_excluded_from_eff_max(self) -> None:
        """Neutral gates must not count toward effective max (100-pt denominator)."""
        # Provide only metrics, no antioverfit/robustness → many neutral gates.
        m = {"sharpe": 1.0, "net_sharpe": 0.8, "n_obs": 2520, "periods_per_year": 252}
        v = compute_strategy_verdict(m, preset="conservative")

        # Recompute effective max manually and verify the score formula holds.
        eff_max = sum(g.weight for g in v.gates if not g.neutral) or 100.0
        earned = sum(g.weight for g in v.gates if g.passed and not g.neutral)
        expected_score = (earned / eff_max) * 100.0
        assert abs(v.score - expected_score) < 1e-9

    def test_all_presets_have_valid_preset_field(self) -> None:
        """The .preset field echoes the requested preset name for all presets."""
        m = {"sharpe": 0.5, "net_sharpe": 0.3, "n_obs": 1000, "periods_per_year": 252}
        for name in VERDICT_PRESETS:
            v = compute_strategy_verdict(m, preset=name)
            assert v.preset == name
