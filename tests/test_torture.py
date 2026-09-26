"""Tests for the adversarial torture score (rigor.analysis.torture).

The torture score re-frames the already-computed validation/optimise report as
"what breaks this edge first?". These tests build synthetic inputs shaped like
rigor's real ``validate()`` + ``optimize_strategy()`` output and assert:

  * the verdict label maps correctly off ``overfit_risk_pct`` band edges;
  * ``overfit_risk_pct`` always lands in ``[0, 100]``;
  * ``ranked_weaknesses`` is sorted by descending severity;
  * ``first_to_break`` is the highest-severity *breaking* (>=0.5) test;
  * missing keys are skipped gracefully (no crash, neutral) — including the
    all-empty case that yields ``UNKNOWN``/NaN;
  * ``to_dict()`` round-trips the scalar fields and weakness list.
"""
from __future__ import annotations

import math

import pytest

from rigor.analysis import TortureResult, Weakness, compute_torture_score
from rigor.analysis.torture import BREAK_SEVERITY

_VALID_VERDICTS = {"SURVIVES", "STRESSED", "FRAGILE", "BROKEN"}


# ---------------------------------------------------------------------------
# Synthetic report builders — shaped like rigor's actual dict outputs.
# ---------------------------------------------------------------------------

def _robust_report() -> dict:
    """A report a clean edge shrugs off: every gate well below its break point."""
    return {
        "cpcv": {"pbo": 0.05, "rank_correlation": 0.85},
        "overfit": {"dsr": 0.95, "haircut_pct": 0.10},
        "robustness": {"walk_forward": {"efficiency": 0.90}},
        "falsification": {
            "noise_injection": {"fragility": 0.05},
            "surrogate_price": {"p_value": 0.01},
        },
        "data_snooping": {"hansen_spa": {"p_consistent": 0.01}},
        "lookahead": {"suspect": False},
        "cost_stress": {
            "verdict": "PASS",
            "net_sharpe_by_mult": {"2.0": 1.2},
            "survival_sharpe_min": 0.5,
        },
        "trade_mc": {"prob_negative_cagr": 0.01},
    }


def _broken_report() -> dict:
    """A report an artefact fails everywhere: every gate at/over its break point."""
    return {
        "cpcv": {"pbo": 0.95, "rank_correlation": -0.70},
        "overfit": {"dsr": 0.05, "haircut_pct": 1.20},
        "robustness": {"walk_forward": {"efficiency": -0.20}},
        "falsification": {
            "noise_injection": {"fragility": 1.50},
            "surrogate_price": {"p_value": 0.90},
        },
        "data_snooping": {"hansen_spa": {"p_consistent": 0.85}},
        "lookahead": {"suspect": True},
        "cost_stress": {
            "verdict": "FAIL",
            "net_sharpe_by_mult": {"2.0": -0.40},
            "survival_sharpe_min": 0.5,
        },
        "trade_mc": {"prob_negative_cagr": 0.95},
    }


# ---------------------------------------------------------------------------
# 1. Verdict label maps off the risk band edges.
# ---------------------------------------------------------------------------

class TestVerdictLabel:
    def test_robust_report_survives(self) -> None:
        res = compute_torture_score(_robust_report())
        assert res.torture_verdict == "SURVIVES"
        assert res.overfit_risk_pct < 25.0

    def test_broken_report_breaks(self) -> None:
        res = compute_torture_score(_broken_report())
        assert res.torture_verdict == "BROKEN"
        assert res.overfit_risk_pct >= 75.0

    def test_verdict_in_valid_set(self) -> None:
        for report in (_robust_report(), _broken_report()):
            assert compute_torture_score(report).torture_verdict in _VALID_VERDICTS

    @pytest.mark.parametrize(
        ("dsr", "expected_band"),
        # severity = (0.90 - dsr) / 0.90, risk = severity*100:
        #   dsr 0.90 -> 0%   SURVIVES | dsr 0.60 -> 33% STRESSED
        #   dsr 0.40 -> 56%  FRAGILE  | dsr 0.00 -> 100% BROKEN
        [(0.90, "SURVIVES"), (0.60, "STRESSED"), (0.40, "FRAGILE"), (0.0, "BROKEN")],
    )
    def test_single_dsr_gate_band(self, dsr: float, expected_band: str) -> None:
        """A lone DSR gate makes risk = its own severity*100, so we can target each band."""
        res = compute_torture_score({"overfit": {"dsr": dsr}})
        assert res.torture_verdict == expected_band


# ---------------------------------------------------------------------------
# 2. overfit_risk_pct bounds [0, 100].
# ---------------------------------------------------------------------------

class TestRiskBounds:
    @pytest.mark.parametrize("report", [_robust_report(), _broken_report()])
    def test_risk_in_range(self, report: dict) -> None:
        res = compute_torture_score(report)
        assert 0.0 <= res.overfit_risk_pct <= 100.0

    def test_adversarial_extremes_clamped(self) -> None:
        """Out-of-range raw inputs (pbo>1, dsr<0, hc>1) must not push risk past 100."""
        report = {
            "cpcv": {"pbo": 5.0},
            "overfit": {"dsr": -3.0, "haircut_pct": 4.0},
            "robustness": {"walk_forward": {"efficiency": -9.0}},
        }
        res = compute_torture_score(report)
        assert 0.0 <= res.overfit_risk_pct <= 100.0
        assert res.overfit_risk_pct == 100.0  # every severity clamps to 1.0


# ---------------------------------------------------------------------------
# 3. ranked_weaknesses ordering (descending severity).
# ---------------------------------------------------------------------------

class TestRanking:
    def test_descending_severity(self) -> None:
        res = compute_torture_score(_broken_report())
        sevs = [w.severity for w in res.ranked_weaknesses]
        assert sevs == sorted(sevs, reverse=True)

    def test_all_weaknesses_present(self) -> None:
        """Every supplied signal becomes exactly one ranked weakness."""
        res = compute_torture_score(_broken_report())
        # 11 gates: pbo, dsr, haircut, wf, fragility, surrogate, spa, cost, trade_mc,
        # lookahead, rank_corr.
        assert len(res.ranked_weaknesses) == 11
        assert all(isinstance(w, Weakness) for w in res.ranked_weaknesses)


# ---------------------------------------------------------------------------
# 4. first_to_break selection.
# ---------------------------------------------------------------------------

class TestFirstToBreak:
    def test_broken_first_is_top_breaking(self) -> None:
        res = compute_torture_score(_broken_report())
        top = res.ranked_weaknesses[0]
        assert top.severity >= BREAK_SEVERITY
        assert res.first_to_break == top.test

    def test_robust_nothing_breaks(self) -> None:
        res = compute_torture_score(_robust_report())
        assert res.first_to_break is None
        assert all(w.severity < BREAK_SEVERITY for w in res.ranked_weaknesses)

    def test_single_breaking_gate_named(self) -> None:
        """Only the haircut gate breaks => it must be first_to_break."""
        report = {
            "overfit": {"dsr": 0.95, "haircut_pct": 0.90},  # dsr fine, haircut broken
            "cpcv": {"pbo": 0.02},
        }
        res = compute_torture_score(report)
        assert res.first_to_break == "Haircut Sharpe"


# ---------------------------------------------------------------------------
# 5. Graceful handling when keys are missing (no crash, neutral).
# ---------------------------------------------------------------------------

class TestGracefulMissing:
    def test_empty_report_is_unknown(self) -> None:
        res = compute_torture_score({})
        assert res.torture_verdict == "UNKNOWN"
        assert math.isnan(res.overfit_risk_pct)
        assert res.first_to_break is None
        assert res.ranked_weaknesses == []

    def test_none_report_is_unknown(self) -> None:
        res = compute_torture_score(None)
        assert res.torture_verdict == "UNKNOWN"
        assert math.isnan(res.overfit_risk_pct)

    def test_partial_report_skips_absent_gates(self) -> None:
        """Only PBO present => exactly one weakness, no crash on the other 10 gates."""
        res = compute_torture_score({"cpcv": {"pbo": 0.30}})
        assert len(res.ranked_weaknesses) == 1
        assert res.ranked_weaknesses[0].test.startswith("PBO")

    def test_unrelated_keys_ignored(self) -> None:
        """Junk/irrelevant keys neither crash nor add weaknesses."""
        res = compute_torture_score({"slug": "foo", "as_of": "2026-01-01", "ranking": []})
        assert res.torture_verdict == "UNKNOWN"
        assert res.ranked_weaknesses == []

    def test_non_numeric_value_skipped(self) -> None:
        """A non-numeric signal is dropped, not crashed on."""
        res = compute_torture_score({"cpcv": {"pbo": "n/a"}, "overfit": {"dsr": 0.80}})
        # pbo unparseable -> skipped; only the dsr gate survives.
        tests = [w.test for w in res.ranked_weaknesses]
        assert tests == ["DSR P(skill)"]

    def test_nan_value_neutralised(self) -> None:
        """A NaN signal is kept but contributes zero severity (neutral)."""
        res = compute_torture_score({"overfit": {"dsr": float("nan")}})
        # NaN dsr -> _num returns None -> gate skipped entirely.
        assert res.torture_verdict == "UNKNOWN"

    def test_walk_forward_opt_fallback(self) -> None:
        """When validation WF efficiency is absent, derive it from walk_forward_opt."""
        report = {
            "walk_forward_opt": {"is_sharpe_mean": 1.0, "oos_sharpe_mean": 0.2},
        }
        res = compute_torture_score(report)
        wf = next(w for w in res.ranked_weaknesses if "WF efficiency" in w.test)
        # eff = 0.2/1.0 = 0.20 -> below 0.50 floor -> severity (0.5-0.2)/0.5 = 0.6.
        assert wf.value == pytest.approx(0.20)
        assert wf.severity == pytest.approx(0.6)

    def test_lookahead_check_alias(self) -> None:
        """``lookahead_check`` is accepted as an alias for ``lookahead``."""
        res = compute_torture_score({"lookahead_check": {"suspect": True}})
        la = res.ranked_weaknesses[0]
        assert la.test == "Look-ahead"
        assert la.severity == 1.0

    def test_cost_stress_kwarg_override(self) -> None:
        """cost_stress passed as a kwarg is honoured even if absent from the report."""
        res = compute_torture_score({}, cost_stress={"verdict": "FAIL"})
        cs = next(w for w in res.ranked_weaknesses if "Cost survival" in w.test)
        assert cs.severity == 1.0


# ---------------------------------------------------------------------------
# 6. to_dict() round-trips.
# ---------------------------------------------------------------------------

class TestToDict:
    def test_scalar_fields_round_trip(self) -> None:
        res = compute_torture_score(_broken_report())
        d = res.to_dict()
        assert d["overfit_risk_pct"] == res.overfit_risk_pct
        assert d["torture_verdict"] == res.torture_verdict
        assert d["first_to_break"] == res.first_to_break
        assert d["summary"] == res.summary

    def test_weakness_list_round_trips(self) -> None:
        res = compute_torture_score(_broken_report())
        d = res.to_dict()
        assert len(d["ranked_weaknesses"]) == len(res.ranked_weaknesses)
        for src, out in zip(res.ranked_weaknesses, d["ranked_weaknesses"], strict=True):
            assert out["test"] == src.test
            assert out["severity"] == round(src.severity, 3)
            assert out["value"] == src.value
            assert out["threshold"] == src.threshold
            assert out["weight"] == src.weight
            assert out["narrative"] == src.narrative

    def test_dict_is_json_serialisable(self) -> None:
        import json
        res = compute_torture_score(_broken_report())
        # Must not raise — every field is a JSON-native scalar/list/dict.
        json.loads(json.dumps(res.to_dict()))

    def test_unknown_result_to_dict(self) -> None:
        """The empty-input UNKNOWN result still serialises (NaN -> float, list empty)."""
        res = compute_torture_score({})
        d = res.to_dict()
        assert d["torture_verdict"] == "UNKNOWN"
        assert d["ranked_weaknesses"] == []
        assert math.isnan(d["overfit_risk_pct"])


# ---------------------------------------------------------------------------
# 7. Dataclass shape sanity.
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_result_type(self) -> None:
        assert isinstance(compute_torture_score(_robust_report()), TortureResult)

    def test_summary_mentions_verdict_and_count(self) -> None:
        res = compute_torture_score(_broken_report())
        assert res.torture_verdict in res.summary
        assert "tests break" in res.summary
