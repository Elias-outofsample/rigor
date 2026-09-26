"""Adversarial torture score — "what breaks this edge first?" (#17).

The framework already runs a battery of overfit/robustness tests, each speaking
its own language (PBO, DSR, p-values, ratios). ``rigor.validation.verdict`` and
``rigor.analysis.verdict`` aggregate those into a *deployment grade* (0-100, higher
is better). This module answers the COMPLEMENTARY, adversarial question the
trader actually asks while torturing a candidate:

    "How likely is this edge an artefact, and WHICH test breaks it first?"

Each available signal is normalised to a ``break_severity`` in ``[0, 1]`` (0 =
the strategy shrugs the test off, 1 = the test fully breaks the edge). The
weighted aggregate is ``overfit_risk_pct`` (0-100, **higher is worse** — the
mirror image of the verdict score), and the severity-ranked weakness list is the
adversarial narrative: read top-down, it tells you the order in which the
strategy falls apart, and ``first_to_break`` names the single test that snaps
first.

This module **does not replace** ``verdict.py`` — it re-reads the same already
computed report and re-frames it. It is pure-Python with no third-party
dependencies, is fully decoupled from ``rigor.validation``/``rigor.optimize`` (it
accepts plain dicts shaped like their outputs), and skips any gate whose input
is absent rather than crashing or penalising a blank — an information-poor report
is scored on what it actually carries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["compute_torture_score", "TortureResult", "Weakness"]

# ---------------------------------------------------------------------------
# Calibration thresholds (inlined).
#
# The original in-house implementation pulled these from
# ``optimization.calibration.DEFAULT_CALIBRATION``,
# which rigor does not ship. They are reproduced here as named module-level
# constants so the gate logic carries no anonymous magic numbers. Each is the
# "breaks at" reference the matching gate normalises severity against.
# ---------------------------------------------------------------------------

PBO_COINFLIP = 0.50              # PBO=0.5 is a coin-flip => fully overfit.
DSR_SKILL_TARGET = 0.90          # DSR P(skill) we want; below it the Sharpe may be luck.
HAIRCUT_FULL = 0.50              # >=100% haircut wipes the edge; severity = haircut fraction.
WF_EFFICIENCY_MIN = 0.50         # OOS/IS retention floor; below it, IS->OOS decay is severe.
FRAGILITY_MAX = 0.80             # noise fragility "breaks at" (mirrors run_falsification default).
SURROGATE_PMAX = 0.05            # surrogate/snooping p above which the edge is indistinct.
SPA_PMAX = 0.05                  # Hansen SPA significance threshold.
TRADE_MC_PNEG_MAX = 0.10         # tolerated P(CAGR<0) on trade resampling.
COST_SURVIVAL_SHARPE_MIN = 0.50  # net Sharpe a cost-stressed edge must keep.
RANK_CORR_TARGET = 0.30          # IS->OOS Kendall tau we want; negative => ranking is noise.
RANK_CORR_SPAN = 1.0             # tau range used to scale severity (tau<=-0.7 => fully broken).
BREAK_SEVERITY = 0.50            # a gate at/above this severity is "breaking".

# Verdict band edges on ``overfit_risk_pct`` (higher = worse).
VERDICT_BANDS = ((25.0, "SURVIVES"), (50.0, "STRESSED"), (75.0, "FRAGILE"))
VERDICT_WORST = "BROKEN"


def _clip01(x: float) -> float:
    """Clamp to ``[0, 1]``; NaN maps to 0 (a missing/degenerate signal does not break)."""
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _num(x) -> float | None:
    """Coerce to a finite float, or ``None`` if absent/non-numeric/non-finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(obj, key, default=None):
    """Read ``key`` from a dict OR a dataclass/attribute object (``None``-safe)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class Weakness:
    """One torture gate's result: how hard this single test breaks the edge."""

    test: str
    severity: float          # 0..1 (1 = the test fully breaks the edge)
    value: float | None      # the raw signal value that was gated
    threshold: float | None  # the "breaks at" reference for that signal
    weight: float            # relative importance in the aggregate risk
    narrative: str           # human-readable reading of the gate


@dataclass
class TortureResult:
    """Aggregated adversarial verdict plus the ranked weakness narrative."""

    overfit_risk_pct: float          # 0..100, HIGHER = worse (mirror of verdict score)
    torture_verdict: str             # SURVIVES | STRESSED | FRAGILE | BROKEN | UNKNOWN
    first_to_break: str | None       # highest-severity *breaking* test, or None
    ranked_weaknesses: list[Weakness] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        """JSON-friendly view; round-trips the scalar fields and each weakness."""
        return {
            "overfit_risk_pct": self.overfit_risk_pct,
            "torture_verdict": self.torture_verdict,
            "first_to_break": self.first_to_break,
            "ranked_weaknesses": [
                {
                    "test": w.test,
                    "severity": round(w.severity, 3),
                    "value": w.value,
                    "threshold": w.threshold,
                    "weight": w.weight,
                    "narrative": w.narrative,
                }
                for w in self.ranked_weaknesses
            ],
            "summary": self.summary,
        }


def _verdict_for(risk_pct: float) -> str:
    """Map an overfit-risk percentage onto a torture band label."""
    for edge, label in VERDICT_BANDS:
        if risk_pct < edge:
            return label
    return VERDICT_WORST


def compute_torture_score(
    report: dict | None,
    *,
    cost_stress: dict | None = None,
    trade_mc: dict | None = None,
) -> TortureResult:
    """Synthesise every available robustness signal into one adversarial score.

    Reads a plain dict shaped like the merged output of ``rigor.validation.validate``
    and ``rigor.optimize.optimize_strategy`` (and friends). It re-runs nothing; it
    re-frames the already computed numbers as "what breaks first". Each gate is
    skipped (contributes nothing — neutral) when its input key is missing, so a
    returns-only report and a full grid-search report both score gracefully.

    Args:
        report: the merged pipeline/validation dict. Recognised keys:
            ``cpcv`` (``pbo``, ``rank_correlation``), ``overfit`` (``dsr``,
            ``haircut_pct``), ``robustness.walk_forward`` /
            ``walk_forward_opt`` (efficiency), ``falsification``
            (``noise_injection.fragility``, ``surrogate_price.p_value``),
            ``data_snooping.hansen_spa`` (``p_consistent``/``p_value``),
            ``lookahead`` / ``lookahead_check`` (``suspect``), ``cost_stress``
            and ``trade_mc``.
        cost_stress: cost-survival gate input; falls back to
            ``report["cost_stress"]``.
        trade_mc: trade-resample Monte-Carlo input; falls back to
            ``report["trade_mc"]``.

    Returns:
        A :class:`TortureResult`. With no recognised signal present the verdict
        is ``"UNKNOWN"`` and ``overfit_risk_pct`` is NaN — never an exception.
    """
    rep = report or {}
    cost_stress = cost_stress if cost_stress is not None else rep.get("cost_stress")
    trade_mc = trade_mc if trade_mc is not None else rep.get("trade_mc")

    weaknesses: list[Weakness] = []

    def add(test: str, severity: float, value, threshold, weight: float, narrative: str) -> None:
        weaknesses.append(Weakness(
            test=test, severity=_clip01(severity), value=value,
            threshold=threshold, weight=weight, narrative=narrative,
        ))

    # 1. PBO — probability of backtest overfitting from CPCV (weight 3, core).
    pbo = _num(_get(rep.get("cpcv"), "pbo"))
    if pbo is not None:
        add("PBO (overfit prob.)", pbo / PBO_COINFLIP, round(pbo, 3), PBO_COINFLIP, 3.0,
            f"PBO={pbo:.0%}: probability the in-sample over-performance is grid "
            f"over-fitting (0%=robust, >={PBO_COINFLIP:.0%}=coin-flip).")

    # 2. DSR P(skill) — deflated Sharpe confidence (weight 3, core).
    dsr = _num(_get(rep.get("overfit"), "dsr"))
    if dsr is not None:
        add("DSR P(skill)", (DSR_SKILL_TARGET - dsr) / DSR_SKILL_TARGET, round(dsr, 3),
            DSR_SKILL_TARGET, 3.0,
            f"DSR={dsr:.0%}: confidence the Sharpe survives the multiple-testing "
            f"correction. Low => the Sharpe is probably luck.")

    # 3. Haircut Sharpe — Harvey-Liu-Zhu multiple-testing cut (weight 2).
    hc = _num(_get(rep.get("overfit"), "haircut_pct"))
    if hc is not None:
        add("Haircut Sharpe", hc, round(hc, 3), HAIRCUT_FULL, 2.0,
            f"Haircut={hc:.0%} of the Sharpe erased by the multiple-testing "
            f"correction. >=100% => the edge vanishes after the data-mining penalty.")

    # 4. Walk-forward efficiency — OOS/IS retention (weight 3, core).
    #    rigor exposes this two ways: validation's ``robustness.walk_forward.efficiency``,
    #    or the optimiser's ``walk_forward_opt`` (oos/is means). Prefer the explicit
    #    efficiency; otherwise derive it from the opt means.
    wf_eff = _num(_get(_get(rep.get("robustness"), "walk_forward"), "efficiency"))
    if wf_eff is None:
        wfo = rep.get("walk_forward_opt") or {}
        is_mean = _num(_get(wfo, "is_sharpe_mean"))
        oos_mean = _num(_get(wfo, "oos_sharpe_mean"))
        if is_mean is not None and oos_mean is not None and is_mean > 0:
            wf_eff = oos_mean / is_mean
    if wf_eff is not None:
        add("WF efficiency (OOS/IS)", (WF_EFFICIENCY_MIN - wf_eff) / WF_EFFICIENCY_MIN,
            round(wf_eff, 3), WF_EFFICIENCY_MIN, 3.0,
            f"Walk-forward efficiency={wf_eff:.0%}: share of the IS edge kept out "
            f"of sample. <{WF_EFFICIENCY_MIN:.0%} => severe IS->OOS decay.")

    # 5. Noise fragility — Sharpe decay under perturbation (weight 2).
    fals = rep.get("falsification") or {}
    frag = _num(_get(_get(fals, "noise_injection"), "fragility"))
    if frag is not None:
        add("Noise fragility", frag / FRAGILITY_MAX, round(frag, 3), FRAGILITY_MAX, 2.0,
            f"Fragility={frag:.2f}: Sharpe lost per unit of injected noise. "
            f">={FRAGILITY_MAX:.1f} => a brittle edge, typical of over-fitting.")

    # 6. Surrogate-price p-value — edge survives structureless same-spectrum data (weight 2).
    surr_p = _num(_get(_get(fals, "surrogate_price"), "p_value"))
    if surr_p is not None:
        sev = 0.0 if surr_p < SURROGATE_PMAX else (surr_p - SURROGATE_PMAX) / (1.0 - SURROGATE_PMAX)
        add("Surrogate price (spectrum)", sev, round(surr_p, 3), SURROGATE_PMAX, 2.0,
            f"p={surr_p:.2f}: if the edge also wins on phase-randomised prices "
            f"(p>{SURROGATE_PMAX:.2f}), it is a linear artefact, not real structure.")

    # 7. Reality-check SPA p-value — data snooping over the searched family (weight 1.5).
    spa = _get(rep.get("data_snooping"), "hansen_spa") or {}
    spa_p = _num(_get(spa, "p_consistent", _get(spa, "p_value")))
    if spa_p is not None:
        sev = 0.0 if spa_p < SPA_PMAX else (spa_p - SPA_PMAX) / (1.0 - SPA_PMAX)
        add("Hansen SPA (snooping)", sev, round(spa_p, 3), SPA_PMAX, 1.5,
            f"SPA p={spa_p:.2f}: significance after the multi-combo data-snooping "
            f"correction. p>{SPA_PMAX:.2f} => indistinguishable from chance.")

    # 8. Cost-stress — does the edge survive stressed costs (weight 2.5)?
    cs_verdict = _get(cost_stress, "verdict")
    if cs_verdict is not None:
        net2x = _num((_get(cost_stress, "net_sharpe_by_mult") or {}).get("2.0"))
        smin = _num(_get(cost_stress, "survival_sharpe_min")) or COST_SURVIVAL_SHARPE_MIN
        if net2x is not None and smin > 0:
            sev = (smin - net2x) / smin
        else:
            sev = {"PASS": 0.0, "WARN": 0.5, "FAIL": 1.0}.get(cs_verdict, 0.0)
        add("Cost survival (2x)", sev,
            (round(net2x, 3) if net2x is not None else None), smin, 2.5,
            f"Cost-stress={cs_verdict}: net Sharpe at 2x real costs"
            + (f" = {net2x:.2f}." if net2x is not None else ".")
            + " A high-turnover edge dies here.")

    # 9. Trade-resample MC — sequence risk on the trade population (weight 2).
    pneg = _num(_get(trade_mc, "prob_negative_cagr"))
    if pneg is not None:
        add("Trade-resample MC", pneg, round(pneg, 3), TRADE_MC_PNEG_MAX, 2.0,
            f"P(CAGR<0)={pneg:.0%} on resampling the trades: probability the edge "
            f"rests on a few lucky trades.")

    # 10. Look-ahead — future-peeking on the selected config (weight 3, fatal).
    la = rep.get("lookahead") or rep.get("lookahead_check") or {}
    if "suspect" in la:
        suspect = bool(_get(la, "suspect"))
        add("Look-ahead", 1.0 if suspect else 0.0, 1.0 if suspect else 0.0, 0.0, 3.0,
            "Look-ahead SUSPECT: the strategy collapses after a 1-bar shift "
            "=> future-information leakage." if suspect
            else "No look-ahead leakage detected (1-bar shift is neutral).")

    # 11. CPCV rank correlation — does the IS-best hold its OOS rank (weight 1.5)?
    tau = _num(_get(rep.get("cpcv"), "rank_correlation"))
    if tau is not None:
        add("CPCV rank corr (IS->OOS)", (RANK_CORR_TARGET - tau) / RANK_CORR_SPAN,
            round(tau, 3), RANK_CORR_TARGET, 1.5,
            f"Kendall tau={tau:.2f}: stability of the parameter ranking IS->OOS. "
            f"Negative => the IS-best becomes OOS-bad (noise).")

    if not weaknesses:
        return TortureResult(
            overfit_risk_pct=float("nan"), torture_verdict="UNKNOWN",
            first_to_break=None, ranked_weaknesses=[],
            summary="No torture test available (inputs missing).",
        )

    total_w = sum(w.weight for w in weaknesses)
    risk = sum(w.severity * w.weight for w in weaknesses) / total_w
    overfit_risk_pct = round(risk * 100.0, 1)

    ranked = sorted(weaknesses, key=lambda w: w.severity, reverse=True)
    first = next((w.test for w in ranked if w.severity >= BREAK_SEVERITY), None)
    verdict = _verdict_for(overfit_risk_pct)

    n_breaking = sum(1 for w in weaknesses if w.severity >= BREAK_SEVERITY)
    summary = (
        f"Overfit risk {overfit_risk_pct:.0f}% | {verdict} | "
        f"{n_breaking}/{len(weaknesses)} tests break"
        + (f" | first to break: {first}" if first else "")
    )
    return TortureResult(
        overfit_risk_pct=overfit_risk_pct, torture_verdict=verdict,
        first_to_break=first, ranked_weaknesses=ranked, summary=summary,
    )
