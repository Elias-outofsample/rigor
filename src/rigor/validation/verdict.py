"""Promotion verdict: turn the validation battery into one decision.

Gates are grouped into four pillars (overfitting, significance, temporal,
viability). The score is the weighted fraction of non-neutral gates passed.
Two gates are CRITICAL (Sharpe > 0, DSR P(skill) > 0.5): failing either forces
REJECT regardless of score. A gate is NEUTRAL when its input is unavailable
(e.g. PBO with no parameter grid) and is excluded from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Gate:
    name: str
    passed: bool
    value: float
    threshold: float
    weight: float
    critical: bool = False
    neutral: bool = False


@dataclass
class Verdict:
    verdict: str            # PROMOTE | CONDITIONAL | REJECT
    grade: str              # ROBUST | MODERATE | FRAGILE | OVERFIT
    score: float            # 0..100
    gates: list[Gate] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "grade": self.grade,
            "score": round(self.score, 1),
            "gates": [
                {"name": g.name, "passed": bool(g.passed), "value": _round(g.value),
                 "threshold": g.threshold, "weight": g.weight,
                 "critical": g.critical, "neutral": g.neutral}
                for g in self.gates
            ],
        }


def _round(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return v


def compute_verdict(metrics: dict, overfit: dict, robustness: dict, cpcv: dict | None) -> Verdict:
    wf = robustness.get("walk_forward", {})
    py = robustness.get("per_year", {})
    boot = robustness.get("bootstrap", {})
    n_obs = int(metrics.get("n_obs", 0))
    g: list[Gate] = []

    # --- Pillar 1: overfitting (30) ---
    dsr = overfit.get("dsr", 0.5)
    g.append(Gate("DSR P(skill) > 0.50", dsr > 0.50, dsr, 0.50, 12, critical=True))
    hc = overfit.get("haircut_pct", 1.0)
    g.append(Gate("Haircut < 50%", hc < 0.50, hc, 0.50, 9))
    if cpcv is None:
        g.append(Gate("PBO < 0.40", False, float("nan"), 0.40, 9, neutral=True))
    else:
        g.append(Gate("PBO < 0.40", cpcv["pbo"] < 0.40, cpcv["pbo"], 0.40, 9))

    # --- Pillar 2: significance (25) ---
    psr = overfit.get("psr", 0.0)
    g.append(Gate("PSR > 0.95", psr > 0.95, psr, 0.95, 9))
    t = overfit.get("harvey_t", 0.0)
    g.append(Gate("Harvey t > 3", t > 3.0, t, 3.0, 8))
    mtrl = overfit.get("min_track_record_length", float("inf"))
    g.append(Gate("Track record sufficient", mtrl <= n_obs, mtrl, n_obs, 8))

    # --- Pillar 3: temporal consistency (25) ---
    eff = wf.get("efficiency", 0.0)
    g.append(Gate("Walk-forward efficiency > 0.50", eff > 0.50, eff, 0.50, 9))
    pf = py.get("positive_fraction", 0.0)
    g.append(Gate("Positive years >= 60%", pf >= 0.60, pf, 0.60, 8))
    cv = wf.get("cv", float("inf"))
    g.append(Gate("Sub-period Sharpe CV < 0.50", cv < 0.50, cv, 0.50, 8))

    # --- Pillar 4: viability (20) ---
    sr = metrics.get("sharpe", 0.0)
    g.append(Gate("Sharpe > 0", sr > 0.0, sr, 0.0, 8, critical=True))
    ws = py.get("worst_sharpe", -99.0)
    g.append(Gate("Worst-year Sharpe > -0.5", ws > -0.5, ws, -0.5, 6))
    lo = boot.get("ci_lower", -99.0)
    g.append(Gate("Bootstrap 95% CI lower > 0", lo > 0.0, lo, 0.0, 6))

    eff_max = sum(x.weight for x in g if not x.neutral)
    earned = sum(x.weight for x in g if x.passed and not x.neutral)
    score = (earned / eff_max * 100.0) if eff_max > 0 else 0.0

    grade = ("ROBUST" if score >= 70 else "MODERATE" if score >= 50
             else "FRAGILE" if score >= 30 else "OVERFIT")
    critical_fail = any(x.critical and not x.passed for x in g)
    if critical_fail:
        verdict = "REJECT"
    elif score >= 70:
        verdict = "PROMOTE"
    elif score >= 50:
        verdict = "CONDITIONAL"
    else:
        verdict = "REJECT"
    return Verdict(verdict=verdict, grade=grade, score=score, gates=g)
