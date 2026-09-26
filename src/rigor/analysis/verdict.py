"""Unified 5-pillar strategy verdict (#1).

A single composite score [0-100] from five pillars, weighted:

  1. Overfitting detection (30) — PBO, DSR P(skill), haircut.
  2. Parameter stability (20) — no cliff, sensitivity CV.
  3. Temporal consistency (20) — WF efficiency, decade Sharpe CV, no edge decay.
  4. Statistical significance (15) — PSR, Harvey t-stat, permutation p.
  5. Practical viability (15) — net Sharpe, worst-period Sharpe, trade count.

Gates whose input is missing are marked **neutral** and dropped from BOTH the
numerator and the effective max, so an information-poor report (e.g. a bare
returns series with no CPCV/permutation inputs) is scored on what's available
rather than auto-failing. Pure numpy; consumes dicts assembled from
``rigor.metrics`` / ``rigor.validation`` outputs.

Archetype threshold presets
---------------------------
Pass ``preset=`` to ``compute_strategy_verdict`` to adjust gate thresholds for a
particular strategy archetype without changing gate names, weights, or rescaling
logic.  ``preset="default"`` (or omitting the argument) reproduces the original
behavior byte-for-byte.  Available presets are exposed in ``VERDICT_PRESETS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["compute_strategy_verdict", "StrategyVerdict", "VerdictGate", "VERDICT_PRESETS"]

# ---------------------------------------------------------------------------
# Archetype threshold presets
# ---------------------------------------------------------------------------
# Each preset is a flat dict of ``threshold_name -> value`` that overrides the
# hard-coded defaults in the pillar helpers.  An empty dict means "no overrides"
# and is equivalent to the original behaviour.
#
# Keys must exactly match the names consumed by ``_thr(overrides, name, default)``
# inside the pillar helpers.  Unrecognised keys are silently ignored (safe to
# add future thresholds without breaking old presets).
# ---------------------------------------------------------------------------

VERDICT_PRESETS: dict[str, dict[str, float]] = {
    # ---- default -------------------------------------------------------
    # Empty — every pillar helper falls back to its hard-coded default.
    # BACKWARD COMPATIBILITY: this must stay an empty dict.
    "default": {},

    # ---- momentum ------------------------------------------------------
    # Trend-following strategies tend to have clustered returns; tighten WF
    # efficiency slightly to demand more consistent out-of-sample performance.
    "momentum": {
        "wf_efficiency": 0.55,      # tighter than default 0.50
    },

    # ---- mean_reversion ------------------------------------------------
    # Higher turnover, short holding periods; relax worst-period floor (more
    # whipsaw periods expected) and raise minimum trade count to compensate for
    # statistical noise from many small trades.
    "mean_reversion": {
        "trades": 200,              # stricter than default 100 (need more samples)
        "worst_period_sharpe": -0.7,  # looser than default -0.5 (choppy regimes ok)
    },

    # ---- conservative --------------------------------------------------
    # Capital-preservation mandate; tighten drawdown floor and demand a
    # materially positive net Sharpe rather than just > 0.
    "conservative": {
        "worst_period_sharpe": -0.3,  # stricter than default -0.5
        "net_sharpe": 0.2,            # stricter than default 0.0
    },

    # ---- institutional -------------------------------------------------
    # Strictest profile for allocator/fund-of-funds due diligence.
    "institutional": {
        "pbo": 0.30,            # stricter than default 0.40
        "dsr_p_skill": 0.60,    # stricter than default 0.50
        "haircut": 0.40,        # stricter than default 0.50
        "sensitivity_cv": 0.25, # stricter than default 0.30
        "trades": 150,          # stricter than default 100
        # PSR > 0.95 and Harvey t > 3 are already strict; left unchanged.
    },

    # ---- exploratory ---------------------------------------------------
    # Most lenient — for early-stage research where sample size is small.
    "exploratory": {
        "pbo": 0.50,              # looser than default 0.40
        "dsr_p_skill": 0.45,     # looser than default 0.50
        "trades": 50,             # looser than default 100
        "worst_period_sharpe": -0.7,  # looser than default -0.5
    },
}

_VALID_PRESETS = frozenset(VERDICT_PRESETS)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VerdictGate:
    name: str
    passed: bool
    value: float
    threshold: float
    weight: float
    neutral: bool = False   # input unavailable → excluded from num + denom


@dataclass
class StrategyVerdict:
    verdict: str            # ROBUST | MODERATE | FRAGILE | OVERFIT
    score: float            # 0-100
    gates: list[VerdictGate]
    summary: str
    preset: str = field(default="default")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_strategy_verdict(
    metrics: dict,
    antioverfit: dict | None = None,
    robustness: dict | None = None,
    n_trials: int = 1,
    preset: str = "default",
) -> StrategyVerdict:
    """Unified 5-pillar verdict, rescaled by the non-neutral gate weight.

    Parameters
    ----------
    metrics:
        Core strategy metrics dict (sharpe, n_obs, net_sharpe, total_trades, …).
    antioverfit:
        Anti-overfit metrics dict (pbo, dsr_p_skill, haircut_pct, psr, …).
    robustness:
        Robustness metrics dict (wf_efficiency, sharpe_cv, sensitivity_cv, …).
    n_trials:
        Number of parameter combinations tried (used in DSR deflation).
    preset:
        Named archetype preset.  ``"default"`` (or omitting the argument)
        produces byte-identical results to the pre-preset implementation.
        See ``VERDICT_PRESETS`` for the full list of valid names.

    Raises
    ------
    ValueError
        If *preset* is not a key in ``VERDICT_PRESETS``.
    """
    if preset not in _VALID_PRESETS:
        valid = ", ".join(sorted(_VALID_PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Valid presets: {valid}")

    overrides = VERDICT_PRESETS[preset]

    gates: list[VerdictGate] = []
    total = 0.0
    for pillar in (
        _pillar_overfitting(antioverfit or {}, n_trials, overrides),
        _pillar_stability(robustness or {}, overrides),
        _pillar_temporal(metrics, robustness or {}, overrides),
        _pillar_significance(metrics, antioverfit or {}, overrides),
        _pillar_viability(metrics, robustness or {}, overrides),
    ):
        gates.extend(pillar["gates"])
        total += pillar["score"]

    eff_max = sum(g.weight for g in gates if not g.neutral) or 100.0
    norm_score = (total / eff_max) * 100.0 if eff_max > 0 else 0.0
    verdict = (
        "ROBUST" if norm_score >= 70
        else "MODERATE" if norm_score >= 50
        else "FRAGILE" if norm_score >= 30
        else "OVERFIT"
    )

    n_passed = sum(1 for g in gates if g.passed)
    n_eval = sum(1 for g in gates if not g.neutral)
    n_neutral = sum(1 for g in gates if g.neutral)
    preset_tag = f" | Preset: {preset}" if preset != "default" else ""
    summary = (
        f"Score: {norm_score:.0f}/100 ({total:.0f}/{eff_max:.0f} effective) | "
        f"{n_passed}/{n_eval} gates passed"
        + (f", {n_neutral} neutral (input missing)" if n_neutral else "")
        + f" | Verdict: {verdict}"
        + preset_tag
    )
    return StrategyVerdict(verdict=verdict, score=norm_score, gates=gates,
                           summary=summary, preset=preset)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _thr(overrides: dict[str, float], name: str, default: float) -> float:
    """Return the override value for *name* if present, else *default*."""
    return overrides.get(name, default)


def _gate(score: float, name: str, value, threshold: float, weight: float,
          *, present: bool, passed_if) -> tuple[float, VerdictGate]:
    """Build a gate; when ``present`` is False the gate is neutral (NaN value)."""
    if not present:
        return score, VerdictGate(name, False, float("nan"), threshold, weight, neutral=True)
    v = float(value)
    passed = bool(passed_if(v))
    return score + (weight if passed else 0.0), VerdictGate(name, passed, v, threshold, weight)


def _pillar_overfitting(ao: dict, n_trials: int, overrides: dict[str, float]) -> dict:
    score, gates = 0.0, []
    thr_pbo = _thr(overrides, "pbo", 0.40)
    thr_dsr = _thr(overrides, "dsr_p_skill", 0.50)
    thr_hc = _thr(overrides, "haircut", 0.50)
    score, g = _gate(score, "PBO < 0.40", ao.get("pbo"), thr_pbo, 10,
                     present=ao.get("pbo") is not None, passed_if=lambda v: v < thr_pbo)
    gates.append(g)
    score, g = _gate(score, "DSR P(skill) > 0.50", ao.get("dsr_p_skill"), thr_dsr, 10,
                     present=ao.get("dsr_p_skill") is not None,
                     passed_if=lambda v: v > thr_dsr)
    gates.append(g)
    score, g = _gate(score, "Haircut < 50%", ao.get("haircut_pct"), thr_hc, 10,
                     present=ao.get("haircut_pct") is not None,
                     passed_if=lambda v: v < thr_hc)
    gates.append(g)
    return {"score": score, "gates": gates}


def _pillar_stability(rob: dict, overrides: dict[str, float]) -> dict:
    score, gates = 0.0, []
    thr_cv = _thr(overrides, "sensitivity_cv", 0.30)
    score, g = _gate(score, "No parameter cliff", 1 - int(bool(rob.get("has_cliff"))), 0, 10,
                     present="has_cliff" in rob, passed_if=lambda v: v >= 1)
    gates.append(g)
    score, g = _gate(score, "Sensitivity CV < 0.30", rob.get("sensitivity_cv"), thr_cv, 10,
                     present="sensitivity_cv" in rob, passed_if=lambda v: v < thr_cv)
    gates.append(g)
    return {"score": score, "gates": gates}


def _pillar_temporal(metrics: dict, rob: dict, overrides: dict[str, float]) -> dict:
    score, gates = 0.0, []
    thr_wf = _thr(overrides, "wf_efficiency", 0.50)
    score, g = _gate(score, "WF efficiency > 0.50", rob.get("wf_efficiency"), thr_wf, 7,
                     present="wf_efficiency" in rob, passed_if=lambda v: v > thr_wf)
    gates.append(g)
    score, g = _gate(score, "Decade Sharpe CV < 0.50", rob.get("sharpe_cv"), 0.50, 7,
                     present="sharpe_cv" in rob, passed_if=lambda v: v < 0.50)
    gates.append(g)
    score, g = _gate(score, "No edge decay", 1 - int(bool(rob.get("has_decay"))), 0, 6,
                     present="has_decay" in rob, passed_if=lambda v: v >= 1)
    gates.append(g)
    return {"score": score, "gates": gates}


def _pillar_significance(metrics: dict, ao: dict, overrides: dict[str, float]) -> dict:
    score, gates = 0.0, []
    score, g = _gate(score, "PSR > 0.95", ao.get("psr", 0), 0.95,
                     5, present=True, passed_if=lambda v: v > 0.95)
    gates.append(g)
    sharpe = float(metrics.get("sharpe", 0) or 0)        # annualised
    n_obs = float(metrics.get("n_obs", 0) or 0)
    ppy = float(metrics.get("periods_per_year", 252) or 252)
    skew = float(metrics.get("skew", 0.0) or 0.0)
    kurt = float(metrics.get("kurtosis", 0.0) or 0.0)    # excess kurtosis
    # Harvey-Liu-Zhu t-stat on the *per-period* Sharpe with the Lo (2002) non-normal
    # SE: a left tail (skew < 0) or fat tail (kurt > 0) inflates the SE and lowers t,
    # so the gate is no longer fooled by skewed payoffs (e.g. short-vol). The prior
    # form used the annualised Sharpe in the SE and ignored skew/kurtosis entirely.
    sr_pp = sharpe / np.sqrt(ppy) if ppy > 0 else 0.0
    se_sq = 1.0 + 0.5 * sr_pp**2 - skew * sr_pp + (kurt / 4.0) * sr_pp**2
    t_stat = (
        sr_pp * np.sqrt(n_obs - 1.0) / np.sqrt(se_sq)
        if n_obs > 1 and se_sq > 0
        else 0.0
    )
    score, g = _gate(score, "Harvey t > 3", t_stat, 3, 5,
                     present=True, passed_if=lambda v: v > 3)
    gates.append(g)
    score, g = _gate(score, "Permutation p < 0.05", ao.get("permutation_p"), 0.05, 5,
                     present="permutation_p" in ao, passed_if=lambda v: v < 0.05)
    gates.append(g)
    return {"score": score, "gates": gates}


def _pillar_viability(metrics: dict, rob: dict, overrides: dict[str, float]) -> dict:
    score, gates = 0.0, []
    thr_ns = _thr(overrides, "net_sharpe", 0.0)
    thr_wp = _thr(overrides, "worst_period_sharpe", -0.5)
    thr_tr = _thr(overrides, "trades", 100.0)
    net_sharpe = metrics.get("net_sharpe", metrics.get("sharpe", 0))
    score, g = _gate(score, "Net Sharpe > 0", net_sharpe, thr_ns, 5,
                     present=True, passed_if=lambda v: v > thr_ns)
    gates.append(g)
    score, g = _gate(score, "Worst period Sharpe > -0.5", rob.get("min_period_sharpe"),
                     thr_wp, 5, present="min_period_sharpe" in rob,
                     passed_if=lambda v: v > thr_wp)
    gates.append(g)
    score, g = _gate(score, "Trades >= 100", metrics.get("total_trades"), thr_tr, 5,
                     present="total_trades" in metrics,
                     passed_if=lambda v: v >= thr_tr)
    gates.append(g)
    return {"score": score, "gates": gates}
