"""Compact one-line strategy quality badge (additive to ``rigor.analysis.verdict``).

``rigor.analysis.verdict`` and ``rigor.validation.verdict`` already produce the full
multi-pillar deployment grade (a ROBUST/MODERATE/FRAGILE/OVERFIT score with a
gate breakdown). This module deliberately does NOT re-derive that battery. It
adds the one genuinely-missing piece: a **compact, four-level badge** for a list
row or rail — ``WEAK · FRAGILE · MODERATE · ROBUST`` — computed from gates that
are *already computed* (PBO, full-sample Sharpe, deployed consistency, recent
decay) plus the hard-cap logic the full verdict does not carry.

The cap logic is the load-bearing, additive part: a near-zero edge, a
catastrophic drawdown, or poor consistency can hold a strategy below ROBUST
*regardless of the gate count* — the distinction that stops a deep-drawdown
momentum sleeve (real edge, scary risk) being labelled identically to a
genuinely overfit artefact, and stops a near-zero-Sharpe sleeve being graded on
"positive 60 % of windows" alone.

dict-in / dict-out: callers pass the metrics they have already computed (no
returns series is recomputed here). Missing keys degrade to a neutral fallback
gate; a fully-empty input yields ``WEAK`` without raising. All thresholds live
in ``BADGE`` so they are configured in ONE place.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["strategy_badge", "badge_from_gates", "BADGE", "BadgeThresholds", "LABELS"]

# Worst -> best.
LABELS = ["WEAK", "FRAGILE", "MODERATE", "ROBUST"]


@dataclass(frozen=True)
class BadgeThresholds:
    """All badge thresholds in one place (see module docstring for rationale)."""

    pbo_max: float = 0.50          # overfit probability ceiling (gate 1)
    sharpe_min: float = 0.50       # full-sample Sharpe floor for a credible edge
    consistency_min: float = 0.60  # min fraction of rolling-1y windows Sharpe > 0
    wf_eff_min: float = 0.60       # fallback when no consistency available
    recent_min: float = 0.30       # last-window Sharpe floor (gate 3)
    recent_frac: float = 0.45      # last window must keep >= this fraction of full SR
    oos_sharpe_min: float = 0.70   # fallback when no recent-Sharpe available
    # Hard caps.
    cap_sharpe_fragile: float = 0.30   # full SR below -> FRAGILE max
    cap_sharpe_moderate: float = 0.50  # full SR below -> MODERATE max
    cap_drawdown: float = -0.60        # deep max-DD -> MODERATE max (risky, not ROBUST)
    cap_drawdown_ruin: float = -0.80   # near-ruin max-DD -> FRAGILE max
    cap_consistency: float = 0.40      # consistency below -> FRAGILE max


BADGE = BadgeThresholds()


def _num(x) -> float | None:
    """Coerce to a finite float, or ``None`` if absent/non-numeric/non-finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def badge_from_gates(
    pbo: float | None,
    full_sharpe: float | None,
    consistency: float | None,
    holdout_ok: bool | None,
    *,
    max_drawdown: float | None = None,
    wf_eff: float | None = None,
    oos_sharpe: float | None = None,
    cfg: BadgeThresholds = BADGE,
) -> str:
    """Grade a strategy from ALREADY-COMPUTED gates; return one of ``LABELS``.

    Three additive gates, then hard caps:
      1. Credible edge — ``pbo <= pbo_max`` AND (no full Sharpe, or it is
         ``>= sharpe_min``). Consistency alone is not an edge.
      2. Deployed consistency — ``consistency >= consistency_min`` (falls back to
         ``wf_eff >= wf_eff_min`` when consistency is absent).
      3. No severe recent decay — ``holdout_ok`` (falls back to
         ``oos_sharpe >= oos_sharpe_min`` when holdout is absent).

    Hard caps then clamp the rank: near-zero edge, catastrophic drawdown, or poor
    consistency cannot be ROBUST regardless of the gate count. Every argument is
    optional; absent inputs simply do not earn (or trigger) their gate/cap.
    """
    pbo = _num(pbo)
    full = _num(full_sharpe)
    consistency = _num(consistency)
    max_dd = _num(max_drawdown)
    wf_eff = _num(wf_eff)
    oos_sharpe = _num(oos_sharpe)

    score = 0
    # Gate 1: credible edge.
    if pbo is not None and pbo <= cfg.pbo_max and (full is None or full >= cfg.sharpe_min):
        score += 1
    # Gate 2: deployed consistency (consistency preferred, wf_eff fallback).
    if consistency is not None:
        score += 1 if consistency >= cfg.consistency_min else 0
    elif wf_eff is not None and wf_eff >= cfg.wf_eff_min:
        score += 1
    # Gate 3: no severe recent decay (holdout preferred, oos_sharpe fallback).
    if holdout_ok is not None:
        score += 1 if holdout_ok else 0
    elif oos_sharpe is not None and oos_sharpe >= cfg.oos_sharpe_min:
        score += 1

    rank = score

    # Hard caps (applied after scoring).
    if full is not None and full < cfg.cap_sharpe_fragile:
        rank = min(rank, 1)
    elif full is not None and full < cfg.cap_sharpe_moderate:
        rank = min(rank, 2)
    # Drawdown is a RISK dimension, not an overfit one: a real edge with a deep
    # DD is risky-not-fragile -> cap to MODERATE; only near-ruin forces FRAGILE.
    if max_dd is not None and max_dd < cfg.cap_drawdown_ruin:
        rank = min(rank, 1)
    elif max_dd is not None and max_dd < cfg.cap_drawdown:
        rank = min(rank, 2)
    if consistency is not None and consistency < cfg.cap_consistency:
        rank = min(rank, 1)

    return LABELS[max(0, min(rank, len(LABELS) - 1))]


def strategy_badge(metrics: dict | None, *, cfg: BadgeThresholds = BADGE) -> dict:
    """dict-in / dict-out wrapper around :func:`badge_from_gates`.

    Reads already-computed fields from *metrics* (all optional):
      ``pbo``, ``full_sharpe`` (or ``sharpe``), ``consistency``, ``holdout_ok``,
      ``max_drawdown`` (or ``max_dd``), ``wf_eff`` (or ``wf_efficiency``),
      ``oos_sharpe``.

    Returns ``{"badge": <label>, "rank": <0-3>}``. A ``None`` or empty dict, or
    one missing every key, yields the neutral worst label ``WEAK`` (rank 0)
    without raising.
    """
    m = metrics or {}
    holdout = m.get("holdout_ok")
    label = badge_from_gates(
        pbo=m.get("pbo"),
        full_sharpe=m.get("full_sharpe", m.get("sharpe")),
        consistency=m.get("consistency"),
        holdout_ok=bool(holdout) if holdout is not None else None,
        max_drawdown=m.get("max_drawdown", m.get("max_dd")),
        wf_eff=m.get("wf_eff", m.get("wf_efficiency")),
        oos_sharpe=m.get("oos_sharpe"),
        cfg=cfg,
    )
    return {"badge": label, "rank": LABELS.index(label)}
