"""Strategy longevity — crowding, life-cycle phase, forward survival probability.

The return/Sharpe/drawdown summary answers *did this edge work?*; this module
answers the question it misses: *will the edge still be exploitable in five
years?* Three honest heuristics, built ONLY from quantities the pipeline already
computes (factor loadings, capacity, rolling Sharpe, edge-decay slope, PBO,
crash ratio) — no capital-flow data, no fabricated precision:

  * ``compute_crowding`` — how arbitraged the edge already is (factor-beta
    crowding, family/recipe prior, capacity & turnover scalability).
  * ``compute_lifecycle`` — emergence / exploitation / maturity / decline, from
    recent vs early-sample Sharpe (refined by the edge-decay slope when given).
  * ``compute_survival`` — a weighted GEOMETRIC aggregate of the longevity
    signals into a forward-durability prior in ``[0, 1]``.

Every score ships with its component breakdown and a plain-language caveat so it
reads as a *prior*, not a measurement. Pure numpy/pandas; dict-in/dict-out; a
degenerate or information-poor input returns a neutral result rather than raising.
"""
from __future__ import annotations

import math
from typing import TypeGuard

import numpy as np
import pandas as pd

__all__ = ["compute_crowding", "compute_lifecycle", "compute_survival"]

# Well-known, heavily-arbitraged academic factors. A strategy whose return is
# largely explained by these is, by construction, more crowded than one trading
# an idiosyncratic residual.
_CROWDED_FACTORS = ("momentum", "value", "size", "low_vol", "quality")

# Family priors: how accessible/popular the *recipe* is (0 = exotic/niche,
# 1 = textbook). A SP500 monthly momentum is far more crowded than a commodity
# cross-sectional carry overlay.
_FAMILY_CROWDING = {
    "Momentum": 0.85,
    "MeanReversion": 0.65,
    "Trend": 0.70,
    "Factor": 0.80,
    "Seasonality": 0.40,
    "GEM": 0.75,
    "PairTrading": 0.55,
}

# Life-cycle phase -> survival contribution (decline is a strong negative).
_PHASE_SURVIVAL = {
    "EMERGENCE": 0.75,
    "EXPLOITATION": 0.85,
    "MATURITY": 0.70,
    "DECLINE": 0.25,
    "INSUFFICIENT_DATA": 0.50,
}


def _clip01(x: float) -> float:
    """Clamp to ``[0, 1]`` as a plain float."""
    return float(min(1.0, max(0.0, x)))


def _finite(x: float | None) -> TypeGuard[float]:
    """True iff *x* is a present, finite number."""
    return x is not None and bool(np.isfinite(x))


def compute_crowding(
    factor_betas: dict | None,
    family: str | None,
    capacity_usd: float | None,
    annual_turnover: float | None,
) -> dict:
    """Heuristic crowding score 0-100 (higher = more crowded / arbitraged).

    Components (each ``[0, 1]``, transparently combined):
      * ``factor_popularity`` — max ``|beta|`` to the crowded academic factors.
      * ``recipe_popularity`` — family prior (textbook recipes are more crowded).
      * ``scalability`` — high capacity + low turnover = easy for large players
        to pile in (more crowdable); tiny capacity / frantic turnover resists it.

    Returns a dict with ``score``, ``verdict`` (HIGH/MODERATE/LOW), ``components``
    and a caveat ``note``. Never raises: a ``None``/empty input contributes its
    neutral default.
    """
    fb = factor_betas or {}
    crowded_loadings = [
        abs(float(fb[k]))
        for k in _CROWDED_FACTORS
        if k in fb and _finite(fb[k])
    ]
    factor_pop = _clip01(max(crowded_loadings) / 1.5) if crowded_loadings else 0.0

    recipe_pop = _FAMILY_CROWDING.get(family or "", 0.5)

    # scalability: capacity above ~$500M and turnover below ~5x/yr = very
    # crowdable; tiny capacity or frantic turnover = niche.
    scal = 0.5
    if _finite(capacity_usd):
        scal = _clip01(math.log10(max(capacity_usd, 1e6) / 1e6) / 3.0)  # 1M->0, 1B->1
    if _finite(annual_turnover) and annual_turnover > 0:
        # high turnover lowers crowdability (impact costs scale with size)
        scal = _clip01(scal * _clip01(1.0 - (annual_turnover - 2.0) / 20.0))

    score01 = 0.45 * factor_pop + 0.30 * recipe_pop + 0.25 * scal
    score = round(100.0 * score01, 1)
    if score >= 70:
        verdict = "HIGH"
        note = "Largely a well-known, scalable factor bet — expect compressed forward returns."
    elif score >= 45:
        verdict = "MODERATE"
        note = "Partly explained by popular factors; some idiosyncratic edge remains."
    else:
        verdict = "LOW"
        note = "Idiosyncratic / niche edge — little evidence of crowding (proxy only)."
    return {
        "score": score,
        "verdict": verdict,
        "components": {
            "factor_popularity": round(factor_pop, 3),
            "recipe_popularity": round(recipe_pop, 3),
            "scalability": round(scal, 3),
        },
        "note": note + " Heuristic proxy (no capital-flow data).",
    }


def compute_lifecycle(
    returns,
    periods_per_year: int,
    edge_decay_slope: float | None = None,
) -> dict:
    """Life-cycle phase from edge now vs historically.

    Compares the recent (~last 2y) annualised Sharpe to the full-sample Sharpe
    and to the earliest third, then labels {EMERGENCE / EXPLOITATION / MATURITY
    / DECLINE}. ``edge_decay_slope`` (Sharpe units per year, e.g. from
    ``structural.edge_decay``) refines the call when available.

    A history shorter than one year returns ``INSUFFICIENT_DATA`` rather than
    raising.
    """
    r = (returns if isinstance(returns, pd.Series) else pd.Series(returns)).dropna()
    n = len(r)
    if periods_per_year <= 0 or n < periods_per_year:
        return {"phase": "INSUFFICIENT_DATA", "note": "history too short for a life-cycle call."}

    def _sharpe(x: pd.Series) -> float:
        sd = x.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return float("nan")
        return float(x.mean() / sd * math.sqrt(periods_per_year))

    recent_win = min(n, 2 * periods_per_year)
    sr_recent = _sharpe(r.iloc[-recent_win:])
    sr_full = _sharpe(r)
    third = max(periods_per_year, n // 3)
    sr_early = _sharpe(r.iloc[:third])

    ratio = (
        sr_recent / sr_full
        if (np.isfinite(sr_recent) and np.isfinite(sr_full) and sr_full != 0)
        else float("nan")
    )

    rising = np.isfinite(sr_recent) and np.isfinite(sr_early) and sr_recent > sr_early + 0.2
    falling = np.isfinite(sr_recent) and np.isfinite(sr_early) and sr_recent < sr_early - 0.2
    if _finite(edge_decay_slope):
        if edge_decay_slope < -0.15:
            falling = True
        elif edge_decay_slope > 0.15:
            rising = True

    young = n < 4 * periods_per_year
    if young and not falling:
        phase = "EMERGENCE"
        note = "Short live history, edge not yet decayed — promising but unproven."
    elif rising and not falling:
        phase = "EXPLOITATION"
        note = "Edge stronger recently than early on — still being harvested."
    elif falling:
        phase = "DECLINE"
        note = "Recent edge materially below its earlier level — possible arbitrage / regime shift."
    else:
        phase = "MATURITY"
        note = "Edge broadly stable across the sample — mature, neither accelerating nor dying."

    return {
        "phase": phase,
        "sharpe_recent_2y": round(sr_recent, 3) if np.isfinite(sr_recent) else None,
        "sharpe_early_third": round(sr_early, 3) if np.isfinite(sr_early) else None,
        "sharpe_full": round(sr_full, 3) if np.isfinite(sr_full) else None,
        "recent_vs_full": round(ratio, 3) if np.isfinite(ratio) else None,
        "edge_decay_slope_per_yr": round(float(edge_decay_slope), 3)
        if _finite(edge_decay_slope) else None,
        "note": note,
    }


def compute_survival(
    pbo: float | None,
    psr: float | None,
    edge_decay_slope: float | None,
    crash_ratio: float | None,
    crowding_score: float | None,
    lifecycle_phase: str | None,
) -> dict:
    """Forward survival probability ``P(edge still exploitable in ~5y)``, 0-1.

    Combines independent longevity signals as a weighted GEOMETRIC mean (any one
    near-zero drags the whole thing down — a single fatal flaw should not be
    averaged away). Each input is first mapped to a ``[0, 1]`` survival
    contribution; absent inputs are simply dropped. An honest aggregate, not a
    calibrated probability — reported with its breakdown and weakest link.

    With no usable inputs it returns ``INSUFFICIENT_DATA`` and a ``None``
    probability rather than raising.
    """
    contribs: dict[str, float] = {}

    # Overfitting: low PBO -> likely real OOS edge.
    if _finite(pbo):
        contribs["not_overfit"] = _clip01(1.0 - pbo)
    # Statistical significance: PSR is already a probability.
    if _finite(psr):
        contribs["significant"] = _clip01(psr)
    # Edge decay: negative slope = dying. Map [-0.5, +0.2] -> [0, 1].
    if _finite(edge_decay_slope):
        contribs["not_decaying"] = _clip01((edge_decay_slope + 0.5) / 0.7)
    # Crisis behaviour: crash_ratio >= 1 (does well in crises) is robust; < 0.3
    # (short-vol premium) is fragile.
    if _finite(crash_ratio):
        contribs["crisis_robust"] = _clip01(crash_ratio / 1.0)
    # Crowding: high crowding -> lower survival.
    if _finite(crowding_score):
        contribs["uncrowded"] = _clip01(1.0 - crowding_score / 100.0)
    # Life-cycle: decline is a strong negative; emergence/exploitation positive.
    if lifecycle_phase:
        contribs["lifecycle"] = _PHASE_SURVIVAL.get(lifecycle_phase, 0.5)

    if not contribs:
        return {"probability": None, "verdict": "INSUFFICIENT_DATA", "components": {},
                "weakest_link": None}

    # weighted geometric mean (equal weights; floor each at 0.05 so a single 0
    # doesn't annihilate the product entirely but still dominates).
    vals = [max(0.05, v) for v in contribs.values()]
    logmean = sum(math.log(v) for v in vals) / len(vals)
    prob = _clip01(math.exp(logmean))

    if prob >= 0.65:
        verdict = "LIKELY_DURABLE"
    elif prob >= 0.45:
        verdict = "UNCERTAIN"
    else:
        verdict = "AT_RISK"
    weakest = min(contribs.items(), key=lambda kv: kv[1])
    return {
        "probability": round(prob, 3),
        "verdict": verdict,
        "components": {k: round(v, 3) for k, v in contribs.items()},
        "weakest_link": weakest[0],
        "note": (
            "Weighted geometric aggregate of overfit/significance/decay/crisis/"
            "crowding/life-cycle signals — a prior on durability, not a calibrated "
            "probability."
        ),
    }
