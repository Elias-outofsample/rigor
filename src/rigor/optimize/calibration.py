"""Calibration profiles — named threshold sets for the optimisation pipeline.

The optimiser already ships every *stage* (search, select, CPCV-PBO, deflated
Sharpe, holdout, falsification, walk-forward) and a ``robust`` selection ladder
with floors (:class:`rigor.optimize.select.RobustFloors`). What it lacked was a
single, named answer to "which floors?". A **calibration profile** is exactly
that: a typed bundle of thresholds — *not* a new algorithm — that the
:mod:`rigor.optimize.pipeline` orchestrator hands to the existing selection ladder
and to a final deployment-readiness gate.

Two profiles ship, mirroring the research-vs-production split:

  * ``discovery`` — **permissive** research floors. The job here is to *surface*
    candidate edges without prematurely killing them; the anti-overfit numbers
    (DSR / PBO / snooping) are then read by a human. These floors are the ones
    the framework already used by default (``RobustFloors()`` defaults), so the
    discovery profile is byte-compatible with today's ``--select robust`` run.
  * ``deployment`` — **strict** production floors aligned with the promotion
    gate. A winner only passes the deployment gate when it clears stricter
    Sharpe / PSR / PBO / DSR / activity thresholds *and* the framework's own
    promotion verdict is not ``REJECT`` (see :mod:`rigor.validation.verdict` and
    :mod:`rigor.project.promotion_gate`). Deployment never *invents* a verdict —
    it tightens the floors and defers the final call to the shared verdict.

A profile reuses OUR existing vocabulary: it projects onto a
:class:`~rigor.optimize.select.RobustFloors` for the selection ladder via
:meth:`CalibrationProfile.robust_floors`, and exposes the extra
deployment-readiness thresholds (DSR, min-trades, the verdict ban-list) the
ladder does not cover.

    from rigor.optimize.calibration import get_profile
    profile = get_profile("deployment")
    floors = profile.robust_floors()          # -> select.RobustFloors

Nothing here is opt-in-breaking: ``get_profile(None)`` and an unknown name both
resolve to ``discovery``, so a caller that never asks for a profile gets the
permissive, backward-compatible behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .select import RobustFloors

__all__ = [
    "CalibrationProfile",
    "DISCOVERY",
    "DEPLOYMENT",
    "PROFILES",
    "get_profile",
]


@dataclass(frozen=True)
class CalibrationProfile:
    """A named bundle of acceptance thresholds for one optimisation run.

    The first block (``sharpe_floor`` … ``pbo_max``) is exactly the
    :class:`~rigor.optimize.select.RobustFloors` vocabulary and feeds the existing
    ``robust`` selection ladder via :meth:`robust_floors`. The second block adds
    the **deployment-readiness** thresholds the ladder does not itself enforce —
    the deflated-Sharpe floor, a minimum-activity floor, and the set of
    promotion verdicts that block deployment — so a single profile fully
    parameterises both selection *and* the final PROMOTE/REJECT gate.
    """

    name: str

    # --- selection-ladder floors (project onto select.RobustFloors) ---------
    sharpe_floor: float       # per-config annualised Sharpe floor
    psr_floor: float          # P(true SR > 0) floor
    cagr_floor: float         # reject configs below this CAGR
    max_dd_floor: float       # reject near-total wipe-outs (drawdown floor)
    pbo_max: float            # grid-level: reject if CPCV-PBO >= this

    # --- deployment-readiness gate (beyond the ladder) ----------------------
    dsr_min: float            # deflated-Sharpe P(skill) floor on the winner
    min_active_bars: int      # minimum traded/active bars (a thin-activity floor)
    # Promotion verdicts that block a deployment-ready outcome. The framework's
    # shared verdict (PROMOTE / CONDITIONAL / REJECT) is the source of truth; a
    # profile only chooses which of those are disqualifying for *its* purpose.
    blocking_verdicts: frozenset[str]

    def robust_floors(self) -> RobustFloors:
        """Project this profile onto a :class:`~rigor.optimize.select.RobustFloors`.

        The selection ladder consumes ``RobustFloors`` and nothing else, so the
        profile's first block is the single source for what the ladder enforces.
        The deployment-readiness extras (``dsr_min``, ``min_active_bars``,
        ``blocking_verdicts``) are applied by the pipeline *after* selection.
        """
        return RobustFloors(
            sharpe=self.sharpe_floor,
            psr=self.psr_floor,
            cagr=self.cagr_floor,
            max_dd=self.max_dd_floor,
            pbo=self.pbo_max,
        )

    def with_overrides(self, **changes: Any) -> CalibrationProfile:
        """Return a copy with selected fields overridden (e.g. for a per-run tweak)."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        """JSON-friendly view of the profile (used in the pipeline report)."""
        return {
            "name": self.name,
            "sharpe_floor": self.sharpe_floor,
            "psr_floor": self.psr_floor,
            "cagr_floor": self.cagr_floor,
            "max_dd_floor": self.max_dd_floor,
            "pbo_max": self.pbo_max,
            "dsr_min": self.dsr_min,
            "min_active_bars": self.min_active_bars,
            "blocking_verdicts": sorted(self.blocking_verdicts),
        }


# ---------------------------------------------------------------------------
# The two shipped profiles.
# ---------------------------------------------------------------------------

#: DISCOVERY — permissive research floors. The first block matches
#: ``RobustFloors()`` defaults exactly, so ``--profile discovery`` selects
#: identically to today's ``--select robust``. The deployment extras are
#: deliberately lax (DSR floor at the verdict's "any skill" line of 0.5, no
#: meaningful activity floor, only an outright REJECT blocks): discovery is for
#: surfacing candidates, not for gating them.
DISCOVERY = CalibrationProfile(
    name="discovery",
    sharpe_floor=0.30,
    psr_floor=0.60,
    cagr_floor=-0.05,
    max_dd_floor=-0.999,
    pbo_max=0.50,
    dsr_min=0.50,
    min_active_bars=20,
    blocking_verdicts=frozenset({"REJECT"}),
)

#: DEPLOYMENT — strict production floors aligned with the promotion gate.
#: Sharpe / PSR / PBO / DSR are tightened to the levels the validation verdict
#: rewards (PSR > 0.95 and PBO < 0.40 are verdict gates; DSR > 0.5 is a CRITICAL
#: verdict gate, lifted here to a stronger 0.60 confidence). A real activity
#: floor and a CONDITIONAL-or-worse ban align "deployment-ready" with the
#: enforced promotion gate, which blocks any deployable strategy carrying a
#: REJECT verdict (:mod:`rigor.project.promotion_gate`). The result: deployment
#: admits a strict subset of what discovery admits.
DEPLOYMENT = CalibrationProfile(
    name="deployment",
    sharpe_floor=0.80,
    psr_floor=0.90,
    cagr_floor=0.00,
    max_dd_floor=-0.60,
    pbo_max=0.40,
    dsr_min=0.60,
    min_active_bars=100,
    blocking_verdicts=frozenset({"REJECT", "CONDITIONAL"}),
)

#: Registry of the shipped profiles, keyed by (lower-cased) name.
PROFILES: dict[str, CalibrationProfile] = {
    DISCOVERY.name: DISCOVERY,
    DEPLOYMENT.name: DEPLOYMENT,
}


def get_profile(name: str | None = "discovery") -> CalibrationProfile:
    """Resolve a calibration profile by (case-insensitive) name.

    ``None`` or an unknown name resolves to the permissive :data:`DISCOVERY`
    profile, so a caller that never asks for a profile — or fat-fingers the
    name — gets the backward-compatible behaviour rather than a crash.
    """
    if name is None:
        return DISCOVERY
    return PROFILES.get(str(name).lower(), DISCOVERY)
