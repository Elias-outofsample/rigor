"""Opt-in parameter optimisation.

A strategy joins the optimiser by declaring a multi-value ``param_grid()``; the
framework never requires it (``rigor optimize`` is a separate command, never part
of ``rigor run`` or CI). The orchestrator searches the grid and reports the winner
through the anti-overfit machinery the framework already ships, plus the
data-snooping tests and search-generalisation diagnostics ported (cleanly,
reusing the existing verdict/scoring) from the prior research library.

    from rigor.optimize import optimize_strategy
    report = optimize_strategy("strategies/trend_following/sma_trend", as_of="2026-06-01")

Deliberately *not* ported from the prior library (would need infrastructure the
framework doesn't have, and would duplicate the existing verdict): the parallel
5-pillar composite scoring (the verdict already grades every result), the HMM
regime optimiser (needs a regime-labelling layer), and standalone config
clustering (the ensemble's diversity selection covers the same need).
"""
from .calibration import DEPLOYMENT, DISCOVERY, CalibrationProfile, get_profile
from .core import optimize_strategy
from .pipeline import PipelineResult, run_pipeline

__all__ = [
    "optimize_strategy",
    "run_pipeline",
    "PipelineResult",
    "CalibrationProfile",
    "get_profile",
    "DISCOVERY",
    "DEPLOYMENT",
]
