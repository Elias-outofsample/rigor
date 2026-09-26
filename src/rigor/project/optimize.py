"""Backward-compatible shim.

The optimiser moved to its own package, :mod:`rigor.optimize`, when it grew the
two-pass search, data-snooping tests, robust selection, and ensemble. This shim
keeps ``rigor.project.optimize_strategy`` working. New code should import from
``rigor.optimize``.
"""
from __future__ import annotations

from ..optimize.core import optimize_strategy

__all__ = ["optimize_strategy"]
