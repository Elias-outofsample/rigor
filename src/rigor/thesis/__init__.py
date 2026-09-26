"""Thesis generator (vendored from an earlier standalone thesis tool, content-driven path only).

Generates publication-quality PDF thesis documents from a ``GenThesisConfig``
(narrative text + DataFrame figures + tables). The original heavy, CSV/Fama-French
"strategy backtest" path was removed during vendoring — Rigor drives this from a
``BacktestResult`` via :mod:`rigor.thesis_card`.
"""

from .gen_thesis_config import (
    FigureSpec,
    FormalHypothesis,
    GenThesisBrief,
    GenThesisConfig,
    Reference,
    TableSpec,
)
from .gen_thesis_generator import GenThesisGenerator

__all__ = [
    "GenThesisGenerator",
    "GenThesisConfig",
    "GenThesisBrief",
    "FigureSpec",
    "TableSpec",
    "FormalHypothesis",
    "Reference",
]
