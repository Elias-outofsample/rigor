"""Rigor strategy contract: subclass StrategyBase, carry a StrategyConfig."""

from .base import StrategyBase
from .config import StrategyConfig

__all__ = ["StrategyBase", "StrategyConfig"]
