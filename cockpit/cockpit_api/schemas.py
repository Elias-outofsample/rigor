"""Request / response models for the cockpit API (Pydantic v2).

Percentages are entered as the UI shows them (e.g. ``max_dd = 30`` means 30%); the
engine adapter converts to fractions before calling ``build_portfolio``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    """Strict allocation gates. ``0`` = "off" for every field."""
    min_sharpe: float = 0.0
    min_cagr: float = 0.0          # percent
    max_beta: float = 0.0
    max_corr_bench: float = 0.0
    max_corr_strat: float = 0.0
    min_divers: float = 0.0
    max_risk: float = 0.0          # percent of portfolio risk per strategy
    max_dd: float = 0.0            # percent
    max_vol: float = 0.0           # percent


class BuildRequest(BaseModel):
    slugs: list[str] = Field(..., min_length=2)
    method: str = "robust (grid)"
    max_weight: float = 0.0        # percent cap per strategy; 0 = no cap
    leverage: float = 1.0          # gross-exposure cap (×)
    use_cpcv: bool = False
    constraints: Constraints = Field(default_factory=Constraints)


class SearchRequest(BaseModel):
    """Grid-search every portfolio of the selected strategies that meets the criteria."""
    slugs: list[str] = Field(..., min_length=2)
    max_weight: float = 0.0        # percent cap per strategy; 0 = no cap
    use_cpcv: bool = False
    n_samples: int = Field(5000, ge=200, le=50000)
    min_strats: int = 0            # min number of strategies carrying weight; 0 = unbounded
    max_strats: int = 0            # max number of strategies carrying weight; 0 = unbounded
    common_window: bool = False    # score each book only where all its legs overlap
    min_start_year: int = 0        # keep only strategies with data since ≤ this year; 0 = off
    constraints: Constraints = Field(default_factory=Constraints)


class DetailRequest(BaseModel):
    """Open one chosen candidate (its weight vector) and return the full dossier."""
    slugs: list[str] = Field(..., min_length=2)
    weights: dict[str, float]      # slug -> weight
    common_window: bool = False    # backtest only where all the book's legs overlap
    constraints: Constraints = Field(default_factory=Constraints)


class SaveRequest(DetailRequest):
    name: str = Field(..., min_length=1)


class StrategyRow(BaseModel):
    slug: str
    name: str
    category: str
    sharpe: float | None = None
    cagr: float | None = None
    max_dd: float | None = None
    pbo: float | None = None
    verdict: str | None = None
    n_obs: int = 0
    start_year: int | None = None


class Health(BaseModel):
    status: str
    n_strategies: int
    strategy_root: str
