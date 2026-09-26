"""``/api/strategies`` — the discoverable strategy book for the left rail."""
from __future__ import annotations

from fastapi import APIRouter

from .. import engine
from ..schemas import StrategyRow

router = APIRouter(tags=["strategies"])


@router.get("/strategies", response_model=list[StrategyRow])
def strategies() -> list[dict]:
    """Every discovered strategy with its headline stats + robustness verdict."""
    return engine.list_strategies()
