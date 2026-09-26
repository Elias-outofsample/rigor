"""``/api/portfolio`` — build / save the multi-strategy book.

Mirrors the portfolio surface of an earlier in-house cockpit API (build / methods / save / saved),
but the ★ Build endpoint returns the whole dossier in one call (allocate → backtest →
validate → gate → analyse) so the browser renders it without a second round-trip.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from portfolio_engine import METHODS

from .. import engine
from ..schemas import BuildRequest, DetailRequest, SaveRequest, SearchRequest

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_METHODS = ["robust (grid)", *METHODS, "kelly", "regime_dependent"]


@router.get("/methods")
def methods() -> dict:
    """Available weighting methods (robust grid search + the engine's modes)."""
    return {"methods": _METHODS}


@router.post("/search")
def search(req: SearchRequest) -> dict:
    """Grid-search every portfolio of the selection that meets the criteria; list them."""
    try:
        return engine.search(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/detail")
def detail(req: DetailRequest) -> dict:
    """Open one chosen candidate (its weights) and return the full dossier."""
    try:
        return engine.detail(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/build")
def build(req: BuildRequest) -> dict:
    """Direct single allocation (a chosen method) → full dossier. Kept for parity."""
    try:
        return engine.build(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/save")
def save(req: SaveRequest) -> dict:
    """Build then persist to ``portfolios/<name>/`` (definition + artifacts + HTML report)."""
    try:
        return engine.save(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/saved")
def saved() -> dict:
    """Slugs of previously saved portfolios."""
    return {"portfolios": engine.saved()}
