"""FastAPI app for the web Portfolio Cockpit.

Run from the repo root::

    uvicorn cockpit_api.main:app --reload --port 8000   # from cockpit/
    # then open http://localhost:8000

The API lives under ``/api`` and the browser frontend (``cockpit/frontend``) is
served at ``/``. Configuration is in ``settings.py`` (env-overridable).
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, engine
from .routers import portfolio, strategies
from .schemas import Health
from .settings import get_settings

settings = get_settings()

app = FastAPI(
    title="quant-cockpit-api",
    version=__version__,
    description="Web Portfolio Cockpit over the Rigor engine (portfolio_engine) + Strategy book.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_frontend(request, call_next):
    """Never let the browser serve a stale frontend — the HTML/CSS/JS change often,
    and a cached app.js against a fresh index.html silently breaks features."""
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", n_strategies=len(engine.candidates()),
                  strategy_root=str(settings.strategy_root))


@app.get("/api/saved-report/{name}")
def saved_report(name: str) -> FileResponse:
    """Serve a saved portfolio's committed HTML tearsheet (slug-validated, no traversal)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(status_code=400, detail="invalid portfolio name")
    artifacts = settings.strategy_root.parent / "portfolios" / name / "artifacts"
    report = next(artifacts.glob(f"{name}_report.html"), None) if artifacts.is_dir() else None
    if report is None:
        raise HTTPException(status_code=404, detail="no report for that portfolio")
    return FileResponse(report)


app.include_router(strategies.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")

# Serve the browser cockpit at / (mounted last so it never shadows /api routes).
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
