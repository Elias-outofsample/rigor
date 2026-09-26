"""Runtime configuration for the web cockpit (all env-overridable).

Env:
    COCKPIT_STRATEGY_ROOT   path to the Rigor ``strategies/`` book
                            (default: auto-discovered repo root / Strategy)
    COCKPIT_CORS_ORIGINS    comma-separated allowed origins (default: ``*``)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _find_strategy_root() -> Path:
    """Walk up for a repo root holding both ``strategies/`` and ``src/rigor/``."""
    here = Path(__file__).resolve()
    for d in (here, *here.parents):
        if (d / "strategies").is_dir() and (d / "src" / "rigor").is_dir():
            return d / "strategies"
    return Path.cwd() / "strategies"


class Settings:
    def __init__(self) -> None:
        env_root = os.environ.get("COCKPIT_STRATEGY_ROOT")
        self.strategy_root: Path = Path(env_root) if env_root else _find_strategy_root()
        origins = os.environ.get("COCKPIT_CORS_ORIGINS", "*")
        self.cors_origins: list[str] = [o.strip() for o in origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
