"""Optional-dependency helper for the analysis layer.

Heavy libraries (statsmodels, arch, hmmlearn, pandas-datareader) are optional.
``optional_import`` returns the module or ``None``; ``require`` raises a clear,
actionable message naming the extra to install. This keeps the core install
light and unbreakable while richer analysis is available on demand.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return None


def have(name: str) -> bool:
    return optional_import(name) is not None


def require(module: ModuleType | None, feature: str, *, package: str) -> ModuleType:
    if module is None:
        raise ImportError(
            f"{feature} needs the optional dependency '{package}'. "
            'Install the analysis extras with:  pip install -e ".[analysis]"'
        )
    return module
