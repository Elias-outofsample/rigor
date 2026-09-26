"""Validate that a strategy folder conforms to the Rigor standard.

Static checks only (no code execution / no network): file presence, config
shape, slug/category rules, and that strategy.py exposes build(config, data).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..strategy import StrategyConfig
from . import layout
from .run import _CONFIG_FIELDS


@dataclass
class ValidationResult:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _has_build_factory(strategy_py: Path) -> bool:
    try:
        tree = ast.parse(strategy_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body)


def validate_strategy(strategy_path: str | Path) -> ValidationResult:
    path = Path(strategy_path)
    res = ValidationResult(path=str(path))

    if not path.is_dir():
        res.errors.append("not a directory")
        return res

    for fname in layout.REQUIRED_FILES:
        if not (path / fname).is_file():
            res.errors.append(f"missing required file: {fname}")

    cfg_file = path / "config.json"
    raw: dict = {}
    if cfg_file.is_file():
        try:
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            res.errors.append(f"config.json is not valid JSON: {e}")

    if raw:
        slug = raw.get("slug")
        category = raw.get("category")
        if not slug:
            res.errors.append("config.json missing 'slug'")
        else:
            if not layout.is_valid_slug(slug):
                res.errors.append(f"slug {slug!r} is not snake_case")
            if slug != path.name:
                res.errors.append(f"slug {slug!r} != folder name {path.name!r}")
        if not category:
            res.errors.append("config.json missing 'category'")
        elif category not in layout.CANONICAL_CATEGORIES:
            res.warnings.append(
                f"category {category!r} not in canonical set {layout.CANONICAL_CATEGORIES}"
            )
        # StrategyConfig must construct from the declared fields.
        try:
            StrategyConfig(**{k: v for k, v in raw.items() if k in _CONFIG_FIELDS})
        except (TypeError, ValueError) as e:
            res.errors.append(f"config.json does not build a valid StrategyConfig: {e}")

    strat_py = path / "strategy.py"
    if strat_py.is_file() and not _has_build_factory(strat_py):
        res.errors.append("strategy.py must define a top-level build(config, data) function")

    if not layout.artifacts_dir(path).exists():
        res.warnings.append("artifacts/ directory missing (created on run)")

    return res
