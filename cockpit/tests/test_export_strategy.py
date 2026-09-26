"""Tests for portfolio_io.export_as_strategy.

All tests are hermetic — they write only into ``tmp_path`` (pytest's isolated
temporary directory).  No file is ever written into the repo's ``strategies/``
tree.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from portfolio_engine.portfolio_io import _safe_slug, export_as_strategy

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MEMBERS = [
    {"slug": "shipping_bdi", "weight": 0.40},
    {"slug": "cat_bond_reinsurer", "weight": 0.35},
    {"slug": "crypto_funding_rate", "weight": 0.25},
]

_PORTFOLIO: dict = {
    "name": "Core Alpha",
    "slug": "core_alpha",
    "created": "2026-06-12",
    "as_of": "2026-06-11",
    "allocation_mode": "inverse_vol",
    "strategies": _MEMBERS,
    "metrics": {"sharpe": 1.42, "cagr": 0.12, "max_drawdown": -0.05},
    "beta": {},
}


def _export(portfolio: dict, dest: Path, **kwargs: object) -> Path:
    """Thin wrapper so tests are concise."""
    return export_as_strategy(portfolio, dest, **kwargs)


# ---------------------------------------------------------------------------
# 1. Required files exist after export
# ---------------------------------------------------------------------------


def test_required_files_created(tmp_path: Path) -> None:
    folder = _export(_PORTFOLIO, tmp_path)

    assert folder == tmp_path / "core_alpha"
    assert (folder / "config.json").exists(), "config.json missing"
    assert (folder / "strategy.py").exists(), "strategy.py missing"
    assert (folder / "README.md").exists(), "README.md missing"


# ---------------------------------------------------------------------------
# 2. config.json is valid JSON with required keys + members block
# ---------------------------------------------------------------------------


def test_config_json_keys(tmp_path: Path) -> None:
    folder = _export(_PORTFOLIO, tmp_path)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))

    # Required top-level keys (per layout.py convention + spec)
    for key in ("name", "slug", "category", "status", "version", "extra"):
        assert key in cfg, f"config.json missing key {key!r}"

    assert cfg["status"] == "idle"
    assert cfg["version"] == "v1"
    assert cfg["category"] == "portfolio"

    # Slug is valid: matches [a-z][a-z0-9_]*
    import re

    assert re.fullmatch(r"[a-z][a-z0-9_]*", cfg["slug"]), (
        f"slug {cfg['slug']!r} fails [a-z][a-z0-9_]*"
    )

    # extra block records members + weights
    extra = cfg["extra"]
    assert "members" in extra, "extra block missing 'members'"
    member_slugs = {m["slug"] for m in extra["members"]}
    assert member_slugs == {"shipping_bdi", "cat_bond_reinsurer", "crypto_funding_rate"}

    total_weight = sum(m["weight"] for m in extra["members"])
    assert abs(total_weight - 1.0) < 1e-6, f"weights don't sum to 1: {total_weight}"


# ---------------------------------------------------------------------------
# 3. Generated strategy.py parses and defines a top-level build() function
# ---------------------------------------------------------------------------


def test_strategy_py_parseable_and_has_build(tmp_path: Path) -> None:
    folder = _export(_PORTFOLIO, tmp_path)
    src = (folder / "strategy.py").read_text(encoding="utf-8")

    # Must parse as valid Python
    tree = ast.parse(src)  # raises SyntaxError on bad code

    # Must define a top-level function named 'build'
    top_level_funcs = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "build" in top_level_funcs, (
        f"strategy.py has no top-level 'build' function; found: {top_level_funcs}"
    )

    # Compile check (catches name errors that parse misses)
    compile(src, "<strategy.py>", "exec")


# ---------------------------------------------------------------------------
# 4. overwrite=False raises on existing non-empty folder; overwrite=True succeeds
# ---------------------------------------------------------------------------


def test_overwrite_false_raises(tmp_path: Path) -> None:
    _export(_PORTFOLIO, tmp_path)  # first export creates the folder
    with pytest.raises(FileExistsError):
        _export(_PORTFOLIO, tmp_path, overwrite=False)


def test_overwrite_true_succeeds(tmp_path: Path) -> None:
    _export(_PORTFOLIO, tmp_path)  # first export
    # Should not raise:
    folder = _export(_PORTFOLIO, tmp_path, overwrite=True)
    assert (folder / "config.json").exists()


# ---------------------------------------------------------------------------
# 5. _safe_slug produces a valid slug from arbitrary names
# ---------------------------------------------------------------------------


def test_safe_slug_valid_output() -> None:
    import re

    _valid = re.compile(r"^[a-z][a-z0-9_]*$")

    test_cases = [
        "My Cool Portfolio!",
        "  Core Alpha  ",
        "3-Signal Combo",
        "123",
        "Cat Bond & Reinsurer — R4",
        "",
        "A",
    ]
    for name in test_cases:
        result = _safe_slug(name)
        assert _valid.fullmatch(result), (
            f"_safe_slug({name!r}) -> {result!r} does not match [a-z][a-z0-9_]*"
        )


def test_safe_slug_specific() -> None:
    assert _safe_slug("My Cool Portfolio!") == "my_cool_portfolio"


# ---------------------------------------------------------------------------
# 6. Slug override and category parameter are respected
# ---------------------------------------------------------------------------


def test_custom_slug_and_category(tmp_path: Path) -> None:
    folder = _export(
        _PORTFOLIO, tmp_path, slug="my_custom_slug", category="macro"
    )
    assert folder == tmp_path / "my_custom_slug"
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert cfg["slug"] == "my_custom_slug"
    assert cfg["category"] == "macro"


def test_invalid_slug_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slug"):
        _export(_PORTFOLIO, tmp_path, slug="1invalid")


# ---------------------------------------------------------------------------
# 7. README.md contains member slugs and run instructions
# ---------------------------------------------------------------------------


def test_readme_content(tmp_path: Path) -> None:
    folder = _export(_PORTFOLIO, tmp_path)
    readme = (folder / "README.md").read_text(encoding="utf-8")

    for m in _MEMBERS:
        assert m["slug"] in readme, f"README missing member slug {m['slug']!r}"

    assert "rigor run" in readme, "README missing 'rigor run' instructions"
    assert "core_alpha" in readme, "README missing the strategy slug"
