"""Create a new strategy folder that conforms to the Rigor standard."""

from __future__ import annotations

from pathlib import Path

from . import layout, templates


def create_strategy(
    root: Path, category: str, slug: str, *, name: str | None = None,
    baseline: bool = False, overwrite: bool = False,
) -> Path:
    """Scaffold the standard files for a strategy.

    ``baseline=True`` scaffolds the raw version under
    ``strategies/Baseline/<category>/<slug>/`` (version ``v0``); otherwise the
    production version under ``strategies/<category>/<slug>/`` (version ``v1``).
    """
    if not layout.is_valid_slug(slug):
        raise ValueError(f"invalid slug {slug!r} — use snake_case starting with a letter")
    if not layout.is_valid_category(category):
        raise ValueError(f"invalid category {category!r} — use snake_case")

    display_name = name or slug.replace("_", " ").title()
    path = layout.strategy_dir(root, category, slug, baseline=baseline)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (use overwrite=True to replace)")

    version = "v0" if baseline else "v1"
    parts = [layout.STRATEGIES_ROOT]
    if baseline:
        parts.append(layout.BASELINE_ROOT)
    parts += [category, slug]
    run_path = "/".join(parts)

    (path / layout.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    (path / "strategy.py").write_text(
        templates.STRATEGY_PY.format(name=display_name, slug=slug, category=category),
        encoding="utf-8",
    )
    (path / "README.md").write_text(
        templates.README_MD.format(
            name=display_name, slug=slug, category=category, run_path=run_path
        ),
        encoding="utf-8",
    )
    (path / "config.json").write_text(
        templates.config_json(display_name, slug, category, version=version), encoding="utf-8"
    )
    (path / layout.ARTIFACTS_DIR / ".gitkeep").write_text("", encoding="utf-8")
    return path
