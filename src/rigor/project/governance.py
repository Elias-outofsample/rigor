"""Controlled-vocabulary governance layer for Rigor strategy configs.

Governance metadata lives OPTIONALLY inside each strategy's ``config.json``
under a ``"governance"`` sub-key, so the existing config loader ignores it and
nothing breaks.  This module defines the vocabulary, validates it when present,
suggests sensible defaults, and rolls the whole book up into a summary.

Usage (advisory — never crashes on a weird config)::

    from rigor.project.governance import check_all, format_report, main

    result = check_all()
    print(format_report(result))

CLI::

    python -m rigor.project.governance [--root PATH] [--strict] [--json]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..data.config import find_repo_root
from . import layout

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

LIFECYCLE: tuple[str, ...] = (
    "research", "backtested", "validated", "paper", "live", "retired",
)

ASSET_CLASSES: tuple[str, ...] = (
    "equity", "crypto", "fx", "commodity", "rates", "credit",
    "multi_asset", "macro", "volatility",
)

EDGE_TYPES: tuple[str, ...] = (
    "mean_reversion", "momentum", "carry", "short_vol", "long_convexity",
    "trend", "seasonal", "event_driven", "microstructure", "relative_value",
    "unclassified",
)

# Map existing ``status`` field → lifecycle when no explicit lifecycle is set.
LEGACY_STATUS_TO_LIFECYCLE: dict[str, str] = {
    "idle": "backtested",
    "paper": "paper",
    "live": "live",
}

# Governance fields that accept a vocabulary-constrained scalar string.
_VOCAB_FIELDS: dict[str, tuple[str, ...]] = {
    "lifecycle": LIFECYCLE,
    "asset_class": ASSET_CLASSES,
    "edge_type": EDGE_TYPES,
}

# Mapping from strategy category → suggested asset class.
_CATEGORY_TO_ASSET_CLASS: dict[str, str] = {
    "crypto": "crypto",
    "macro": "macro",
    "carry": "credit",
    "intraday": "equity",
    "breakout": "equity",
    "trend_following": "equity",
    "momentum": "equity",
    "mean_reversion": "equity",
    "seasonal": "equity",
    "event_driven": "equity",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GovIssue:
    """A single governance finding for a strategy config."""

    code: str
    level: str   # "ERROR" | "WARN" | "INFO"
    field: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def suggest_asset_class(category: str) -> str:
    """Return a deterministic ASSET_CLASSES value for a strategy category.

    Unknown categories fall back to ``"equity"``.
    """
    return _CATEGORY_TO_ASSET_CLASS.get(category, "equity")


def read_config(strategy_dir: Path | str) -> dict | None:
    """Read and return the config.json for *strategy_dir*, or ``None`` on error."""
    path = Path(strategy_dir) / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_governance(config: dict) -> list[GovIssue]:
    """Validate governance metadata in *config* against the controlled vocabularies.

    If a ``"governance"`` sub-key exists, every recognised vocabulary field is
    checked; out-of-vocabulary values produce an ERROR.  Missing recommended
    fields produce WARN/INFO issues regardless of whether ``"governance"`` is
    present (advisory only, never ERROR).

    This function never raises — a malformed config just produces more issues.
    """
    issues: list[GovIssue] = []
    gov = config.get("governance")

    if not isinstance(gov, dict):
        # No governance block at all — emit advisory notices.
        issues.append(GovIssue(
            code="MISSING_GOVERNANCE",
            level="WARN",
            field="governance",
            message="No 'governance' sub-key found in config. Consider adding lifecycle, "
                    "asset_class, and edge_type.",
        ))
        return issues

    # Vocabulary checks for recognised scalar fields.
    for field, vocab in _VOCAB_FIELDS.items():
        value = gov.get(field)
        if value is None:
            issues.append(GovIssue(
                code="MISSING_GOVERNANCE",
                level="INFO",
                field=field,
                message=f"governance.{field} not set. Suggested values: {vocab}",
            ))
        elif not isinstance(value, str) or value not in vocab:
            issues.append(GovIssue(
                code="INVALID_VOCAB",
                level="ERROR",
                field=field,
                message=(
                    f"governance.{field} = {value!r} is not in the controlled vocabulary. "
                    f"Allowed: {vocab}"
                ),
            ))

    # Cross-link validation: must be str or list[str] when present.
    for link_field in ("linked_thesis", "linked_research"):
        value = gov.get(link_field)
        if value is None:
            continue
        if isinstance(value, str):
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            continue
        issues.append(GovIssue(
            code="INVALID_VOCAB",
            level="ERROR",
            field=link_field,
            message=(
                f"governance.{link_field} must be a string or list of strings, "
                f"got {type(value).__name__!r}."
            ),
        ))

    return issues


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich(
    config: dict,
    *,
    lifecycle: str | None = None,
    asset_class: str | None = None,
    edge_type: str | None = None,
    linked_thesis: str | list[str] | None = None,
    linked_research: str | list[str] | None = None,
) -> dict:
    """Return a new config dict with a ``"governance"`` sub-dict added/merged.

    Defaults are derived from the existing ``status`` and ``category`` fields
    when not explicitly provided.  All existing top-level config keys are
    preserved unchanged.
    """
    result = copy.deepcopy(config)

    # Resolve defaults from the existing config.
    if lifecycle is None:
        status = config.get("status", "")
        lifecycle = LEGACY_STATUS_TO_LIFECYCLE.get(status, "backtested")

    if asset_class is None:
        category = config.get("category", "")
        asset_class = suggest_asset_class(category)

    # Merge with any existing governance block.
    gov: dict = copy.deepcopy(result.get("governance") or {})
    gov["lifecycle"] = lifecycle
    gov["asset_class"] = asset_class

    if edge_type is not None:
        gov["edge_type"] = edge_type
    if linked_thesis is not None:
        gov["linked_thesis"] = linked_thesis
    if linked_research is not None:
        gov["linked_research"] = linked_research

    result["governance"] = gov
    return result


# ---------------------------------------------------------------------------
# Book-level roll-up
# ---------------------------------------------------------------------------

def _iter_production_configs(root: Path):
    """Yield (slug, config_dict) for every production strategy under *root*."""
    sroot = layout.strategies_root(root)
    if not sroot.is_dir():
        return
    for cfg_file in sorted(sroot.rglob("config.json")):
        rel = cfg_file.parent.relative_to(sroot)
        if layout.is_baseline_rel(rel):
            continue
        cfg = None
        with contextlib.suppress(OSError, json.JSONDecodeError):
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if cfg is None:
            continue
        slug = cfg.get("slug", cfg_file.parent.name)
        yield slug, cfg


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (temp file + os.replace) so a crash/sync can't half-write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def backfill_all(root: Path | None = None, *, dry_run: bool = False) -> list[str]:
    """Add a default ``governance`` block to every production config that lacks one.

    Purely additive — existing keys are never changed (``enrich`` deep-copies and only
    appends a ``governance`` sub-dict). Returns the slugs that were (or would be) updated.
    """
    root = root or find_repo_root()
    sroot = layout.strategies_root(root)
    updated: list[str] = []
    if not sroot.is_dir():
        return updated
    for cfg_file in sorted(sroot.rglob("config.json")):
        if layout.is_baseline_rel(cfg_file.parent.relative_to(sroot)):
            continue
        cfg = None
        with contextlib.suppress(OSError, json.JSONDecodeError):
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if cfg is None or cfg.get("governance"):
            continue
        slug = cfg.get("slug", cfg_file.parent.name)
        updated.append(slug)
        if not dry_run:
            _atomic_write_json(cfg_file, enrich(cfg))
    return updated


def summarize(root: Path | None = None) -> dict:
    """Walk all production strategy configs and return book-level governance counts.

    Returns a dict with keys:
    - ``by_lifecycle``    : {lifecycle_value: count}
    - ``by_asset_class``  : {asset_class_value: count}
    - ``by_edge_type``    : {edge_type_value: count}
    - ``n_total``         : total production strategy count
    - ``n_with_explicit_governance``: strategies that have a ``governance`` key
    - ``vocab_error_slugs``: list of slugs with INVALID_VOCAB ERRORs
    """
    root = Path(root) if root else find_repo_root()

    by_lifecycle: dict[str, int] = {}
    by_asset_class: dict[str, int] = {}
    by_edge_type: dict[str, int] = {}
    n_total = 0
    n_explicit = 0
    error_slugs: list[str] = []

    for slug, cfg in _iter_production_configs(root):
        n_total += 1
        gov = cfg.get("governance")
        if isinstance(gov, dict):
            n_explicit += 1

        # Resolve effective lifecycle.
        lc = (gov or {}).get("lifecycle") or LEGACY_STATUS_TO_LIFECYCLE.get(
            cfg.get("status", ""), "backtested"
        )
        by_lifecycle[lc] = by_lifecycle.get(lc, 0) + 1

        # Resolve effective asset class.
        ac = (gov or {}).get("asset_class") or suggest_asset_class(cfg.get("category", ""))
        by_asset_class[ac] = by_asset_class.get(ac, 0) + 1

        # Edge type only counts when explicitly set.
        et = (gov or {}).get("edge_type", "unclassified")
        by_edge_type[et] = by_edge_type.get(et, 0) + 1

        # Check for vocab errors.
        issues = validate_governance(cfg)
        if any(i.level == "ERROR" for i in issues):
            error_slugs.append(slug)

    return {
        "by_lifecycle": by_lifecycle,
        "by_asset_class": by_asset_class,
        "by_edge_type": by_edge_type,
        "n_total": n_total,
        "n_with_explicit_governance": n_explicit,
        "vocab_error_slugs": error_slugs,
    }


def check_all(root: Path | None = None) -> dict:
    """Return ``{summary, issues}`` where *issues* maps slug → [GovIssue, ...]
    (only slugs with ERROR or WARN — INFO-only slugs are omitted).
    """
    root = Path(root) if root else find_repo_root()
    summary = summarize(root)
    issues: dict[str, list[GovIssue]] = {}
    for slug, cfg in _iter_production_configs(root):
        slug_issues = [i for i in validate_governance(cfg) if i.level != "INFO"]
        if slug_issues:
            issues[slug] = slug_issues
    return {"summary": summary, "issues": issues}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(result: dict) -> str:
    """Return a concise human-readable governance report string."""
    s = result["summary"]
    lines: list[str] = [
        "=== Rigor Governance Report ===",
        f"Production strategies : {s['n_total']}",
        f"With explicit gov block: {s['n_with_explicit_governance']}",
        "",
        "Lifecycle breakdown:",
    ]
    for k, v in sorted(s["by_lifecycle"].items()):
        lines.append(f"  {k:<18} {v}")
    lines += ["", "Asset class breakdown:"]
    for k, v in sorted(s["by_asset_class"].items()):
        lines.append(f"  {k:<18} {v}")
    lines += ["", "Edge type breakdown:"]
    for k, v in sorted(s["by_edge_type"].items()):
        lines.append(f"  {k:<18} {v}")

    error_slugs = s["vocab_error_slugs"]
    if error_slugs:
        lines += ["", f"Vocabulary ERRORs ({len(error_slugs)} slug(s)):"]
        for slug in error_slugs:
            lines.append(f"  {slug}")

    issues: dict = result.get("issues", {})
    if issues:
        lines += ["", "Issues (WARN+):"]
        for slug, slug_issues in sorted(issues.items()):
            for issue in slug_issues:
                lines.append(f"  [{issue.level}] {slug}: {issue.field} — {issue.message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m rigor.project.governance [--root PATH] [--strict] [--json]``."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-governance",
        description="Controlled-vocabulary governance check for the strategy book.",
    )
    parser.add_argument("--root", default=None, help="Repo root (default: auto-detect)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any vocabulary ERROR is found (default: advisory only, exit 0).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Add a default governance block to configs that lack one, then report.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else None
    if args.backfill:
        updated = backfill_all(root)
        print(f"governance backfill: added a default block to {len(updated)} config(s)")
    result = check_all(root)

    if args.as_json:
        # GovIssue objects are not JSON-serialisable; convert to plain dicts.
        serialisable = {
            "summary": result["summary"],
            "issues": {
                slug: [
                    {"code": i.code, "level": i.level, "field": i.field, "message": i.message}
                    for i in issue_list
                ]
                for slug, issue_list in result["issues"].items()
            },
        }
        print(json.dumps(serialisable, indent=2))
    else:
        print(format_report(result))

    if args.strict and result["summary"]["vocab_error_slugs"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
