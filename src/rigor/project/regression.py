"""Regression-check: compare strategy summary.json artifacts between two git refs.

Detects metric degradation (Sharpe, CAGR, MDD, score) introduced by a code
change so CI can block or warn before a merge.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Regression:
    """One metric comparison between an old and new summary."""

    metric: str
    old: float
    new: float
    delta: float
    severity: str   # "REGRESSION" | "IMPROVEMENT" | "UNCHANGED"
    message: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_metric(d: dict, *keys: str) -> float | None:
    """Navigate a nested dict by key path; return None if missing or NaN."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    if isinstance(cur, float) and math.isnan(cur):
        return None
    try:
        v = float(cur)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _load_json_text(text: str) -> dict:
    """Parse JSON that may contain bare NaN tokens (Python json quirk: accepted)."""
    # Python's C-accelerated json module accepts bare NaN — but to be safe we
    # normalise them to null first so the result dict holds None, not float('nan').
    cleaned = re.sub(r'\bNaN\b', 'null', text)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def diff_summaries(
    old: dict,
    new: dict,
    *,
    sharpe_tol: float = 0.05,
    score_tol: float = 5.0,
    dd_tol: float = 0.005,
    cagr_tol: float = 0.005,
) -> list[Regression]:
    """Compare two summary dicts; return one Regression per metric.

    Rules
    -----
    sharpe       drop > sharpe_tol → REGRESSION; rise > sharpe_tol → IMPROVEMENT
    cagr         drop > cagr_tol   → REGRESSION; rise > cagr_tol   → IMPROVEMENT
    max_drawdown more-negative by > dd_tol → REGRESSION; less-negative → IMPROVEMENT
                 (e.g. -0.10 → -0.30 is REGRESSION; -0.30 → -0.10 is IMPROVEMENT)
    volatility   always UNCHANGED (informational)
    verdict.score drop > score_tol → REGRESSION; rise > score_tol → IMPROVEMENT
    Missing / NaN values are skipped silently.
    """
    results: list[Regression] = []

    def _compare(
        metric: str,
        old_val: float | None,
        new_val: float | None,
        tol: float,
        *,
        higher_is_better: bool = True,
        always_unchanged: bool = False,
    ) -> None:
        if old_val is None or new_val is None:
            return
        delta = new_val - old_val
        if always_unchanged:
            severity = "UNCHANGED"
            msg = f"{metric}: {old_val:.4g} → {new_val:.4g} (informational)"
        elif higher_is_better:
            if -delta > tol:   # value dropped
                severity = "REGRESSION"
                msg = f"{metric} dropped {-delta:.4g} (tol {tol}): {old_val:.4g} → {new_val:.4g}"
            elif delta > tol:
                severity = "IMPROVEMENT"
                msg = (
                    f"{metric} improved {delta:.4g} (tol {tol}): {old_val:.4g} → {new_val:.4g}"
                )
            else:
                severity = "UNCHANGED"
                msg = f"{metric}: {old_val:.4g} → {new_val:.4g}"
        else:
            # lower is better (max_drawdown: more negative = worse)
            if delta < -tol:   # became more negative → regression
                severity = "REGRESSION"
                msg = (
                    f"{metric} worsened {delta:.4g} (tol {tol}): {old_val:.4g} → {new_val:.4g}"
                )
            elif delta > tol:  # became less negative → improvement
                severity = "IMPROVEMENT"
                msg = (
                    f"{metric} improved {delta:.4g} (tol {tol}): {old_val:.4g} → {new_val:.4g}"
                )
            else:
                severity = "UNCHANGED"
                msg = f"{metric}: {old_val:.4g} → {new_val:.4g}"
        results.append(Regression(
            metric=metric,
            old=old_val,
            new=new_val,
            delta=delta,
            severity=severity,
            message=msg,
        ))

    _compare(
        "sharpe",
        _get_metric(old, "metrics", "sharpe"),
        _get_metric(new, "metrics", "sharpe"),
        sharpe_tol,
    )
    _compare(
        "cagr",
        _get_metric(old, "metrics", "cagr"),
        _get_metric(new, "metrics", "cagr"),
        cagr_tol,
    )
    _compare(
        "max_drawdown",
        _get_metric(old, "metrics", "max_drawdown"),
        _get_metric(new, "metrics", "max_drawdown"),
        dd_tol,
        higher_is_better=False,  # more-negative = worse
    )
    _compare(
        "volatility",
        _get_metric(old, "metrics", "volatility"),
        _get_metric(new, "metrics", "volatility"),
        0.0,
        always_unchanged=True,
    )
    _compare(
        "verdict.score",
        _get_metric(old, "verdict", "score"),
        _get_metric(new, "verdict", "score"),
        score_tol,
    )
    return results


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Walk up from this file until we find the directory containing '.git'."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not locate a .git directory above " + str(p))


def current_summary(strategy_dir: Path, slug: str) -> dict | None:
    """Load the working-tree summary JSON; return None if absent or unreadable."""
    paths = layout.artifact_paths(Path(strategy_dir), slug)
    summary_path = paths["summary"]
    if not summary_path.exists():
        return None
    try:
        return _load_json_text(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def baseline_summary(
    strategy_dir: Path,
    slug: str,
    ref: str = "origin/main",
) -> dict | None:
    """Load summary JSON from a git ref via ``git show``; never raises.

    Returns None if git is unavailable, the ref doesn't exist, the file isn't
    tracked at that ref, or JSON parsing fails.
    """
    try:
        repo_root = _repo_root()
    except FileNotFoundError:
        return None

    paths = layout.artifact_paths(Path(strategy_dir), slug)
    summary_path = paths["summary"]
    try:
        rel = summary_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None

    # git show uses forward slashes even on Windows
    git_path = rel.as_posix()
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{git_path}"],
            capture_output=True,
            cwd=str(repo_root),
            timeout=15,
        )
    except Exception:  # noqa: BLE001
        return None

    if proc.returncode != 0:
        return None

    try:
        return _load_json_text(proc.stdout.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# High-level scan
# ---------------------------------------------------------------------------


def regression_check(
    root: Path | None = None,
    *,
    ref: str = "origin/main",
) -> list[dict]:
    """For every production strategy, diff baseline (git ref) vs working tree.

    Returns a list of dicts::

        {
            "slug": str,
            "category": str,
            "regressions": list[Regression],
            "has_regression": bool,
            "has_baseline": bool,
        }

    Strategies under ``strategies/Baseline/`` are skipped.
    """
    if root is None:
        try:
            root = _repo_root()
        except FileNotFoundError:
            return []

    results: list[dict] = []
    strat_root = layout.strategies_root(root)
    if not strat_root.is_dir():
        return results

    for cfg_file in sorted(strat_root.rglob("config.json")):
        rel = cfg_file.parent.relative_to(strat_root)
        if layout.is_baseline_rel(rel):
            continue

        d = cfg_file.parent
        try:
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        slug = raw.get("slug", d.name)
        category = raw.get("category", d.parent.name)

        old = baseline_summary(d, slug, ref)
        new = current_summary(d, slug)

        regressions: list[Regression] = []
        if old is not None and new is not None:
            regressions = diff_summaries(old, new)

        results.append({
            "slug": slug,
            "category": category,
            "regressions": regressions,
            "has_regression": any(r.severity == "REGRESSION" for r in regressions),
            "has_baseline": old is not None,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(results: list[dict]) -> str:
    """Return a concise text report listing only strategies with regressions."""
    bad = [r for r in results if r.get("has_regression")]
    if not bad:
        checked = len(results)
        return f"regression-check: all {checked} strategies OK (no regressions detected)"

    lines = [f"regression-check: {len(bad)} strategy/strategies with regressions:\n"]
    for item in bad:
        lines.append(f"  [{item['category']}] {item['slug']}")
        for reg in item["regressions"]:
            if reg.severity == "REGRESSION":
                lines.append(f"    REGRESSION  {reg.message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """CLI: rigor regression-check [--root PATH] [--ref REF] [--json]."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-regression",
        description="Compare strategy metrics between a git ref and the working tree.",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="Repo root (default: auto-detect)"
    )
    parser.add_argument(
        "--ref", default="origin/main", help="Git ref for baseline (default: origin/main)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON to stdout"
    )
    args = parser.parse_args(argv)

    results = regression_check(root=args.root, ref=args.ref)

    if args.json:
        def _ser(obj):
            if isinstance(obj, Regression):
                return obj.__dict__
            raise TypeError(type(obj))

        print(json.dumps(results, default=_ser, indent=2))
    else:
        print(format_report(results))

    return 1 if any(r.get("has_regression") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
