"""Enforced optimize gate: every strategy must declare a real, sweepable param_grid.

A strategy is only *optimisable* — in-house via ``rigor optimize`` and by the
external optimiser — if it declares which parameters are tunable and
over what ranges.  That surface is the strategy's ``param_grid()`` method: a
``dict[str, list]`` whose cartesian product is the search space.  Until now that
was purely optional — ``StrategyBase.param_grid`` defaults to ``{}`` (a single
run), so a strategy could be written, backtested and committed with **no tunable
surface at all** and every gate still passed.  ~half the catalog drifted that
way (mostly ported strategies that reproduced one committed configuration).

This module turns "declares a sweepable grid" into an **enforced, ratcheting
invariant**, exactly like :mod:`rigor.project.audit` (artifact coherence) and
:mod:`rigor.project.promotion_gate` (the verdict):

* **Static, data-free.**  It parses each ``strategy.py`` (AST) and inspects the
  literal returned by ``param_grid()`` — no import, no data fetch, no cache — so
  it runs in milliseconds against the whole catalog and never flakes on a
  missing EODHD key.
* **Compliant** = a literal ``dict`` with >=1 axis carrying >=2 values, every
  axis a non-empty list of type-consistent values.  A grid computed at runtime
  (non-literal) is UNVERIFIABLE and given the benefit of the doubt (reported,
  not blocked) — a runtime contract test is the place to pin those down.
* **Grandfathered.**  The strategies that predate the gate and still lack a grid
  are listed in ``optimize_gate_baseline.json`` (accepted backfill debt).  They
  WARN but do not block.  A *new* or *regressed* non-compliant strategy that is
  NOT in that list FAILS the build — this is what stops the bleeding.
* **Ratchet.**  As each grandfathered strategy is backfilled with a real grid,
  it becomes compliant and the gate flags it as removable from the baseline —
  the list only ever shrinks.  Regenerate with ``--update-baseline``.
* **Exemptions.**  A strategy with a genuinely empty tunable surface (a fixed
  calendar/seasonal pattern with no knobs) sets ``optimize_exempt_reason`` in its
  ``config.json`` — a reviewed, explicit opt-out, never a silent empty grid.

CLI::

    python -m rigor.project.optimize_gate               # check; exit 1 on new violations
    python -m rigor.project.optimize_gate --json        # machine-readable
    python -m rigor.project.optimize_gate --strict      # also fail on grandfathered debt
    python -m rigor.project.optimize_gate --update-baseline   # snapshot current debt

Exit code is ``1`` if any non-grandfathered, non-exempt strategy lacks a
sweepable ``param_grid()`` (or, under ``--strict``, if any debt remains at all),
otherwise ``0``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .layout import BASELINE_ROOT, STRATEGIES_ROOT

# ---------------------------------------------------------------------------
# Classification vocabulary
# ---------------------------------------------------------------------------

#: A strategy.py whose param_grid() we can statically prove is sweepable.
_COMPLIANT: frozenset[str] = frozenset({"MULTI"})
#: Present but not statically decidable — reported, never blocking.
_UNVERIFIABLE: frozenset[str] = frozenset({"DYNAMIC", "NO_FILE", "PARSE_ERROR"})
#: Definitely not sweepable — blocks unless grandfathered or exempt.
_VIOLATION: frozenset[str] = frozenset({"NO_METHOD", "EMPTY", "SINGLE", "INCONSISTENT"})

#: The committed backfill-debt baseline lives next to this module.
_BASELINE_PATH = Path(__file__).with_name("optimize_gate_baseline.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """The optimize-gate outcome for a single strategy."""

    slug: str
    path: str
    status: str            # MULTI | DYNAMIC | NO_METHOD | EMPTY | SINGLE | INCONSISTENT | ...
    level: str             # PASS | EXEMPT | UNVERIFIABLE | WARN | FAIL
    grandfathered: bool
    message: str

    @property
    def failed(self) -> bool:
        """True when this strategy is a blocking failure (new/regressed violation)."""
        return self.level == "FAIL"

    @property
    def debt(self) -> bool:
        """True when this is a grandfathered violation (accepted, non-blocking debt)."""
        return self.level == "WARN"


# ---------------------------------------------------------------------------
# Static param_grid classification
# ---------------------------------------------------------------------------

def _direct_returns(fn: ast.FunctionDef) -> list[ast.expr]:
    """Return the value expressions of ``return`` statements in ``fn``'s OWN scope.

    Descends through ``if`` / ``for`` / ``with`` blocks but NOT into nested
    ``def`` / ``lambda`` — so a helper closure's ``return`` (e.g. the ``axis``
    helper some grids define) is not mistaken for the grid itself.
    """
    out: list[ast.expr] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                out.append(child.value)
            visit(child)

    visit(fn)
    return out


def _types_consistent(values: list[object]) -> bool:
    """Mirror the external optimiser's contract: numbers may mix int/float; else same type."""
    if len(values) < 2:
        return True
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return True
    first = type(values[0])
    return all(type(v) is first for v in values)


def _classify_grid_value(val: object) -> tuple[str, str]:
    """Classify an already-evaluated (literal) grid value."""
    if not isinstance(val, dict) or len(val) == 0:
        return "EMPTY", "param_grid() returns an empty dict — no tunable surface"
    multi_axes = 0
    for key, values in val.items():
        if not isinstance(key, str) or not key:
            return "INCONSISTENT", f"param key {key!r} is not a non-empty string"
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return "INCONSISTENT", f"axis {key!r} is not a non-empty list"
        if not _types_consistent(list(values)):
            return "INCONSISTENT", f"axis {key!r} has inconsistent value types"
        if len(values) >= 2:
            multi_axes += 1
    if multi_axes == 0:
        return "SINGLE", "param_grid() has no axis with >=2 values — nothing to sweep"
    return "MULTI", f"{len(val)} axis/axes, {multi_axes} sweepable"


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, SyntaxError, TypeError):
        return False


def _resolve_engine(strategy_py: Path, src: str) -> Path | None:
    """If ``strategy_py`` re-exports a shared engine, return the engine's file path.

    Thin re-export strategies (`from rigor.strategies.<engine> import Strategy`) carry
    no local param_grid; the grid lives on the shared engine class. Resolve the
    engine module file so the gate can classify the inherited grid.
    """
    m = re.search(r"rigor\.strategies\.([a-z_][a-z0-9_]*)", src)
    if not m:
        return None
    for parent in strategy_py.parents:
        cand = parent / "src" / "rigor" / "strategies" / f"{m.group(1)}.py"
        if cand.is_file():
            return cand
    return None


def classify_param_grid(strategy_py: Path) -> tuple[str, str]:
    """Statically classify a strategy's ``param_grid()``.

    Returns ``(status, message)`` where ``status`` is one of ``MULTI`` (sweepable
    literal), ``DYNAMIC`` (computed at runtime — unverifiable), ``NO_METHOD``,
    ``EMPTY``, ``SINGLE``, ``INCONSISTENT``, ``NO_FILE`` or ``PARSE_ERROR``.
    """
    if not strategy_py.is_file():
        return "NO_FILE", "no strategy.py found"
    try:
        src = strategy_py.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError) as exc:
        return "PARSE_ERROR", f"cannot parse strategy.py: {exc}"

    fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "param_grid":
            fn = node
            break
    if fn is None:
        # A thin re-export (`from rigor.strategies.<engine> import Strategy`) inherits
        # param_grid from the shared engine — follow it and classify the engine's grid.
        engine = _resolve_engine(strategy_py, src)
        if engine is not None:
            status, msg = classify_param_grid(engine)
            if status not in ("NO_METHOD", "NO_FILE", "PARSE_ERROR"):
                return status, f"inherited from shared engine {engine.stem}: {msg}"
        return "NO_METHOD", "no param_grid() override — inherits the empty default {}"

    returns = _direct_returns(fn)
    if not returns:
        return "EMPTY", "param_grid() returns nothing"

    literal_returns = [r for r in returns if _is_literal(r)]
    # If ANY literal branch is a sweepable grid, the strategy is optimisable.
    best_violation: tuple[str, str] | None = None
    for rv in literal_returns:
        status, msg = _classify_grid_value(ast.literal_eval(rv))
        if status == "MULTI":
            return "MULTI", msg
        if best_violation is None:
            best_violation = (status, msg)

    # A computed branch we cannot evaluate → benefit of the doubt (runtime check).
    if len(literal_returns) < len(returns):
        return "DYNAMIC", "param_grid() is computed at runtime — verify with a runtime test"

    return best_violation or ("EMPTY", "param_grid() returns no usable grid")


# ---------------------------------------------------------------------------
# Baseline (accepted backfill debt)
# ---------------------------------------------------------------------------

def load_baseline(path: Path | None = None) -> set[str]:
    """Return the set of grandfathered (accepted-debt) slugs."""
    path = path or _BASELINE_PATH
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    slugs = data.get("slugs", [])
    return {s for s in slugs if isinstance(s, str)}


def write_baseline(slugs: set[str], path: Path | None = None) -> None:
    """Snapshot the current backfill debt to the baseline file (sorted, stable)."""
    path = path or _BASELINE_PATH
    payload = {
        "_comment": (
            "Grandfathered strategies that predate rigor.project.optimize_gate and "
            "still lack a sweepable param_grid(). Accepted backfill debt: the gate "
            "WARNs on these but does not block. A NEW or REGRESSED non-compliant "
            "strategy not in this list FAILS CI. As each is backfilled, remove it "
            "here (the list only shrinks). Regenerate: "
            "python -m rigor.project.optimize_gate --update-baseline"
        ),
        "generated": date.today().isoformat(),
        "count": len(slugs),
        "slugs": sorted(slugs),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Repo-root discovery (mirrors promotion_gate)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Walk up to the directory holding ``.git``; fall back to the source layout."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    # No git checkout (e.g. a downloaded archive): src/rigor/project -> repository root.
    candidate = here.parents[3]
    if (candidate / "strategies").is_dir():
        return candidate
    raise FileNotFoundError("Could not locate repo root (.git not found)")


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_strategy(strategy_dir: Path | str, baseline: set[str]) -> GateResult:
    """Evaluate the optimize gate for a single strategy folder."""
    strategy_dir = Path(strategy_dir)
    slug = strategy_dir.name

    config: dict = {}
    cfg_path = strategy_dir / "config.json"
    if cfg_path.is_file():
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
    slug = config.get("slug") or slug
    grandfathered = slug in baseline

    reason = config.get("optimize_exempt_reason")
    if isinstance(reason, str) and reason.strip():
        return GateResult(
            slug=slug, path=str(strategy_dir), status="EXEMPT", level="EXEMPT",
            grandfathered=grandfathered, message=f"exempt: {reason.strip()}",
        )

    status, detail = classify_param_grid(strategy_dir / "strategy.py")

    if status in _COMPLIANT:
        level = "PASS"
    elif status in _UNVERIFIABLE:
        level = "UNVERIFIABLE"
    else:  # a real violation
        level = "WARN" if grandfathered else "FAIL"

    return GateResult(
        slug=slug, path=str(strategy_dir), status=status, level=level,
        grandfathered=grandfathered, message=detail,
    )


def check_all(root: Path | None = None, baseline: set[str] | None = None) -> list[GateResult]:
    """Run :func:`check_strategy` for every production strategy in the repo.

    Baseline strategies (under ``strategies/Baseline/``) are skipped — raw research
    references are never optimised independently.
    """
    if root is None:
        root = _repo_root()
    if baseline is None:
        baseline = load_baseline()
    strategies_root = Path(root) / STRATEGIES_ROOT

    results: list[GateResult] = []
    if not strategies_root.is_dir():
        return results
    for config_path in sorted(strategies_root.glob("*/*/config.json")):
        rel = config_path.parent.relative_to(strategies_root)
        if rel.parts and rel.parts[0] == BASELINE_ROOT:
            continue
        results.append(check_strategy(config_path.parent, baseline))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def stale_baseline_entries(results: list[GateResult], baseline: set[str]) -> list[str]:
    """Baseline slugs that are now compliant/exempt/gone — removable (ratchet)."""
    by_slug = {r.slug: r for r in results}
    stale: list[str] = []
    for slug in sorted(baseline):
        r = by_slug.get(slug)
        if r is None or r.status not in _VIOLATION:
            stale.append(slug)
    return stale


def format_report(results: list[GateResult], baseline: set[str], *, strict: bool) -> str:
    """Return a concise human-readable summary of the optimize gate."""
    failures = [r for r in results if r.failed]
    debt = [r for r in results if r.debt]
    passed = [r for r in results if r.level == "PASS"]
    exempt = [r for r in results if r.level == "EXEMPT"]
    unverifiable = [r for r in results if r.level == "UNVERIFIABLE"]
    stale = stale_baseline_entries(results, baseline)

    lines: list[str] = [
        f"Optimize gate: {len(results)} strategies checked"
        + ("  [STRICT]" if strict else ""),
        f"  sweepable (PASS) : {len(passed)}",
        f"  exempt           : {len(exempt)} (declared no tunable surface)",
        f"  unverifiable     : {len(unverifiable)} (runtime-computed grid)",
        f"  backfill debt    : {len(debt)} (grandfathered — WARN)",
        f"  failures         : {len(failures)}",
    ]
    if failures:
        lines.append("")
        lines.append("BLOCKING failures (no sweepable param_grid, not grandfathered/exempt):")
        for r in sorted(failures, key=lambda r: r.slug):
            lines.append(f"  [FAIL] {r.slug}  ({r.status})  {r.path}")
            lines.append(f"         {r.message}")
        lines.append("")
        lines.append("Fix: declare a real multi-value param_grid() (>=1 axis with >=2 "
                     "values), or set optimize_exempt_reason in config.json if the "
                     "strategy genuinely has no tunable surface.")
    if strict and debt:
        lines.append("")
        lines.append("STRICT: grandfathered backfill debt still present:")
        for r in sorted(debt, key=lambda r: r.slug):
            lines.append(f"  [DEBT] {r.slug}  ({r.status})")
    if stale:
        lines.append("")
        lines.append(f"Ratchet: {len(stale)} baseline entr(y/ies) now compliant/exempt/"
                     "gone — remove via --update-baseline:")
        lines.append("  " + ", ".join(stale[:20]) + (" ..." if len(stale) > 20 else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: ``python -m rigor.project.optimize_gate``."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-optimize-gate",
        description=(
            "Enforced optimize gate: every strategy must declare a sweepable "
            "param_grid() (or an explicit optimize_exempt_reason)."
        ),
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Repo root (default: auto-detect via .git).")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Emit machine-readable JSON instead of prose.")
    parser.add_argument("--strict", action="store_true",
                        help="Also fail on grandfathered backfill debt (post-backfill check).")
    parser.add_argument("--update-baseline", action="store_true", dest="update_baseline",
                        help="Snapshot current violations into the baseline and exit.")
    args = parser.parse_args(argv)

    root = args.root
    if args.update_baseline:
        # Recompute against an empty baseline so every current violation is captured.
        results = check_all(root=root, baseline=set())
        violations = {r.slug for r in results if r.status in _VIOLATION}
        write_baseline(violations)
        print(f"optimize-gate baseline updated: {len(violations)} strategies "
              f"({_BASELINE_PATH})")
        return 0

    baseline = load_baseline()
    results = check_all(root=root, baseline=baseline)

    if args.as_json:
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        print(format_report(results, baseline, strict=args.strict))

    blocking = any(r.failed for r in results)
    if args.strict:
        blocking = blocking or any(r.debt for r in results)
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
