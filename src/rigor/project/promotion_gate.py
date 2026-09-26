"""Enforced promotion gate: a deployable strategy may not carry a REJECT verdict.

The validation battery already computes a promotion ``verdict`` (PROMOTE /
CONDITIONAL / REJECT — see :mod:`rigor.validation.verdict`) and ``rigor run``
persists it into each strategy's ``artifacts/<slug>_summary.json`` under the
``verdict.verdict`` key (see :func:`rigor.report.write_report_card`).  Until now
that verdict was *reported only*: nothing stopped a REJECT strategy from being
promoted to paper/live and shipped.

This module turns the verdict into an **enforced, status-aware gate**:

* It reads the verdict cheaply from committed ``summary.json`` data — no live
  data fetch is required.
* It only enforces for strategies whose lifecycle is in the **deployable**
  subset (paper / live).  Research / backtested / idle strategies are EXEMPT,
  so the whole research catalog passes vacuously today and the gate only bites
  the moment someone actually promotes a strategy.
* A deployable strategy with **no verdict recorded** (legacy artifact) is also
  exempt — the gate never fails on missing evidence, only on a recorded REJECT.

The lifecycle vocabulary and the legacy ``status`` → lifecycle mapping are
imported from :mod:`rigor.project.governance` so there is exactly one source of
truth for what "deployable" means.

CLI::

    python -m rigor.project.promotion_gate [--root PATH] [--json]

Exit code is ``1`` if any *deployable* strategy carries a REJECT verdict,
otherwise ``0``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .governance import LEGACY_STATUS_TO_LIFECYCLE
from .layout import BASELINE_ROOT, STRATEGIES_ROOT, artifact_paths

# ---------------------------------------------------------------------------
# Deployable vocabulary
# ---------------------------------------------------------------------------

#: Lifecycle states that count as "deployed / promoted" — these are the only
#: states the gate enforces against.  Everything else (research, backtested,
#: idle, candidate, retired, ...) is treated as not-yet-deployed and EXEMPT.
DEPLOYABLE_LIFECYCLES: frozenset[str] = frozenset({"paper", "live"})

#: The verdict string that blocks a promotion.
REJECT_VERDICT = "REJECT"

# Finding levels, ordered for "worst level" reductions.
_LEVEL_ORDER = {"EXEMPT": 0, "PASS": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """The promotion-gate outcome for a single strategy."""

    slug: str
    path: str
    lifecycle: str
    deployable: bool
    verdict: str | None        # PROMOTE | CONDITIONAL | REJECT | None (not recorded)
    level: str                 # "EXEMPT" | "PASS" | "FAIL"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """True when this strategy is a deployable REJECT (a blocking failure)."""
        return self.level == "FAIL"


# ---------------------------------------------------------------------------
# Repo-root discovery
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
# Verdict / lifecycle resolution
# ---------------------------------------------------------------------------

def resolve_lifecycle(config: dict) -> str:
    """Return the effective lifecycle for a strategy *config*.

    Resolution order mirrors :func:`rigor.project.governance.summarize`:

    1. An explicit ``governance.lifecycle`` string, if present.
    2. Otherwise the legacy ``status`` mapped through
       :data:`~rigor.project.governance.LEGACY_STATUS_TO_LIFECYCLE`
       (e.g. ``"idle"`` → ``"backtested"``, ``"paper"`` → ``"paper"``).
    3. Otherwise ``"backtested"`` (a safe, non-deployable default).

    A raw ``status`` of ``"paper"``/``"live"`` therefore resolves directly to a
    deployable lifecycle, so the gate enforces even when no explicit governance
    block has been added yet.
    """
    gov = config.get("governance")
    if isinstance(gov, dict):
        lc = gov.get("lifecycle")
        if isinstance(lc, str) and lc:
            return lc
    status = config.get("status", "")
    return LEGACY_STATUS_TO_LIFECYCLE.get(status, "backtested")


def read_verdict(summary_path: Path) -> str | None:
    """Return the recorded verdict string from a ``summary.json``, or ``None``.

    The verdict is persisted at ``verdict.verdict`` (see
    :func:`rigor.report.write_report_card`).  ``None`` means "no verdict recorded"
    — a missing/unreadable file, a legacy summary without a ``verdict`` block,
    or a malformed verdict — and is treated by the gate as EXEMPT, never a
    failure.  ``summary.json`` may contain bare ``NaN`` literals (the engine
    writes them); they are neutralised to ``null`` before parsing.
    """
    if not summary_path.is_file():
        return None
    try:
        raw = summary_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    raw_safe = re.sub(r"\bNaN\b", "null", raw)
    try:
        data = json.loads(raw_safe)
    except json.JSONDecodeError:
        return None
    verdict_block = data.get("verdict")
    if not isinstance(verdict_block, dict):
        return None
    verdict = verdict_block.get("verdict")
    return verdict if isinstance(verdict, str) else None


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_strategy(strategy_dir: Path | str) -> GateResult:
    """Evaluate the promotion gate for a single strategy folder.

    Returns a :class:`GateResult`.  The strategy FAILS only when it is both
    *deployable* (lifecycle in :data:`DEPLOYABLE_LIFECYCLES`) and carries a
    recorded REJECT verdict.  Non-deployable strategies, and deployable
    strategies with no recorded verdict, are EXEMPT (they pass).
    """
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

    lifecycle = resolve_lifecycle(config)
    deployable = lifecycle in DEPLOYABLE_LIFECYCLES

    summary_path = artifact_paths(strategy_dir, slug)["summary"]
    verdict = read_verdict(summary_path)

    if not deployable:
        return GateResult(
            slug=slug, path=str(strategy_dir), lifecycle=lifecycle,
            deployable=False, verdict=verdict, level="EXEMPT",
            message=f"lifecycle '{lifecycle}' is not deployable — exempt",
            detail={"verdict": verdict},
        )

    if verdict is None:
        return GateResult(
            slug=slug, path=str(strategy_dir), lifecycle=lifecycle,
            deployable=True, verdict=None, level="EXEMPT",
            message=("deployable but no verdict recorded — exempt "
                     "(legacy artifact; re-run to record a verdict)"),
            detail={"verdict": None},
        )

    if verdict == REJECT_VERDICT:
        return GateResult(
            slug=slug, path=str(strategy_dir), lifecycle=lifecycle,
            deployable=True, verdict=verdict, level="FAIL",
            message=(f"deployable (lifecycle '{lifecycle}') strategy carries a "
                     f"REJECT verdict — promotion is blocked"),
            detail={"verdict": verdict},
        )

    return GateResult(
        slug=slug, path=str(strategy_dir), lifecycle=lifecycle,
        deployable=True, verdict=verdict, level="PASS",
        message=f"deployable (lifecycle '{lifecycle}') with verdict {verdict}",
        detail={"verdict": verdict},
    )


# ---------------------------------------------------------------------------
# Enumerate all strategies
# ---------------------------------------------------------------------------

def check_all(root: Path | None = None) -> list[GateResult]:
    """Run :func:`check_strategy` for every production strategy in the repo.

    Baseline strategies (under ``strategies/Baseline/``) are skipped — they are
    raw research references, never independently deployed.
    """
    if root is None:
        root = _repo_root()
    strategies_root = Path(root) / STRATEGIES_ROOT

    results: list[GateResult] = []
    if not strategies_root.is_dir():
        return results
    for config_path in sorted(strategies_root.glob("*/*/config.json")):
        rel = config_path.parent.relative_to(strategies_root)
        if rel.parts and rel.parts[0] == BASELINE_ROOT:
            continue
        results.append(check_strategy(config_path.parent))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: list[GateResult]) -> str:
    """Return a concise human-readable summary of the promotion gate."""
    failures = [r for r in results if r.failed]
    deployable = [r for r in results if r.deployable]
    exempt = [r for r in results if not r.deployable]
    # Deployable-but-no-verdict are deployable yet EXEMPT; surface them too.
    deployable_exempt = [r for r in deployable if r.level == "EXEMPT"]

    lines: list[str] = [
        f"Promotion gate: {len(results)} strategies checked",
        f"  deployable : {len(deployable)} "
        f"(exempt/no-verdict: {len(deployable_exempt)})",
        f"  exempt     : {len(exempt)} (non-deployable lifecycle)",
        f"  failures   : {len(failures)}",
    ]
    if failures:
        lines.append("")
        lines.append("BLOCKING failures (deployable + REJECT):")
        for r in failures:
            lines.append(f"  [FAIL] {r.slug}  ({r.lifecycle})  {r.path}")
            lines.append(f"         {r.message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: ``python -m rigor.project.promotion_gate``.

    Returns ``1`` if any deployable strategy carries a REJECT verdict (blocking),
    otherwise ``0``.  This is an *enforced* gate — there is no advisory mode;
    the deployable subset is intentionally narrow so the research catalog never
    blocks the build.
    """
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-promotion-gate",
        description=(
            "Enforced promotion gate: a deployable (paper/live) strategy may "
            "not carry a REJECT validation verdict."
        ),
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Repo root (default: auto-detect via .git).")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    results = check_all(root=args.root)

    if args.as_json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))

    return 1 if any(r.failed for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
