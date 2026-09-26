"""Publish-gate: verify README headline-metric claims against committed summary.json.

Checks ONLY the three headline metrics (Sharpe, CAGR, Max Drawdown) to avoid
false-flagging incidental numbers in prose.  Designed as advisory (PASS/WARN/FAIL);
set ``--strict`` on the CLI to treat WARNs as failures.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rigor.project.layout import BASELINE_ROOT, STRATEGIES_ROOT, artifact_paths

# ---------------------------------------------------------------------------
# Finding codes
# ---------------------------------------------------------------------------
CODE_PASS = "PASS"
CODE_MISSING_SUMMARY = "MISSING_SUMMARY"
CODE_UNSOURCED_NUMBER = "UNSOURCED_NUMBER"
CODE_NO_HEADLINE = "NO_HEADLINE_IN_PROSE"

_LEVEL_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class GateFinding:
    """A single check result for one metric claim."""

    code: str
    level: str  # "PASS" | "WARN" | "FAIL"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    """Aggregated findings for one strategy."""

    slug: str
    path: str
    findings: list[GateFinding] = field(default_factory=list)

    @property
    def worst_level(self) -> str:
        if not self.findings:
            return "PASS"
        return max(self.findings, key=lambda f: _LEVEL_ORDER.get(f.level, 0)).level

    @property
    def passed(self) -> bool:
        return _LEVEL_ORDER.get(self.worst_level, 0) == 0


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
# Regex helpers for extracting headline metrics from README prose
# ---------------------------------------------------------------------------

# Sharpe: "Sharpe 2.83", "Sharpe of 2.83", "Sharpe | 2.83", "Sharpe: 2.83"
# We deliberately capture only the FIRST numeric token after the keyword.
_RE_SHARPE = re.compile(
    r"(?i)\bsharpe\b"               # keyword (case-insensitive)
    r"(?:\s*(?:of|ratio|:|\|))?"    # optional connector
    r"\s+"                          # whitespace
    r"([+-]?\d+(?:\.\d+)?)",        # number (group 1)
)

# CAGR: "CAGR 3.75%", "CAGR of 3.75%", "CAGR: 3.75%"
_RE_CAGR = re.compile(
    r"(?i)\bCAGR\b"
    r"(?:\s*(?:of|:|\|))?"
    r"\s+"
    r"([+-]?\d+(?:\.\d+)?)%",
)

# MaxDD / Max Drawdown / MaxDrawdown: "MaxDD −19.9%", "Max Drawdown -19.9%", "MDD -1.85%"
# Also matches "MaxDD | -1.85%".  The minus sign may be a Unicode en-dash (−).
_RE_MAXDD = re.compile(
    r"(?i)\b(?:max\s*draw\s*down|maxdd|max\s*dd|mdd)\b"
    r"(?:\s*(?:of|:|\|))?"
    r"\s+"
    r"([+\-−]?\d+(?:\.\d+)?)%",
)

# Normalise Unicode minus and en-dash to ASCII minus.
_MINUS_RE = re.compile(r"[−–]")


def _parse_num(s: str) -> float:
    return float(_MINUS_RE.sub("-", s))


def _extract_claims(readme_text: str) -> dict[str, float | None]:
    """Return the FIRST occurrence of each headline metric in the prose.

    Two-column table lines look like "Sharpe | 2.83 | 1.59".  We capture only
    the FIRST numeric token after the keyword, which is the strategy-under-review
    column, not a comparison column — so we never false-flag the second value.
    """
    sharpe_m = _RE_SHARPE.search(readme_text)
    cagr_m = _RE_CAGR.search(readme_text)
    maxdd_m = _RE_MAXDD.search(readme_text)

    return {
        "sharpe": _parse_num(sharpe_m.group(1)) if sharpe_m else None,
        "cagr": _parse_num(cagr_m.group(1)) if cagr_m else None,
        "max_drawdown": _parse_num(maxdd_m.group(1)) if maxdd_m else None,
    }


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def check_readme(
    strategy_dir: Path | str,
    *,
    sharpe_tol: float = 0.15,
    pct_tol: float = 0.5,
) -> GateReport:
    """Check headline-metric claims in README.md against artifacts summary.json.

    Args:
        strategy_dir: Path to the strategy folder (contains README.md + artifacts/).
        sharpe_tol:   Absolute tolerance for Sharpe comparison.
        pct_tol:      Tolerance in percentage points for CAGR and MaxDD comparisons.

    Returns:
        A :class:`GateReport` with one finding per metric checked.
    """
    strategy_dir = Path(strategy_dir)
    slug = strategy_dir.name
    report = GateReport(slug=slug, path=str(strategy_dir))

    # -- 1. Load summary.json ------------------------------------------------
    summary_path = artifact_paths(strategy_dir, slug)["summary"]
    if not summary_path.exists():
        report.findings.append(
            GateFinding(
                code=CODE_MISSING_SUMMARY,
                level="WARN",
                message=f"summary.json not found at {summary_path}",
                detail={"path": str(summary_path)},
            )
        )
        return report

    raw = summary_path.read_text(encoding="utf-8", errors="replace")
    # summary.json may contain bare NaN literals (Python json module rejects them
    # by default).  Replace with null so we can load cleanly.
    raw_safe = re.sub(r"\bNaN\b", "null", raw)
    data = json.loads(raw_safe)
    metrics: dict[str, Any] = data.get("metrics", {})

    stored_sharpe: float | None = metrics.get("sharpe")
    stored_cagr: float | None = metrics.get("cagr")  # fraction, e.g. 0.0375
    stored_maxdd: float | None = metrics.get("max_drawdown")  # fraction, e.g. -0.199

    # -- 2. Parse README.md --------------------------------------------------
    readme_path = strategy_dir / "README.md"
    if not readme_path.exists():
        report.findings.append(
            GateFinding(
                code=CODE_NO_HEADLINE,
                level="PASS",
                message="README.md not found; no claims to check.",
                detail={},
            )
        )
        return report

    # READMEs are occasionally saved in cp1252 (Windows) — read leniently so a
    # stray non-UTF-8 byte can never crash the gate for a contributor.
    readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
    claims = _extract_claims(readme_text)

    any_claim = False
    all_ok = True

    # -- 3. Sharpe -----------------------------------------------------------
    if claims["sharpe"] is not None:
        any_claim = True
        claimed = claims["sharpe"]
        if stored_sharpe is not None:
            diff = abs(claimed - stored_sharpe)
            if diff > sharpe_tol:
                all_ok = False
                report.findings.append(
                    GateFinding(
                        code=CODE_UNSOURCED_NUMBER,
                        level="WARN",
                        message=(
                            f"Sharpe claim {claimed:.4f} differs from stored"
                            f" {stored_sharpe:.4f} by {diff:.4f} (tol {sharpe_tol})"
                        ),
                        detail={
                            "metric": "sharpe",
                            "claimed": claimed,
                            "stored": stored_sharpe,
                            "diff": diff,
                            "tolerance": sharpe_tol,
                        },
                    )
                )

    # -- 4. CAGR -------------------------------------------------------------
    if claims["cagr"] is not None:
        any_claim = True
        claimed_pct = claims["cagr"]  # e.g. 3.75
        if stored_cagr is not None:
            stored_pct = stored_cagr * 100  # convert fraction → pct
            diff = abs(claimed_pct - stored_pct)
            if diff > pct_tol:
                all_ok = False
                report.findings.append(
                    GateFinding(
                        code=CODE_UNSOURCED_NUMBER,
                        level="WARN",
                        message=(
                            f"CAGR claim {claimed_pct:.2f}% differs from stored"
                            f" {stored_pct:.2f}% by {diff:.2f}pp (tol {pct_tol}pp)"
                        ),
                        detail={
                            "metric": "cagr",
                            "claimed_pct": claimed_pct,
                            "stored_pct": stored_pct,
                            "diff_pp": diff,
                            "tolerance_pp": pct_tol,
                        },
                    )
                )

    # -- 5. Max Drawdown -----------------------------------------------------
    if claims["max_drawdown"] is not None:
        any_claim = True
        # Normalise to negative magnitude: "-19.9" or "19.9" → -19.9
        claimed_raw = claims["max_drawdown"]
        claimed_pct = -abs(claimed_raw)
        if stored_maxdd is not None:
            stored_pct = stored_maxdd * 100  # fraction → pct, already negative
            diff = abs(claimed_pct - stored_pct)
            if diff > pct_tol:
                all_ok = False
                report.findings.append(
                    GateFinding(
                        code=CODE_UNSOURCED_NUMBER,
                        level="WARN",
                        message=(
                            f"MaxDD claim {claimed_pct:.2f}% differs from stored"
                            f" {stored_pct:.2f}% by {diff:.2f}pp (tol {pct_tol}pp)"
                        ),
                        detail={
                            "metric": "max_drawdown",
                            "claimed_pct": claimed_pct,
                            "stored_pct": stored_pct,
                            "diff_pp": diff,
                            "tolerance_pp": pct_tol,
                        },
                    )
                )

    # -- 6. Summary finding --------------------------------------------------
    if not any_claim:
        report.findings.append(
            GateFinding(
                code=CODE_NO_HEADLINE,
                level="PASS",
                message="No headline Sharpe/CAGR/MaxDD claims detected in README prose.",
                detail={},
            )
        )
    elif all_ok:
        report.findings.append(
            GateFinding(
                code=CODE_PASS,
                level="PASS",
                message="All detected headline claims reconcile with summary.json.",
                detail={
                    "sharpe_claimed": claims["sharpe"],
                    "cagr_claimed_pct": claims["cagr"],
                    "maxdd_claimed_pct": claims["max_drawdown"],
                },
            )
        )

    return report


# ---------------------------------------------------------------------------
# Enumerate all strategies
# ---------------------------------------------------------------------------


def check_all(root: Path | None = None) -> list[GateReport]:
    """Run :func:`check_readme` for every production strategy in the repo.

    Baseline strategies (under ``strategies/Baseline/``) are skipped; they
    intentionally have minimal READMEs with no headline claims.
    """
    if root is None:
        root = _repo_root()
    strategies_root = root / STRATEGIES_ROOT

    reports: list[GateReport] = []
    for config_path in sorted(strategies_root.glob("*/*/config.json")):
        # Skip Baseline subtree.
        rel = config_path.parent.relative_to(strategies_root)
        if rel.parts and rel.parts[0] == BASELINE_ROOT:
            continue
        reports.append(check_readme(config_path.parent))
    return reports


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(reports: list[GateReport]) -> str:
    """Return a concise human-readable summary of all gate reports."""
    lines: list[str] = []
    warn_fail = [r for r in reports if r.worst_level in ("WARN", "FAIL")]
    ok = [r for r in reports if r.worst_level == "PASS"]

    lines.append(f"Thesis gate: {len(reports)} strategies checked")
    lines.append(f"  PASS: {len(ok)}   WARN/FAIL: {len(warn_fail)}")

    if warn_fail:
        lines.append("")
        lines.append("Issues:")
        for rep in warn_fail:
            lines.append(f"  [{rep.worst_level}] {rep.slug}  ({rep.path})")
            for f in rep.findings:
                if f.level in ("WARN", "FAIL"):
                    lines.append(f"        {f.code}: {f.message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point.

    Returns:
        0 on advisory success (or when all checks pass).
        1 when ``--strict`` is set and any WARN/FAIL findings exist.
    """
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="thesis_gate",
        description="Check README headline-metric claims against summary.json.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Repo root (auto-detected)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN findings as failures (exit 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON output instead of prose.",
    )
    args = parser.parse_args(argv)

    reports = check_all(root=args.root)

    if args.as_json:
        import dataclasses

        print(json.dumps([dataclasses.asdict(r) for r in reports], indent=2))
    else:
        print(format_report(reports))

    has_issues = any(r.worst_level in ("WARN", "FAIL") for r in reports)
    if args.strict and has_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
