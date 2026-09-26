"""Tests for rigor.project.promotion_gate.

All tests are hermetic — they build strategy folders under tmp_path and never
touch the real repo.

The gate enforces ONE rule: a *deployable* strategy (lifecycle in
``DEPLOYABLE_LIFECYCLES`` = {paper, live}) may not carry a REJECT verdict.
Everything else — non-deployable lifecycles, or deployable strategies with no
recorded verdict — is EXEMPT.
"""

from __future__ import annotations

import json
from pathlib import Path

from rigor.project.promotion_gate import (
    DEPLOYABLE_LIFECYCLES,
    check_all,
    check_strategy,
    main,
    read_verdict,
    resolve_lifecycle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    root: Path,
    slug: str,
    *,
    status: str = "idle",
    verdict: str | None = "PROMOTE",
    governance: dict | None = None,
    category: str = "test_cat",
    write_summary: bool = True,
    baseline: bool = False,
) -> Path:
    """Scaffold a minimal strategy folder and return its path.

    ``verdict=None`` writes a summary.json with no ``verdict`` block (legacy
    artifact). ``write_summary=False`` writes no summary.json at all.
    """
    parts = ["strategies"]
    if baseline:
        parts.append("Baseline")
    parts += [category, slug]
    strat_dir = root.joinpath(*parts)
    strat_dir.mkdir(parents=True)

    cfg: dict = {"slug": slug, "status": status}
    if governance is not None:
        cfg["governance"] = governance
    (strat_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    if write_summary:
        artifacts = strat_dir / "artifacts"
        artifacts.mkdir()
        summary: dict = {"name": slug, "metrics": {"sharpe": 1.0}}
        if verdict is not None:
            summary["verdict"] = {
                "verdict": verdict, "grade": "X", "score": 1.0, "gates": [],
            }
        (artifacts / f"{slug}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    return strat_dir


def _make_repo(root: Path) -> None:
    """Mark *root* as a repo so the gate's .git walk stops here."""
    (root / ".git").mkdir()


# ---------------------------------------------------------------------------
# Core rule: deployable + REJECT FAILS
# ---------------------------------------------------------------------------


def test_deployable_reject_fails(tmp_path: Path) -> None:
    """A paper-lifecycle strategy with a REJECT verdict is a blocking FAIL."""
    strat = _make_strategy(tmp_path, "bad", status="paper", verdict="REJECT")
    result = check_strategy(strat)
    assert result.deployable is True
    assert result.verdict == "REJECT"
    assert result.level == "FAIL"
    assert result.failed is True


def test_live_reject_fails(tmp_path: Path) -> None:
    """A live-lifecycle strategy with a REJECT verdict is a blocking FAIL."""
    strat = _make_strategy(tmp_path, "bad_live", status="live", verdict="REJECT")
    result = check_strategy(strat)
    assert result.level == "FAIL"
    assert result.failed is True


# ---------------------------------------------------------------------------
# Deployable + PROMOTE/CONDITIONAL PASSES
# ---------------------------------------------------------------------------


def test_deployable_promote_passes(tmp_path: Path) -> None:
    """A paper-lifecycle strategy with a PROMOTE verdict passes."""
    strat = _make_strategy(tmp_path, "good", status="paper", verdict="PROMOTE")
    result = check_strategy(strat)
    assert result.deployable is True
    assert result.verdict == "PROMOTE"
    assert result.level == "PASS"
    assert result.failed is False


def test_deployable_conditional_passes(tmp_path: Path) -> None:
    """CONDITIONAL is not REJECT, so a deployable CONDITIONAL passes."""
    strat = _make_strategy(tmp_path, "cond", status="live", verdict="CONDITIONAL")
    result = check_strategy(strat)
    assert result.level == "PASS"
    assert result.failed is False


# ---------------------------------------------------------------------------
# Non-deployable is EXEMPT (even when REJECT)
# ---------------------------------------------------------------------------


def test_idle_reject_exempt(tmp_path: Path) -> None:
    """An idle (research) strategy with REJECT is EXEMPT — the gate ignores it."""
    strat = _make_strategy(tmp_path, "research", status="idle", verdict="REJECT")
    result = check_strategy(strat)
    assert result.deployable is False
    assert result.lifecycle == "backtested"  # idle -> backtested, not deployable
    assert result.level == "EXEMPT"
    assert result.failed is False


# ---------------------------------------------------------------------------
# Legacy artifacts: deployable but NO verdict recorded -> EXEMPT
# ---------------------------------------------------------------------------


def test_deployable_no_verdict_block_exempt(tmp_path: Path) -> None:
    """Deployable strategy whose summary.json lacks a verdict block is EXEMPT."""
    strat = _make_strategy(tmp_path, "legacy", status="paper", verdict=None)
    result = check_strategy(strat)
    assert result.deployable is True
    assert result.verdict is None
    assert result.level == "EXEMPT"
    assert result.failed is False


def test_deployable_no_summary_exempt(tmp_path: Path) -> None:
    """Deployable strategy with no summary.json at all is EXEMPT (no evidence)."""
    strat = _make_strategy(tmp_path, "nosumm", status="paper", write_summary=False)
    result = check_strategy(strat)
    assert result.verdict is None
    assert result.level == "EXEMPT"
    assert result.failed is False


# ---------------------------------------------------------------------------
# Governance lifecycle overrides legacy status
# ---------------------------------------------------------------------------


def test_governance_lifecycle_promotes_idle_to_deployable(tmp_path: Path) -> None:
    """An explicit governance.lifecycle='live' makes an idle config deployable."""
    strat = _make_strategy(
        tmp_path, "gov_live", status="idle", verdict="REJECT",
        governance={"lifecycle": "live"},
    )
    result = check_strategy(strat)
    assert result.lifecycle == "live"
    assert result.deployable is True
    assert result.level == "FAIL"


def test_governance_lifecycle_research_is_exempt(tmp_path: Path) -> None:
    """governance.lifecycle='research' is non-deployable even if status='live'."""
    strat = _make_strategy(
        tmp_path, "gov_research", status="live", verdict="REJECT",
        governance={"lifecycle": "research"},
    )
    result = check_strategy(strat)
    assert result.lifecycle == "research"
    assert result.deployable is False
    assert result.level == "EXEMPT"


# ---------------------------------------------------------------------------
# resolve_lifecycle / read_verdict units
# ---------------------------------------------------------------------------


def test_resolve_lifecycle_mapping() -> None:
    """idle->backtested, paper->paper, live->live, unknown->backtested."""
    assert resolve_lifecycle({"status": "idle"}) == "backtested"
    assert resolve_lifecycle({"status": "paper"}) == "paper"
    assert resolve_lifecycle({"status": "live"}) == "live"
    assert resolve_lifecycle({"status": "whatever"}) == "backtested"
    assert resolve_lifecycle({}) == "backtested"
    assert resolve_lifecycle({"governance": {"lifecycle": "paper"}}) == "paper"


def test_deployable_set_excludes_research_states() -> None:
    """The deployable set is exactly {paper, live} — research states excluded."""
    assert set(DEPLOYABLE_LIFECYCLES) == {"paper", "live"}
    for state in ("research", "backtested", "validated", "retired", "idle"):
        assert state not in DEPLOYABLE_LIFECYCLES


def test_read_verdict_handles_nan_tokens(tmp_path: Path) -> None:
    """summary.json with bare NaN literals still parses and yields the verdict."""
    p = tmp_path / "summary.json"
    p.write_text(
        '{"verdict": {"verdict": "REJECT", "gates": [{"value": NaN}]}}',
        encoding="utf-8",
    )
    assert read_verdict(p) == "REJECT"


def test_read_verdict_missing_file_is_none(tmp_path: Path) -> None:
    """A missing summary.json yields None (treated as exempt downstream)."""
    assert read_verdict(tmp_path / "does_not_exist.json") is None


# ---------------------------------------------------------------------------
# check_all + main (exit codes)
# ---------------------------------------------------------------------------


def test_check_all_skips_baseline(tmp_path: Path) -> None:
    """Baseline strategies are not enforced even when deployable + REJECT."""
    _make_repo(tmp_path)
    _make_strategy(tmp_path, "base_bad", status="live", verdict="REJECT", baseline=True)
    _make_strategy(tmp_path, "prod_ok", status="idle", verdict="PROMOTE")
    results = check_all(root=tmp_path)
    slugs = {r.slug for r in results}
    assert "base_bad" not in slugs
    assert "prod_ok" in slugs


def test_main_passes_on_clean_catalog(tmp_path: Path, capsys) -> None:
    """Exit 0 when no deployable strategy carries a REJECT verdict."""
    _make_repo(tmp_path)
    _make_strategy(tmp_path, "research", status="idle", verdict="REJECT")  # exempt
    _make_strategy(tmp_path, "good_paper", status="paper", verdict="PROMOTE")  # pass
    rc = main(["--root", str(tmp_path)])
    assert rc == 0


def test_main_fails_on_deployable_reject(tmp_path: Path, capsys) -> None:
    """Exit 1 when a deployable strategy carries a REJECT verdict."""
    _make_repo(tmp_path)
    _make_strategy(tmp_path, "shipit", status="live", verdict="REJECT")
    rc = main(["--root", str(tmp_path)])
    assert rc == 1


def test_main_json_output(tmp_path: Path, capsys) -> None:
    """--json emits machine-readable output for every checked strategy."""
    _make_repo(tmp_path)
    _make_strategy(tmp_path, "s1", status="paper", verdict="REJECT")
    rc = main(["--root", str(tmp_path), "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["slug"] == "s1"
    assert parsed[0]["level"] == "FAIL"
    assert rc == 1
