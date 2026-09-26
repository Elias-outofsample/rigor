"""Tests for rigor.project.thesis_gate.

All tests are hermetic — they use tmp_path and never touch the real repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigor.project.thesis_gate import (
    CODE_MISSING_SUMMARY,
    CODE_NO_HEADLINE,
    CODE_PASS,
    CODE_UNSOURCED_NUMBER,
    check_readme,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    tmp_path: Path,
    slug: str,
    *,
    readme: str,
    metrics: dict | None = None,
    write_summary: bool = True,
) -> Path:
    """Scaffold a minimal strategy directory under tmp_path."""
    strat_dir = tmp_path / "strategies" / "test_cat" / slug
    strat_dir.mkdir(parents=True)
    (strat_dir / "config.json").write_text(
        json.dumps({"name": slug, "version": "v1"}), encoding="utf-8"
    )
    (strat_dir / "README.md").write_text(readme, encoding="utf-8")

    if write_summary:
        artifacts = strat_dir / "artifacts"
        artifacts.mkdir()
        default_metrics: dict = {
            "sharpe": 0.76,
            "cagr": 0.0375,  # 3.75 %
            "max_drawdown": -0.199,  # -19.9 %
            "volatility": 0.05,
        }
        if metrics is not None:
            default_metrics.update(metrics)
        summary = {
            "as_of": "2026-01-01",
            "name": slug,
            "metrics": default_metrics,
            "verdict": {"verdict": "PROMOTE", "grade": "ROBUST", "score": 75.0, "gates": []},
        }
        (artifacts / f"{slug}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    return strat_dir


# ---------------------------------------------------------------------------
# Test 1: Matching claims → no UNSOURCED_NUMBER findings
# ---------------------------------------------------------------------------


README_MATCH = """\
# My Strategy

## Results

| metric | this port | original |
|---|---|---|
| Sharpe | 0.76 | 1.59 |
| CAGR | 3.75% | 2.49% |
| MaxDD | −19.9% | — |

Sharpe 0.76 is strong; CAGR 3.75% beats the benchmark; MaxDD -19.9% is acceptable.
"""


def test_matching_claims_pass(tmp_path: Path) -> None:
    """README claims that match summary.json should produce no UNSOURCED_NUMBER."""
    strat_dir = _make_strategy(tmp_path, "alpha_match", readme=README_MATCH)
    report = check_readme(strat_dir)

    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert unsourced == [], f"Expected no UNSOURCED_NUMBER; got: {unsourced}"

    # Worst level must be PASS.
    assert report.worst_level == "PASS"
    assert report.passed is True

    # Should have a final PASS finding.
    assert any(f.code == CODE_PASS for f in report.findings)


def test_two_column_table_no_false_flag(tmp_path: Path) -> None:
    """The second column value (1.59) must NOT be flagged even though it is far from 0.76."""
    # summary has sharpe=0.76; the table second column is 1.59.
    # The gate should capture 0.76 (first token) and ignore 1.59.
    strat_dir = _make_strategy(
        tmp_path,
        "two_col",
        readme="| Sharpe | 0.76 | 1.59 |\n| CAGR | 3.75% | 2.49% |\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    report = check_readme(strat_dir)
    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert unsourced == [], f"Second column falsely flagged: {unsourced}"


# ---------------------------------------------------------------------------
# Test 2: Contradictory Sharpe → UNSOURCED_NUMBER WARN
# ---------------------------------------------------------------------------


README_BAD_SHARPE = """\
# Inflated Strategy

Sharpe 9.99 looks amazing! CAGR 3.75% and MaxDD -19.9%.
"""


def test_contradictory_sharpe_flagged(tmp_path: Path) -> None:
    """A README claiming Sharpe 9.99 when stored sharpe~0.76 → UNSOURCED_NUMBER WARN."""
    strat_dir = _make_strategy(tmp_path, "bad_sharpe", readme=README_BAD_SHARPE)
    report = check_readme(strat_dir)

    sharpe_flags = [
        f
        for f in report.findings
        if f.code == CODE_UNSOURCED_NUMBER and f.detail.get("metric") == "sharpe"
    ]
    assert len(sharpe_flags) == 1, f"Expected exactly 1 sharpe flag; got {report.findings}"
    assert sharpe_flags[0].level == "WARN"
    assert sharpe_flags[0].detail["claimed"] == pytest.approx(9.99)
    assert sharpe_flags[0].detail["stored"] == pytest.approx(0.76, abs=0.01)

    assert report.worst_level == "WARN"
    assert report.passed is False


# ---------------------------------------------------------------------------
# Test 3: Missing summary.json → MISSING_SUMMARY
# ---------------------------------------------------------------------------


def test_missing_summary(tmp_path: Path) -> None:
    """Strategy without artifacts/summary.json should emit MISSING_SUMMARY."""
    strat_dir = _make_strategy(
        tmp_path, "no_summary", readme="Sharpe 1.23\n", write_summary=False
    )
    report = check_readme(strat_dir)

    codes = [f.code for f in report.findings]
    assert CODE_MISSING_SUMMARY in codes

    miss = next(f for f in report.findings if f.code == CODE_MISSING_SUMMARY)
    assert miss.level == "WARN"


# ---------------------------------------------------------------------------
# Test 4: No detectable headline claims → NO_HEADLINE_IN_PROSE (PASS-level)
# ---------------------------------------------------------------------------


def test_no_headline_claims(tmp_path: Path) -> None:
    """README with no Sharpe/CAGR/MaxDD mentions → NO_HEADLINE_IN_PROSE, still PASS."""
    strat_dir = _make_strategy(
        tmp_path,
        "no_claims",
        readme="# Strategy\nThis strategy uses breakout signals. No metrics stated here.\n",
    )
    report = check_readme(strat_dir)

    codes = [f.code for f in report.findings]
    assert CODE_NO_HEADLINE in codes
    assert report.worst_level == "PASS"
    assert CODE_UNSOURCED_NUMBER not in codes


# ---------------------------------------------------------------------------
# Test 5: main() CLI — strict mode exits 1, advisory mode exits 0
# ---------------------------------------------------------------------------


def _make_repo_tree(tmp_path: Path) -> Path:
    """Build a minimal repo-tree that check_all() can enumerate."""
    # A fake .git marker so _repo_root() can find the root.
    (tmp_path / ".git").mkdir()

    # One good strategy.
    _make_strategy(
        tmp_path / "strategies" / "test_cat" / "good",
        "good",
        readme="Sharpe 0.76, CAGR 3.75%, MaxDD -19.9%\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    # Relocate: _make_strategy nests under tmp_path; we need them under the
    # fake-root tmp_path.  Rebuild directly.
    return tmp_path


def _scaffold_repo(base: Path) -> None:
    """Directly write both a passing and a failing strategy into base."""
    # Passing strategy.
    good = base / "strategies" / "test_cat" / "good_strat"
    good.mkdir(parents=True, exist_ok=True)
    (good / "config.json").write_text("{}", encoding="utf-8")
    (good / "README.md").write_text(
        "| Sharpe | 0.76 | 1.59 |\n| CAGR | 3.75% | 2.49% |\n| MaxDD | -19.9% | — |\n",
        encoding="utf-8",
    )
    (good / "artifacts").mkdir(exist_ok=True)
    (good / "artifacts" / "good_strat_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "sharpe": 0.76,
                    "cagr": 0.0375,
                    "max_drawdown": -0.199,
                    "volatility": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )

    # Failing strategy — Sharpe is wildly wrong.
    bad = base / "strategies" / "test_cat" / "bad_strat"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "config.json").write_text("{}", encoding="utf-8")
    (bad / "README.md").write_text("Sharpe 9.99\n", encoding="utf-8")
    (bad / "artifacts").mkdir(exist_ok=True)
    (bad / "artifacts" / "bad_strat_summary.json").write_text(
        json.dumps({"metrics": {"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199}}),
        encoding="utf-8",
    )


def test_main_strict_returns_nonzero(tmp_path: Path) -> None:
    """--strict mode must return non-zero when a WARN exists."""
    (tmp_path / ".git").mkdir()
    _scaffold_repo(tmp_path)
    result = main(["--root", str(tmp_path), "--strict"])
    assert result != 0, "Expected non-zero exit in strict mode with WARN findings"


def test_main_advisory_returns_zero(tmp_path: Path) -> None:
    """Non-strict (advisory) mode must return 0 even when WARNs exist."""
    (tmp_path / ".git").mkdir()
    _scaffold_repo(tmp_path)
    result = main(["--root", str(tmp_path)])
    assert result == 0, "Expected exit 0 in advisory mode regardless of WARNs"


# ---------------------------------------------------------------------------
# Test 6: Tolerance edge-cases
# ---------------------------------------------------------------------------


def test_sharpe_within_tolerance_passes(tmp_path: Path) -> None:
    """Claimed Sharpe exactly at tolerance boundary should pass."""
    # stored=0.76, claimed=0.76+0.14=0.90 — within default tol=0.15
    strat_dir = _make_strategy(
        tmp_path,
        "tol_edge",
        readme="Sharpe 0.90\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    report = check_readme(strat_dir)
    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert unsourced == [], "Within-tolerance Sharpe should not be flagged"


def test_sharpe_just_outside_tolerance_flagged(tmp_path: Path) -> None:
    """Claimed Sharpe just outside tolerance must be flagged."""
    # stored=0.76, claimed=0.76+0.16=0.92 — outside default tol=0.15
    strat_dir = _make_strategy(
        tmp_path,
        "tol_exceed",
        readme="Sharpe 0.92\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    report = check_readme(strat_dir, sharpe_tol=0.15)
    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert len(unsourced) == 1


def test_cagr_pct_conversion(tmp_path: Path) -> None:
    """CAGR stored as fraction 0.0375 must compare correctly to prose '3.75%'."""
    strat_dir = _make_strategy(
        tmp_path,
        "cagr_conv",
        readme="CAGR 3.75%\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    report = check_readme(strat_dir)
    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert unsourced == [], "CAGR 3.75% vs stored 0.0375 should reconcile"


def test_maxdd_sign_normalisation(tmp_path: Path) -> None:
    """MaxDD claimed as positive '19.9%' must normalise to -19.9 before comparing."""
    strat_dir = _make_strategy(
        tmp_path,
        "maxdd_sign",
        # Positive sign in prose — some authors write "MaxDD 19.9%" not "-19.9%"
        readme="Max Drawdown 19.9%\n",
        metrics={"sharpe": 0.76, "cagr": 0.0375, "max_drawdown": -0.199},
    )
    report = check_readme(strat_dir)
    unsourced = [f for f in report.findings if f.code == CODE_UNSOURCED_NUMBER]
    assert unsourced == [], "Positive MaxDD in prose should normalise and reconcile"


def test_nan_in_summary_json(tmp_path: Path) -> None:
    """summary.json containing bare NaN literals must load without error."""
    strat_dir = tmp_path / "strategies" / "cat" / "nan_strat"
    strat_dir.mkdir(parents=True)
    (strat_dir / "config.json").write_text("{}", encoding="utf-8")
    (strat_dir / "README.md").write_text("No claims here.\n", encoding="utf-8")
    (strat_dir / "artifacts").mkdir()
    # Write raw NaN as Python/JS convention (not valid strict JSON).
    (strat_dir / "artifacts" / "nan_strat_summary.json").write_text(
        '{"metrics": {"sharpe": NaN, "cagr": NaN, "max_drawdown": NaN}}',
        encoding="utf-8",
    )
    # Should not raise.
    report = check_readme(strat_dir)
    assert report is not None
