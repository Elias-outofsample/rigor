"""Unit tests for rigor.project.regression — hermetic, no real git required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigor.project.regression import (
    Regression,
    baseline_summary,
    current_summary,
    diff_summaries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(*, sharpe=1.0, cagr=0.10, max_drawdown=-0.10,
                  volatility=0.12, score=70.0) -> dict:
    return {
        "metrics": {
            "sharpe": sharpe,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        },
        "verdict": {"score": score},
    }


def _severity(results: list[Regression], metric: str) -> str:
    for r in results:
        if r.metric == metric:
            return r.severity
    raise KeyError(metric)


# ---------------------------------------------------------------------------
# 1. Identical summaries → all UNCHANGED, no regression
# ---------------------------------------------------------------------------


def test_identical_summaries_all_unchanged():
    s = _make_summary()
    results = diff_summaries(s, s)
    assert results, "expected at least one metric result"
    non_info = [r for r in results if r.metric != "volatility"]
    assert all(r.severity == "UNCHANGED" for r in non_info), (
        "expected all metrics UNCHANGED for identical summaries"
    )
    assert not any(r.severity == "REGRESSION" for r in results)


# ---------------------------------------------------------------------------
# 2. Sharpe 1.5 → 1.2 (drop 0.3 > tol 0.05) → REGRESSION
# ---------------------------------------------------------------------------


def test_sharpe_drop_is_regression():
    old = _make_summary(sharpe=1.5)
    new = _make_summary(sharpe=1.2)
    results = diff_summaries(old, new)
    assert _severity(results, "sharpe") == "REGRESSION"


# ---------------------------------------------------------------------------
# 3. Sharpe 1.0 → 1.4 → IMPROVEMENT, not a regression
# ---------------------------------------------------------------------------


def test_sharpe_rise_is_improvement():
    old = _make_summary(sharpe=1.0)
    new = _make_summary(sharpe=1.4)
    results = diff_summaries(old, new)
    sev = _severity(results, "sharpe")
    assert sev == "IMPROVEMENT"
    assert not any(r.severity == "REGRESSION" for r in results)


# ---------------------------------------------------------------------------
# 4. verdict.score 80 → 60 (drop 20 > tol 5) → REGRESSION on score
# ---------------------------------------------------------------------------


def test_score_drop_is_regression():
    old = _make_summary(score=80.0)
    new = _make_summary(score=60.0)
    results = diff_summaries(old, new)
    assert _severity(results, "verdict.score") == "REGRESSION"


# ---------------------------------------------------------------------------
# 5a. max_drawdown -0.10 → -0.30 (more negative) → REGRESSION
# ---------------------------------------------------------------------------


def test_drawdown_worse_is_regression():
    old = _make_summary(max_drawdown=-0.10)
    new = _make_summary(max_drawdown=-0.30)
    results = diff_summaries(old, new)
    assert _severity(results, "max_drawdown") == "REGRESSION"


# ---------------------------------------------------------------------------
# 5b. max_drawdown -0.30 → -0.10 (less negative) → IMPROVEMENT
# ---------------------------------------------------------------------------


def test_drawdown_better_is_improvement():
    old = _make_summary(max_drawdown=-0.30)
    new = _make_summary(max_drawdown=-0.10)
    results = diff_summaries(old, new)
    assert _severity(results, "max_drawdown") == "IMPROVEMENT"


# ---------------------------------------------------------------------------
# 6. Missing / NaN fields → skipped without crashing
# ---------------------------------------------------------------------------


def test_missing_fields_skipped():
    old = {"metrics": {}, "verdict": {}}
    new = {"metrics": {}, "verdict": {}}
    results = diff_summaries(old, new)
    # No metric values present → no Regression objects returned (all skipped)
    assert isinstance(results, list)
    assert len(results) == 0


def test_nan_fields_skipped():
    # NaN encoded as float('nan') in the dict (as if parsed from JSON with NaN→None)
    old = {
        "metrics": {"sharpe": float("nan"), "cagr": 0.10, "max_drawdown": -0.10,
                    "volatility": 0.12},
        "verdict": {"score": 70.0},
    }
    new = _make_summary()
    results = diff_summaries(old, new)
    # sharpe missing in old (NaN) → skipped; other metrics still compared
    metrics = {r.metric for r in results}
    assert "sharpe" not in metrics, "NaN sharpe should be skipped"
    assert "cagr" in metrics


def test_partial_missing_no_crash():
    old = {"metrics": {"sharpe": 1.2}}    # only sharpe; no verdict
    new = {"metrics": {"sharpe": 1.3}}
    results = diff_summaries(old, new)
    assert len(results) == 1
    assert results[0].metric == "sharpe"
    assert results[0].severity == "IMPROVEMENT"


# ---------------------------------------------------------------------------
# 7. current_summary reads a summary written into a tmp_path strategy layout
# ---------------------------------------------------------------------------


def test_current_summary_reads_file(tmp_path: Path):
    slug = "test_strat"
    # Build the layout: <tmp_path>/<slug>/artifacts/<slug>_summary.json
    strat_dir = tmp_path / slug
    artifacts = strat_dir / "artifacts"
    artifacts.mkdir(parents=True)
    payload = _make_summary(sharpe=2.0)
    (artifacts / f"{slug}_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = current_summary(strat_dir, slug)
    assert result is not None
    assert result["metrics"]["sharpe"] == pytest.approx(2.0)


def test_current_summary_returns_none_if_absent(tmp_path: Path):
    strat_dir = tmp_path / "no_strat"
    strat_dir.mkdir()
    result = current_summary(strat_dir, "no_strat")
    assert result is None


# ---------------------------------------------------------------------------
# 8. baseline_summary returns None for bogus ref without raising
# ---------------------------------------------------------------------------


def test_baseline_summary_bogus_ref_returns_none(tmp_path: Path):
    """A non-existent git ref must return None, never raise."""
    strat_dir = tmp_path / "some_strat"
    strat_dir.mkdir()
    # tmp_path is NOT a git repo → _repo_root() will fail → returns None
    result = baseline_summary(strat_dir, "some_strat", ref="refs/heads/does-not-exist")
    assert result is None


def test_baseline_summary_no_git_returns_none(tmp_path: Path):
    """No .git ancestor → returns None without raising."""
    strat_dir = tmp_path / "s"
    strat_dir.mkdir()
    result = baseline_summary(strat_dir, "s", ref="origin/main")
    assert result is None


# ---------------------------------------------------------------------------
# Regression dataclass sanity
# ---------------------------------------------------------------------------


def test_regression_dataclass_fields():
    r = Regression(
        metric="sharpe",
        old=1.5,
        new=1.2,
        delta=-0.3,
        severity="REGRESSION",
        message="sharpe dropped 0.3",
    )
    assert r.metric == "sharpe"
    assert r.delta == pytest.approx(-0.3)
    assert r.severity == "REGRESSION"
