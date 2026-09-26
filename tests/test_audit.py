"""Hermetic tests for rigor.project.audit.

All tests use tmp_path to build synthetic strategy dirs and do NOT rely on
any committed strategies/ contents — they must pass offline and independently
of what strategies happen to be checked in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rigor import metrics as _m
from rigor.project.audit import (
    AuditReport,
    Finding,
    audit_all,
    audit_strategy,
    format_report,
    main,
)

# ── helpers ───────────────────────────────────────────────────────────────────

_TRADING_DAYS = _m.TRADING_DAYS


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Create a deterministic daily returns series with a real (non-zero) std."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0005, 0.01, size=n)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(vals, index=dates, name="returns")


def _write_strategy(
    root: Path,
    category: str,
    slug: str,
    returns: pd.Series | None,
    summary: dict | None,
    *,
    write_config: bool = True,
    write_thesis: bool = True,
    write_positions: bool = True,
    ledger_exempt: bool = False,
) -> Path:
    """Write a minimal strategy folder structure under root/Strategy/<cat>/<slug>/.

    ``write_positions=True`` (default) writes a stub positions CSV so clean
    strategies don't trigger MISSING_LEDGER.  Pass ``write_positions=False``
    to test that finding explicitly.  ``ledger_exempt=True`` adds the flag to
    config.json so the audit suppresses ledger findings.
    """
    sdir = root / "strategies" / category / slug
    arts = sdir / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)

    # Stub thesis.pdf so a coherent strategy passes the MISSING_THESIS gate.
    # (audit only checks for presence; content is irrelevant.)
    if write_thesis and returns is not None and summary is not None:
        (arts / f"{slug}_thesis.pdf").write_bytes(b"%PDF-1.4 stub\n")

    if write_config:
        cfg = {
            "name": slug.replace("_", " ").title(),
            "slug": slug,
            "category": category,
            "status": "idle",
            "version": "v1",
        }
        if ledger_exempt:
            cfg["ledger_exempt"] = True
        (sdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    if returns is not None:
        df = returns.reset_index()
        df.columns = pd.Index(["date", "returns"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        (arts / f"{slug}_returns.csv").write_text(
            df.to_csv(index=False), encoding="utf-8"
        )

    if summary is not None:
        # json.dumps does NOT support NaN by default, but the real writer uses
        # allow_nan=True; use a replace trick for tests.
        text = json.dumps(summary, allow_nan=True)
        (arts / f"{slug}_summary.json").write_text(text, encoding="utf-8")

    # Stub positions CSV so clean strategies pass the MISSING_LEDGER gate.
    # The audit checks for *presence* first; a stub with real-ish columns is
    # enough for non-degenerate tests.
    if write_positions and returns is not None and summary is not None:
        positions_stub = (
            "entry_date,exit_date,symbol,entry_price,exit_price,"
            "max_adverse_excursion,max_favorable_excursion\n"
            "2020-01-02,2020-01-10,SPY,320.0,325.0,-0.01,0.02\n"
        )
        (arts / f"{slug}_positions.csv").write_text(positions_stub, encoding="utf-8")

    return sdir


def _summary_for(returns: pd.Series) -> dict:
    """Build a summary.json dict whose metrics exactly match *returns*."""
    ppy = _m.periods_per_year_of(returns.index)
    m = _m.compute_core_metrics(returns.values, ppy)
    return {
        "as_of": "2026-01-01",
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "name": "Test Strategy",
        "metrics": m,
        "verdict": {"score": 50, "gates": []},
    }


# ── test 1: coherent strategy → no ERROR ─────────────────────────────────────

def test_clean_strategy_ok(tmp_path: Path) -> None:
    """A coherent strategy (returns + matching summary) should audit clean."""
    r = _make_returns()
    sdir = _write_strategy(tmp_path, "momentum", "clean_strat", r, _summary_for(r))
    report = audit_strategy(sdir)
    assert isinstance(report, AuditReport)
    assert report.ok is True
    assert not any(f.severity == "ERROR" for f in report.findings)


# ── test 2: wrong sharpe → PERF_NE_CURVE ─────────────────────────────────────

def test_wrong_sharpe_yields_perf_ne_curve(tmp_path: Path) -> None:
    """A summary.json with a deliberately wrong sharpe must yield PERF_NE_CURVE."""
    r = _make_returns()
    bad_summary = _summary_for(r)
    bad_summary["metrics"]["sharpe"] = 9.9  # far from actual ~0.5
    sdir = _write_strategy(tmp_path, "momentum", "bad_sharpe", r, bad_summary)
    report = audit_strategy(sdir)
    assert not report.ok
    codes = [f.code for f in report.findings]
    assert "PERF_NE_CURVE" in codes


# ── test 3: all-zero returns → DEGENERATE_ARTIFACT ───────────────────────────

def test_allzero_returns_degenerate(tmp_path: Path) -> None:
    """An all-zero returns series (std == 0) must yield DEGENERATE_ARTIFACT."""
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    zero_r = pd.Series(np.zeros(300), index=dates, name="returns")
    # Build a plausible (but irrelevant) summary
    summary = {
        "as_of": "2026-01-01", "start": "2020-01-02", "end": "2021-12-31",
        "name": "Zero", "metrics": {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0,
                                    "volatility": 0.0, "n_obs": 300},
        "verdict": {"score": 0, "gates": []},
    }
    sdir = _write_strategy(tmp_path, "volatility", "zero_strat", zero_r, summary)
    report = audit_strategy(sdir)
    assert not report.ok
    codes = [f.code for f in report.findings]
    assert "DEGENERATE_ARTIFACT" in codes


# ── test 4: missing summary.json → MISSING_ARTIFACT ──────────────────────────

def test_missing_summary_yields_missing_artifact(tmp_path: Path) -> None:
    """A strategy folder with no summary.json must yield MISSING_ARTIFACT."""
    r = _make_returns()
    # pass summary=None so it is not written
    sdir = _write_strategy(tmp_path, "carry", "no_summary", r, summary=None)
    report = audit_strategy(sdir)
    assert not report.ok
    codes = [f.code for f in report.findings]
    assert "MISSING_ARTIFACT" in codes


# ── test 4b: missing thesis.pdf → MISSING_THESIS ─────────────────────────────

def test_missing_thesis_yields_error(tmp_path: Path) -> None:
    """A coherent strategy with no thesis.pdf must yield MISSING_THESIS (ERROR)."""
    r = _make_returns()
    sdir = _write_strategy(
        tmp_path, "momentum", "no_thesis", r, _summary_for(r), write_thesis=False
    )
    report = audit_strategy(sdir)
    assert not report.ok
    codes = [f.code for f in report.findings]
    assert "MISSING_THESIS" in codes


# ── test 5: audit_all detects DUAL_SLUG_SHADOW ───────────────────────────────

def test_audit_all_dual_slug_shadow(tmp_path: Path) -> None:
    """Two dirs sharing the same slug should produce DUAL_SLUG_SHADOW warnings."""
    r = _make_returns()
    # Production in two different categories with the same slug
    _write_strategy(tmp_path, "momentum", "shared_slug", r, _summary_for(r))
    _write_strategy(tmp_path, "carry", "shared_slug", r, _summary_for(r))

    reports = audit_all(root=tmp_path)
    shadow_reports = [
        rep for rep in reports
        if any(f.code == "DUAL_SLUG_SHADOW" for f in rep.findings)
    ]
    assert len(shadow_reports) >= 2, (
        "Both dirs sharing the slug should get a DUAL_SLUG_SHADOW finding"
    )
    assert all(
        any(f.severity == "WARN" for f in rep.findings) for rep in shadow_reports
    )


# ── test 6: main() exit codes ────────────────────────────────────────────────

def test_main_returns_nonzero_on_error(tmp_path: Path) -> None:
    """main() must return non-zero when at least one strategy has an ERROR."""
    r = _make_returns()
    bad_summary = _summary_for(r)
    bad_summary["metrics"]["sharpe"] = 9.9
    _write_strategy(tmp_path, "momentum", "err_strat", r, bad_summary)
    rc = main(["--root", str(tmp_path)])
    assert rc != 0


def test_main_returns_zero_on_all_clean(tmp_path: Path) -> None:
    """main() must return 0 when all strategies pass."""
    r = _make_returns()
    _write_strategy(tmp_path, "momentum", "clean_only", r, _summary_for(r))
    rc = main(["--root", str(tmp_path)])
    assert rc == 0


# ── additional coverage ───────────────────────────────────────────────────────

def test_format_report_counts(tmp_path: Path) -> None:
    """format_report() produces a non-empty string with correct counts."""
    r = _make_returns()
    _write_strategy(tmp_path, "momentum", "good", r, _summary_for(r))
    _write_strategy(tmp_path, "carry", "bad", r, None)  # MISSING_ARTIFACT
    reports = audit_all(root=tmp_path)
    text = format_report(reports)
    assert "ERROR" in text
    assert "OK" in text


def test_audit_report_worst_severity_ordering(tmp_path: Path) -> None:
    """worst_severity returns ERROR when both WARN and ERROR findings exist."""
    findings = [
        Finding(code="OK", severity="OK", message="fine"),
        Finding(code="DUAL_SLUG_SHADOW", severity="WARN", message="warn"),
        Finding(code="MISSING_ARTIFACT", severity="ERROR", message="err"),
    ]
    rep = AuditReport(slug="x", category="y", path="/x", findings=findings)
    assert rep.worst_severity == "ERROR"
    assert rep.ok is False


def test_main_json_flag(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--json flag produces valid JSON output."""
    r = _make_returns()
    _write_strategy(tmp_path, "momentum", "json_strat", r, _summary_for(r))
    rc = main(["--root", str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert data[0]["slug"] == "json_strat"
    assert rc == 0


def test_include_baseline(tmp_path: Path) -> None:
    """include_baseline=True should include Baseline/ strategies in the report."""
    r = _make_returns()
    # production
    _write_strategy(tmp_path, "momentum", "mom_strat", r, _summary_for(r))
    # baseline under strategies/Baseline/<cat>/<slug>/
    base_dir = tmp_path / "strategies" / "Baseline" / "momentum" / "mom_strat"
    arts = base_dir / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)
    cfg = {"name": "Mom Strat", "slug": "mom_strat", "category": "momentum",
            "status": "idle", "version": "v0"}
    (base_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    df = r.reset_index()
    df.columns = pd.Index(["date", "returns"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    (arts / "mom_strat_returns.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (arts / "mom_strat_summary.json").write_text(
        json.dumps(_summary_for(r), allow_nan=True), encoding="utf-8"
    )

    reports_excl = audit_all(root=tmp_path, include_baseline=False)
    reports_incl = audit_all(root=tmp_path, include_baseline=True)
    assert len(reports_incl) > len(reports_excl)


# ── test: missing positions CSV → MISSING_LEDGER WARN ────────────────────────

def test_missing_ledger_yields_warn(tmp_path: Path) -> None:
    """A strategy without a positions CSV must yield MISSING_LEDGER (WARN, not ERROR)."""
    r = _make_returns()
    sdir = _write_strategy(
        tmp_path, "momentum", "no_ledger", r, _summary_for(r),
        write_positions=False,  # deliberately omit positions CSV
    )
    report = audit_strategy(sdir)
    # WARN must not flip ok to False (ok = no ERROR)
    assert report.ok is True, "MISSING_LEDGER must be WARN, not ERROR"
    codes = [f.code for f in report.findings]
    assert "MISSING_LEDGER" in codes
    warns = [f for f in report.findings if f.code == "MISSING_LEDGER"]
    assert warns[0].severity == "WARN"


def test_ledger_exempt_suppresses_missing_ledger(tmp_path: Path) -> None:
    """ledger_exempt=True in config.json must suppress MISSING_LEDGER."""
    r = _make_returns()
    sdir = _write_strategy(
        tmp_path, "carry", "exempt_strat", r, _summary_for(r),
        write_positions=False,  # no positions CSV
        ledger_exempt=True,     # but strategy declares itself exempt
    )
    report = audit_strategy(sdir)
    assert report.ok is True
    codes = [f.code for f in report.findings]
    assert "MISSING_LEDGER" not in codes
    assert "DEGENERATE_LEDGER" not in codes
