"""Tests for the extra STRUCTURAL coherence checks in rigor.project.audit.

These complement test_audit.py (which covers the recompute checks). Here we
exercise the cross-strategy structural pass adapted from an in-house coherence audit
and re-implemented in our Finding/severity model:

  * DUAL_SLUG_DUPLICATE        (ERROR) — identical returns + identical config
  * DUAL_SLUG_NEAR_DUPLICATE   (WARN)  — identical returns, configs differ
  * RETURNS_CORRELATION        (WARN)  — distinct strategies, corr > ~0.999
  * DEGENERATE_ARTIFACT enrichment      — flat returns + a claimed Sharpe

All synthetic tests use tmp_path and do NOT depend on committed strategies/
contents. The single real-catalog test is a smoke check that the current
catalog audit still exits 0 (CRITICAL: a new ERROR check must never red CI on a
legitimate existing strategy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rigor import metrics as _m
from rigor.project import audit as _audit
from rigor.project.audit import audit_all, main

# ── helpers (mirror test_audit.py's folder writer) ───────────────────────────


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    """Deterministic daily returns series with a real (non-zero) std."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0005, 0.01, size=n)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(vals, index=dates, name="returns")


def _summary_for(returns: pd.Series) -> dict:
    """Build a summary.json whose metrics match *returns* exactly."""
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


def _write_strategy(
    root: Path,
    category: str,
    slug: str,
    returns: pd.Series | None,
    summary: dict | None,
    *,
    extra: dict | None = None,
) -> Path:
    """Write a minimal, *coherent* strategy folder under root/Strategy/.

    Writes thesis + positions stubs so a clean strategy reaches OK (the
    structural pass only compares OK strategies). ``extra`` is merged into
    config.json's top-level ``extra`` block so param-difference can be tested.
    """
    sdir = root / "strategies" / category / slug
    arts = sdir / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)

    if returns is not None and summary is not None:
        (arts / f"{slug}_thesis.pdf").write_bytes(b"%PDF-1.4 stub\n")
        positions_stub = (
            "entry_date,exit_date,symbol,entry_price,exit_price,"
            "max_adverse_excursion,max_favorable_excursion\n"
            "2020-01-02,2020-01-10,SPY,320.0,325.0,-0.01,0.02\n"
        )
        (arts / f"{slug}_positions.csv").write_text(positions_stub, encoding="utf-8")

    cfg: dict = {
        "name": slug.replace("_", " ").title(),
        "slug": slug,
        "category": category,
        "status": "idle",
        "version": "v1",
    }
    if extra is not None:
        cfg["extra"] = extra
    (sdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    if returns is not None:
        df = returns.reset_index()
        df.columns = pd.Index(["date", "returns"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        (arts / f"{slug}_returns.csv").write_text(
            df.to_csv(index=False), encoding="utf-8"
        )

    if summary is not None:
        (arts / f"{slug}_summary.json").write_text(
            json.dumps(summary, allow_nan=True), encoding="utf-8"
        )

    return sdir


def _findings_for(reports, category: str, slug: str) -> list:
    """Return the findings list for one strategy in a report set."""
    for r in reports:
        if r.category == category and r.slug == slug:
            return r.findings
    raise AssertionError(f"no report for {category}/{slug}")


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


def _severity_of(findings, code: str) -> str:
    return next(f.severity for f in findings if f.code == code)


# ── test 1: TRUE copy-paste duplicate → DUAL_SLUG_DUPLICATE (ERROR) ──────────


def test_dual_slug_duplicate_errors(tmp_path: Path) -> None:
    """Identical returns under two slugs with param-identical configs = ERROR."""
    r = _make_returns()
    params = {"normalizer": "atr", "top_n": 10}
    # Same engine params, same returns, different slugs → genuine mis-slug.
    _write_strategy(tmp_path, "momentum", "alpha_strat", r, _summary_for(r), extra=params)
    _write_strategy(tmp_path, "momentum", "alpha_copy", r, _summary_for(r), extra=params)

    reports = audit_all(root=tmp_path)
    fa = _findings_for(reports, "momentum", "alpha_strat")
    fb = _findings_for(reports, "momentum", "alpha_copy")

    assert "DUAL_SLUG_DUPLICATE" in _codes(fa)
    assert "DUAL_SLUG_DUPLICATE" in _codes(fb)
    assert _severity_of(fa, "DUAL_SLUG_DUPLICATE") == "ERROR"
    assert _severity_of(fb, "DUAL_SLUG_DUPLICATE") == "ERROR"
    # ERROR must flip ok to False (fails the build).
    both = [r for r in reports if r.slug in ("alpha_strat", "alpha_copy")]
    assert all(not r.ok for r in both)
    # main() must return non-zero.
    assert main(["--root", str(tmp_path)]) != 0


# ── test 2: identical returns but DIFFERENT params → near-dup (WARN only) ─────


def test_dual_slug_near_duplicate_is_warn(tmp_path: Path) -> None:
    """Identical returns under DIFFERENT declared params = WARN, never ERROR.

    Mirrors the real filtre/tendance pair: same engine, intentionally distinct
    variant, byte-identical returns. Must NOT block CI.
    """
    r = _make_returns()
    _write_strategy(tmp_path, "momentum", "variant_filtre", r, _summary_for(r),
                    extra={"variant": "filtre", "top_n": 10})
    _write_strategy(tmp_path, "momentum", "variant_tendance", r, _summary_for(r),
                    extra={"variant": "tendance", "top_n": 10})

    reports = audit_all(root=tmp_path)
    fa = _findings_for(reports, "momentum", "variant_filtre")
    fb = _findings_for(reports, "momentum", "variant_tendance")

    assert "DUAL_SLUG_NEAR_DUPLICATE" in _codes(fa)
    assert "DUAL_SLUG_NEAR_DUPLICATE" in _codes(fb)
    assert _severity_of(fa, "DUAL_SLUG_NEAR_DUPLICATE") == "WARN"
    # No ERROR-level duplicate finding.
    assert "DUAL_SLUG_DUPLICATE" not in _codes(fa)
    assert "DUAL_SLUG_DUPLICATE" not in _codes(fb)
    both = [r for r in reports if r.slug in ("variant_filtre", "variant_tendance")]
    assert all(r.ok for r in both)
    assert main(["--root", str(tmp_path)]) == 0


# ── test 3: legitimately-correlated distinct strategies → WARN advisory ──────


def test_high_correlation_is_warn_only(tmp_path: Path) -> None:
    """Two distinct strategies correlating > 0.999 must be at most WARN."""
    r = _make_returns()
    # A tiny independent tilt keeps them NON-identical but ~0.9999 correlated.
    rng = np.random.default_rng(7)
    tilt = pd.Series(rng.normal(0.0, 1e-5, size=len(r)), index=r.index, name="returns")
    r2 = (r + tilt).rename("returns")

    _write_strategy(tmp_path, "momentum", "tilt_base", r, _summary_for(r),
                    extra={"variant": "base"})
    _write_strategy(tmp_path, "momentum", "tilt_highvol", r2, _summary_for(r2),
                    extra={"variant": "highvol"})

    reports = audit_all(root=tmp_path)
    fa = _findings_for(reports, "momentum", "tilt_base")
    fb = _findings_for(reports, "momentum", "tilt_highvol")

    assert "RETURNS_CORRELATION" in _codes(fa)
    assert "RETURNS_CORRELATION" in _codes(fb)
    assert _severity_of(fa, "RETURNS_CORRELATION") == "WARN"
    # Correlation must NEVER be an ERROR — verify no ERROR finding present.
    assert not any(f.severity == "ERROR" for f in fa)
    assert not any(f.severity == "ERROR" for f in fb)
    both = [r for r in reports if r.slug in ("tilt_base", "tilt_highvol")]
    assert all(r.ok for r in both)
    assert main(["--root", str(tmp_path)]) == 0


# ── test 4: degenerate returns + a claimed Sharpe → enriched ERROR ───────────


def test_degenerate_returns_with_claimed_sharpe_errors(tmp_path: Path) -> None:
    """A flat returns series whose summary claims a real Sharpe is incoherent."""
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    flat = pd.Series(np.zeros(300), index=dates, name="returns")
    summary = {
        "as_of": "2026-01-01", "start": "2020-01-02", "end": "2021-12-31",
        "name": "Flat", "metrics": {"sharpe": 1.85, "cagr": 0.0,
                                     "max_drawdown": 0.0, "volatility": 0.0,
                                     "n_obs": 300},
        "verdict": {"score": 0, "gates": []},
    }
    sdir = _write_strategy(tmp_path, "volatility", "flat_but_claims", flat, summary)

    report = _audit.audit_strategy(sdir)
    assert not report.ok
    deg = next(f for f in report.findings if f.code == "DEGENERATE_ARTIFACT")
    assert deg.severity == "ERROR"
    # The incoherence is spelled out for the author.
    assert "internally incoherent" in deg.message
    assert deg.detail.get("claimed_sharpe") == 1.85


def test_degenerate_returns_without_claim_still_errors(tmp_path: Path) -> None:
    """A flat curve with an honest (0) Sharpe is still a degenerate ERROR."""
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    flat = pd.Series(np.zeros(300), index=dates, name="returns")
    summary = {
        "as_of": "2026-01-01", "start": "2020-01-02", "end": "2021-12-31",
        "name": "Flat", "metrics": {"sharpe": 0.0, "n_obs": 300},
        "verdict": {"score": 0, "gates": []},
    }
    sdir = _write_strategy(tmp_path, "volatility", "flat_honest", flat, summary)

    report = _audit.audit_strategy(sdir)
    assert not report.ok
    deg = next(f for f in report.findings if f.code == "DEGENERATE_ARTIFACT")
    assert deg.severity == "ERROR"
    # No false incoherence claim when no real Sharpe is asserted.
    assert "internally incoherent" not in deg.message


# ── test 5: distinct, uncorrelated strategies → no structural findings ───────


def test_distinct_strategies_no_structural_findings(tmp_path: Path) -> None:
    """Two genuinely-different strategies get no dual-slug / correlation flags."""
    r1 = _make_returns(seed=1)
    r2 = _make_returns(seed=999)
    _write_strategy(tmp_path, "momentum", "strat_one", r1, _summary_for(r1))
    _write_strategy(tmp_path, "carry", "strat_two", r2, _summary_for(r2))

    reports = audit_all(root=tmp_path)
    for slug, cat in (("strat_one", "momentum"), ("strat_two", "carry")):
        codes = _codes(_findings_for(reports, cat, slug))
        assert "DUAL_SLUG_DUPLICATE" not in codes
        assert "DUAL_SLUG_NEAR_DUPLICATE" not in codes
        assert "RETURNS_CORRELATION" not in codes
    assert main(["--root", str(tmp_path)]) == 0


# ── test 6: SMOKE — the real catalog audit still exits 0 (no new ERRORs) ──────


def test_real_catalog_audit_exits_zero() -> None:
    """CRITICAL: the committed catalog must still pass (0 ERROR) after new checks.

    A structural ERROR firing on a legitimate existing strategy would red CI.
    Skips gracefully if run outside the repo (no strategies/ tree present).
    """
    reports = audit_all()
    if not reports:  # running detached from the repo — nothing to assert
        return
    errored = [
        (r.category, r.slug, [f.code for f in r.findings if f.severity == "ERROR"])
        for r in reports
        if not r.ok
    ]
    assert errored == [], f"unexpected ERROR-level findings in catalog: {errored}"
    assert main([]) == 0
