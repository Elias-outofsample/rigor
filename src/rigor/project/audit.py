"""Coherence / artifact auditor for Rigor strategy folders.

Loads each strategy's saved artifacts (returns.csv + summary.json), recomputes
headline metrics from the raw returns, and reports any drift between the stored
summary and the live recomputation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rigor import metrics as _m

# ── repo-root locator ────────────────────────────────────────────────────────

def _repo_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: this file) until a .git directory is found."""
    p = (start or Path(__file__).resolve()).parent
    while True:
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[3]  # src/rigor/project -> repo root


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class Finding:
    """Single audit finding attached to one strategy."""

    code: str
    severity: str  # "ERROR" | "WARN" | "OK"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Collected findings for one strategy."""

    slug: str
    category: str
    path: str
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        """True when no ERROR-level findings are present."""
        return not any(f.severity == "ERROR" for f in self.findings)

    @property
    def worst_severity(self) -> str:
        """Highest severity across all findings (ERROR > WARN > OK)."""
        order = {"ERROR": 2, "WARN": 1, "OK": 0}
        if not self.findings:
            return "OK"
        return max(self.findings, key=lambda f: order.get(f.severity, 0)).severity


# ── internal helpers ─────────────────────────────────────────────────────────

def _load_returns(csv_path: Path) -> pd.Series | None:
    """Load returns.csv; return None if unreadable.

    Tolerates both committed layouts: a named ``date`` column (``date,returns``)
    and an unnamed leading index column (``,returns``) — the date is always the
    first column regardless of its header.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return None
    if "returns" not in df.columns or df.empty:
        return None
    date_col = "date" if "date" in df.columns else df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:  # noqa: BLE001
        return None
    df = df.sort_values(date_col).set_index(date_col)
    return df["returns"]


def _load_summary(json_path: Path) -> dict | None:
    """Load summary.json; accepts literal NaN tokens (python json module does)."""
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _claimed_sharpe(json_path: Path) -> float | None:
    """Return the finite Sharpe claimed by summary.json, or None.

    Reads ``metrics.sharpe`` (and tolerates a flat top-level ``sharpe``). Used by
    the degenerate-returns check to tell an honest flat curve (no Sharpe claim)
    apart from an internally-incoherent one (flat curve, real Sharpe claimed).
    """
    summary = _load_summary(json_path)
    if not isinstance(summary, dict):
        return None
    metrics = summary.get("metrics")
    raw = metrics.get("sharpe") if isinstance(metrics, dict) else summary.get("sharpe")
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


_RATIO_METRICS = {"sharpe", "sortino", "calmar"}
_FRAC_METRICS = {"cagr", "volatility", "max_drawdown"}

_ABS_TOL_RATIO = 0.05
_ABS_TOL_FRAC = 0.005
_REL_TOL_FRAC = 0.02


def _metric_mismatch(name: str, recomputed: float, stored: float) -> bool:
    """Return True when the two values disagree beyond tolerance."""
    if not math.isfinite(recomputed) or not math.isfinite(stored):
        return False  # skip NaN/inf comparisons
    diff = abs(recomputed - stored)
    if name in _RATIO_METRICS:
        return diff > _ABS_TOL_RATIO
    # fraction-like metrics: absolute OR relative
    rel = diff / max(abs(stored), 1e-12)
    return diff > _ABS_TOL_FRAC or rel > _REL_TOL_FRAC


# ── structural-coherence tuning constants ─────────────────────────────────────
# These calibrate the cross-strategy structural checks (adapted from an in-house
# audit_coherence.py and re-implemented in our Finding/severity model). They are
# deliberately conservative so the *current* catalog passes (0 ERROR): the only
# ERROR a structural check raises is a TRUE copy-paste (identical returns under
# two slugs whose configs are parameter-identical — a genuine mis-slug). All
# softer signals (near-duplicate-by-design variants, high correlation) are WARN.

# Minimum overlapping observations before two series are compared at all.
_STRUCT_MIN_OVERLAP = 50
# Numerical tolerance under which two return series are treated as "identical".
_STRUCT_IDENTICAL_ATOL = 1e-12
# Correlation above which two distinct strategies are flagged as near-duplicate.
_STRUCT_CORR_WARN = 0.999
# Sharpe magnitude in summary.json that makes a degenerate (flat) returns series
# *internally incoherent* — a flat curve cannot honestly claim a real Sharpe.
_STRUCT_DEGENERATE_SHARPE = 0.10

# config.json keys that identify the strategy's *declared* identity rather than
# its engine parameters; equal engine params with only these differing still
# count as parameter-identical for copy-paste detection.
_IDENTITY_KEYS = frozenset({"name", "slug"})


def _series_identical(a: pd.Series, b: pd.Series) -> tuple[bool, int]:
    """Return (identical, n_overlap) for two date-indexed returns series.

    "Identical" means: the two series share at least ``_STRUCT_MIN_OVERLAP``
    dates, cover the *same* set of dates, and every aligned value matches within
    ``_STRUCT_IDENTICAL_ATOL`` — i.e. a genuine byte/near-byte duplicate.
    """
    idx = a.index.intersection(b.index)
    n = len(idx)
    if n < _STRUCT_MIN_OVERLAP:
        return False, n
    # A true duplicate covers the same span — require near-total index overlap.
    if n < int(0.99 * max(len(a), len(b))):
        return False, n
    va = a.loc[idx].to_numpy(dtype=float)
    vb = b.loc[idx].to_numpy(dtype=float)
    identical = bool(np.allclose(va, vb, atol=_STRUCT_IDENTICAL_ATOL, rtol=0.0))
    return identical, n


def _series_correlation(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    """Return (pearson_corr, n_overlap) for two date-indexed returns series.

    ``None`` when the overlap is too small or either series is constant on the
    overlap (correlation undefined).
    """
    idx = a.index.intersection(b.index)
    n = len(idx)
    if n < _STRUCT_MIN_OVERLAP:
        return None, n
    va = a.loc[idx].to_numpy(dtype=float)
    vb = b.loc[idx].to_numpy(dtype=float)
    if float(np.std(va)) == 0.0 or float(np.std(vb)) == 0.0:
        return None, n
    corr = float(np.corrcoef(va, vb)[0, 1])
    if not math.isfinite(corr):
        return None, n
    return corr, n


def _config_engine_params(cfg_path: Path) -> dict[str, Any] | None:
    """Load a strategy's *engine* parameters from config.json.

    Returns the config with identity-only keys (name/slug) stripped, so two
    configs that differ ONLY in their declared name/slug compare equal. Returns
    None when the file is absent or unparseable.
    """
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cfg, dict):
        return None
    return {k: v for k, v in cfg.items() if k not in _IDENTITY_KEYS}


def _params_identical(cfg_a: Path, cfg_b: Path) -> bool:
    """True when two strategies declare *parameter-identical* configs.

    Identity keys (name/slug) are ignored. Two configs that differ in any engine
    parameter (e.g. ``extra.variant``) are NOT parameter-identical — they declare
    a deliberate semantic distinction, so identical returns between them is a
    near-duplicate-by-design (WARN), not a copy-paste mis-slug (ERROR).
    """
    pa = _config_engine_params(cfg_a)
    pb = _config_engine_params(cfg_b)
    if pa is None or pb is None:
        return False
    return pa == pb


# ── public API ───────────────────────────────────────────────────────────────

def audit_strategy(strategy_dir: Path | str) -> AuditReport:
    """Audit one strategy folder; return an :class:`AuditReport`."""
    sdir = Path(strategy_dir)
    slug = sdir.name
    category = sdir.parent.name

    # Try to read slug/category from config.json for better accuracy.
    cfg_file = sdir / "config.json"
    if cfg_file.is_file():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            slug = cfg.get("slug") or slug
            category = cfg.get("category") or category
        except Exception:  # noqa: BLE001
            pass

    from rigor.project import layout as _layout  # local import avoids circular

    paths = _layout.artifact_paths(sdir, slug)
    returns_path: Path = paths["returns"]
    summary_path: Path = paths["summary"]

    findings: list[Finding] = []

    # ── MISSING_ARTIFACT ────────────────────────────────────────────────────
    missing: list[str] = []
    if not returns_path.is_file():
        missing.append(str(returns_path.name))
    if not summary_path.is_file():
        missing.append(str(summary_path.name))

    if missing:
        findings.append(Finding(
            code="MISSING_ARTIFACT",
            severity="ERROR",
            message=f"Missing artifact file(s): {', '.join(missing)}",
            detail={"missing": missing},
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── load returns ────────────────────────────────────────────────────────
    returns = _load_returns(returns_path)
    if returns is None:
        findings.append(Finding(
            code="MISSING_ARTIFACT",
            severity="ERROR",
            message="returns.csv could not be parsed",
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── DEGENERATE_ARTIFACT (returns) ────────────────────────────────────────
    r_vals = returns.values
    if (
        len(r_vals) == 0
        or all(math.isnan(v) for v in r_vals)
        or float(pd.Series(r_vals).std(ddof=1)) == 0.0
    ):
        # Base message (returns are degenerate). If summary.json nonetheless
        # claims a non-trivial Sharpe, the artifact is *internally incoherent* —
        # a flat/constant curve cannot honestly produce a real Sharpe. Surface
        # that explicitly so the author sees *why* it is wrong, not just that
        # the curve is flat. (Structural check #2.)
        msg = "Returns series is empty, all-NaN, all-zero, or constant (std == 0)"
        detail: dict[str, Any] = {}
        claimed = _claimed_sharpe(summary_path)
        if claimed is not None and abs(claimed) >= _STRUCT_DEGENERATE_SHARPE:
            msg += (
                f"; yet summary.json claims sharpe={claimed:.3f} "
                "(internally incoherent — flat curve cannot earn a real Sharpe)"
            )
            detail = {"claimed_sharpe": claimed}
        findings.append(Finding(
            code="DEGENERATE_ARTIFACT",
            severity="ERROR",
            message=msg,
            detail=detail,
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── load summary ────────────────────────────────────────────────────────
    summary = _load_summary(summary_path)
    if summary is None:
        findings.append(Finding(
            code="MISSING_ARTIFACT",
            severity="ERROR",
            message="summary.json could not be parsed",
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    stored_metrics: dict | None = summary.get("metrics")
    if not isinstance(stored_metrics, dict):
        findings.append(Finding(
            code="DEGENERATE_ARTIFACT",
            severity="ERROR",
            message="summary.json is missing the 'metrics' block",
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── recompute metrics ────────────────────────────────────────────────────
    ppy = (
        _m.periods_per_year_of(returns.index)
        if isinstance(returns.index, pd.DatetimeIndex)
        else _m.TRADING_DAYS
    )
    recomputed = _m.compute_core_metrics(returns.values, ppy)

    # ── PERF_NE_CURVE ────────────────────────────────────────────────────────
    mismatches: list[tuple[str, float, float]] = []
    for mname in ("sharpe", "cagr", "max_drawdown", "volatility"):
        stored_val = stored_metrics.get(mname)
        if stored_val is None or (isinstance(stored_val, float) and math.isnan(stored_val)):
            continue
        try:
            stored_float = float(stored_val)
        except (TypeError, ValueError):
            continue
        recomp_val = recomputed.get(mname)
        if recomp_val is None:
            continue
        if _metric_mismatch(mname, float(recomp_val), stored_float):
            mismatches.append((mname, float(recomp_val), stored_float))

    if mismatches:
        findings.append(Finding(
            code="PERF_NE_CURVE",
            severity="ERROR",
            message=(
                f"Recomputed metric(s) disagree with summary.json: "
                f"{[m[0] for m in mismatches]}"
            ),
            detail={"mismatches": [{"metric": m, "recomputed": r, "stored": s}
                                   for m, r, s in mismatches]},
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── MISSING_THESIS ───────────────────────────────────────────────────────
    # Every strategy must ship its thesis artifact (generated by `rigor run` unless
    # --no-thesis). Enforce it so the catalog never ships a strategy without one.
    thesis_path = summary_path.parent / f"{slug}_thesis.pdf"
    if not thesis_path.is_file():
        findings.append(Finding(
            code="MISSING_THESIS",
            severity="ERROR",
            message=(f"Missing thesis artifact: {thesis_path.name} "
                     "(run `rigor run` without --no-thesis)"),
            detail={"missing": [thesis_path.name]},
        ))
        return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)

    # ── MISSING_LEDGER / DEGENERATE_LEDGER ───────────────────────────────────
    # Per-trade position ledger (MAE/MFE) is a required output of every `rigor run`.
    # Strategies where per-trade excursion is semantically N/A (e.g. delta-neutral
    # carry, always-invested, or returns-only wrappers) may set
    # ``ledger_exempt: true`` in config.json to silence these findings.
    #
    # NOTE: Severity is WARN now (migration window — existing strategies pre-date
    # the ledger requirement).  Flip to ERROR after all committed strategies have
    # been re-run and their positions CSV is present.
    _ledger_exempt: bool = False
    if cfg_file.is_file():
        try:
            _cfg_raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            _ledger_exempt = bool(_cfg_raw.get("ledger_exempt", False))
        except Exception:  # noqa: BLE001
            pass

    if not _ledger_exempt:
        positions_path: Path = paths["positions"]
        if not positions_path.is_file():
            findings.append(Finding(
                code="MISSING_LEDGER",
                severity="WARN",
                message=(
                    f"Missing position ledger: {positions_path.name} "
                    "(run `rigor run` to regenerate; or set ledger_exempt=true "
                    "in config.json for semantically-N/A strategies)"
                ),
                detail={"missing": [positions_path.name]},
            ))
        else:
            # Degenerate ledger: file present but excursion columns entirely NaN.
            try:
                _led = pd.read_csv(positions_path)
                _mae_all_nan = (
                    "max_adverse_excursion" not in _led.columns
                    or _led["max_adverse_excursion"].isna().all()
                )
                _mfe_all_nan = (
                    "max_favorable_excursion" not in _led.columns
                    or _led["max_favorable_excursion"].isna().all()
                )
                if _mae_all_nan and _mfe_all_nan:
                    findings.append(Finding(
                        code="DEGENERATE_LEDGER",
                        severity="WARN",
                        message=(
                            f"Position ledger {positions_path.name} exists but "
                            "max_adverse_excursion and max_favorable_excursion are "
                            "entirely NaN (re-run with a BacktestResult that carries "
                            "weight/price data)"
                        ),
                        detail={"path": str(positions_path)},
                    ))
            except Exception:  # noqa: BLE001
                pass  # unreadable CSV — MISSING_LEDGER already handles total absence

    # ── all good ─────────────────────────────────────────────────────────────
    findings.append(Finding(
        code="OK",
        severity="OK",
        message="Returns and summary metrics reconcile within tolerance; thesis present",
    ))
    return AuditReport(slug=slug, category=category, path=str(sdir), findings=findings)


def audit_all(
    root: Path | None = None,
    *,
    include_baseline: bool = False,
) -> list[AuditReport]:
    """Audit every production strategy under *root*/Strategy/.

    Parameters
    ----------
    root:
        Repo root. Auto-detected via .git walk when None.
    include_baseline:
        When True, also audit strategies/Baseline/ strategies.
    """
    if root is None:
        root = _repo_root()

    strategies_root = root / "strategies"
    reports: list[AuditReport] = []

    if not strategies_root.is_dir():
        return reports

    # Collect all strategy dirs: strategies/<category>/<slug>/config.json
    strategy_dirs: list[Path] = []
    for cfg in sorted(strategies_root.glob("*/*/config.json")):
        # cfg is  strategies/<cat>/<slug>/config.json
        cat_dir = cfg.parent.parent  # strategies/<cat>
        if cat_dir.name == "Baseline":
            continue  # skip top-level Baseline pass-through
        if cfg.parent.parent.parent.name == "Baseline" and not include_baseline:
            continue
        strategy_dirs.append(cfg.parent)

    if include_baseline:
        for cfg in sorted(strategies_root.glob("Baseline/*/*/config.json")):
            strategy_dirs.append(cfg.parent)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for d in strategy_dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    # Detect slug collisions across production dirs (DUAL_SLUG_SHADOW)
    slug_map: dict[str, list[Path]] = {}
    for d in unique_dirs:
        slug_map.setdefault(d.name, []).append(d)

    shadow_slugs = {slug for slug, dirs in slug_map.items() if len(dirs) > 1}

    for sdir in unique_dirs:
        try:
            report = audit_strategy(sdir)
        except Exception as exc:  # noqa: BLE001
            report = AuditReport(
                slug=sdir.name,
                category=sdir.parent.name,
                path=str(sdir),
                findings=[Finding(
                    code="MISSING_ARTIFACT",
                    severity="ERROR",
                    message=f"Unexpected error during audit: {exc}",
                )],
            )
        # Attach DUAL_SLUG_SHADOW if this slug appears in multiple dirs
        if report.slug in shadow_slugs:
            siblings = [str(p) for p in slug_map[report.slug] if p != sdir]
            report.findings.append(Finding(
                code="DUAL_SLUG_SHADOW",
                severity="WARN",
                message=(
                    f"Slug '{report.slug}' appears in multiple strategy directories"
                ),
                detail={"also_at": siblings},
            ))
        reports.append(report)

    # ── cross-strategy STRUCTURAL coherence pass ─────────────────────────────
    # Adapted from an in-house coherence audit, re-implemented in our Finding model.
    # Runs once over the whole catalog (pairwise) and attaches structural
    # findings to the relevant reports. Cheap: O(n^2) over already-loaded series.
    _attach_structural_findings(reports, {r.path: sdir for r, sdir in
                                          zip(reports, unique_dirs, strict=True)})

    return reports


def _attach_structural_findings(
    reports: list[AuditReport],
    dir_by_path: dict[str, Path],
) -> None:
    """Attach cross-strategy structural findings to *reports* in place.

    Three structural checks, calibrated so the current catalog stays at 0 ERROR:

    * ``DUAL_SLUG_DUPLICATE`` (ERROR) — two slugs with *numerically identical*
      returns whose configs are also *parameter-identical*. That is an
      unambiguous copy-paste / mis-slug: same strategy committed twice. When the
      configs declare different engine parameters (e.g. ``variant``) the identity
      is intentional → downgraded to ``DUAL_SLUG_NEAR_DUPLICATE`` (WARN).
    * ``RETURNS_CORRELATION`` (WARN) — two *distinct* strategies whose returns
      correlate above ``_STRUCT_CORR_WARN``. ADVISORY ONLY: our momentum
      tilt/base variants are legitimately ~1.0 correlated by design, so this must
      never block CI.
    """
    # Load each strategy's returns once (skip reports that already ERROR'd —
    # their returns are missing/degenerate and not comparable).
    series_by_path: dict[str, pd.Series] = {}
    for rep in reports:
        if not rep.ok:
            continue
        sdir = dir_by_path.get(rep.path)
        if sdir is None:
            continue
        rpath = _layout_returns_path(sdir, rep.slug)
        if rpath is None or not rpath.is_file():
            continue
        s = _load_returns(rpath)
        if s is None:
            continue
        # Drop NaNs but keep the datetime index so series align on shared dates.
        s = s.dropna()
        if len(s) >= _STRUCT_MIN_OVERLAP:
            series_by_path[rep.path] = s

    report_by_path = {r.path: r for r in reports}
    paths = sorted(series_by_path.keys())

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pa, pb = paths[i], paths[j]
            ra, rb = report_by_path[pa], report_by_path[pb]
            sa, sb = series_by_path[pa], series_by_path[pb]

            identical, _ = _series_identical(sa, sb)
            if identical:
                cfg_a = dir_by_path[pa] / "config.json"
                cfg_b = dir_by_path[pb] / "config.json"
                if _params_identical(cfg_a, cfg_b):
                    # TRUE copy-paste: same returns AND same engine params.
                    for rep, other in ((ra, rb), (rb, ra)):
                        rep.findings.append(Finding(
                            code="DUAL_SLUG_DUPLICATE",
                            severity="ERROR",
                            message=(
                                f"Returns are byte-identical to a different slug "
                                f"'{other.category}/{other.slug}' with a "
                                "parameter-identical config (copy-paste / mis-slug)"
                            ),
                            detail={"duplicate_of": f"{other.category}/{other.slug}",
                                    "duplicate_path": other.path},
                        ))
                else:
                    # Identical returns but configs declare distinct params:
                    # intentional redundancy (e.g. filtre vs tendance). Advisory.
                    for rep, other in ((ra, rb), (rb, ra)):
                        rep.findings.append(Finding(
                            code="DUAL_SLUG_NEAR_DUPLICATE",
                            severity="WARN",
                            message=(
                                f"Returns identical to '{other.category}/{other.slug}' "
                                "despite a declared parameter difference "
                                "(redundant variant — review whether both are needed)"
                            ),
                            detail={"identical_to": f"{other.category}/{other.slug}"},
                        ))
                continue  # identical pairs need no separate correlation note

            corr, n = _series_correlation(sa, sb)
            if corr is not None and corr > _STRUCT_CORR_WARN:
                # ADVISORY ONLY — legitimately-correlated variants are expected.
                for rep, other in ((ra, rb), (rb, ra)):
                    rep.findings.append(Finding(
                        code="RETURNS_CORRELATION",
                        severity="WARN",
                        message=(
                            f"Returns correlate {corr:.4f} with "
                            f"'{other.category}/{other.slug}' over {n} shared days "
                            "(near-duplicate — advisory, not a defect)"
                        ),
                        detail={"correlated_with": f"{other.category}/{other.slug}",
                                "correlation": corr, "n_overlap": n},
                    ))


def _layout_returns_path(sdir: Path, slug: str) -> Path | None:
    """Resolve the committed returns.csv path for *sdir* via the layout module."""
    try:
        from rigor.project import layout as _layout  # local import avoids circular
        return _layout.artifact_paths(sdir, slug)["returns"]
    except Exception:  # noqa: BLE001
        return None


def format_report(reports: list[AuditReport]) -> str:
    """Return a concise human-readable summary of audit results."""
    n_error = sum(1 for r in reports if not r.ok)
    n_warn = sum(
        1 for r in reports
        if r.ok and any(f.severity == "WARN" for f in r.findings)
    )
    n_ok = len(reports) - n_error - n_warn

    lines: list[str] = [
        f"Audit: {len(reports)} strategies - "
        f"{n_error} ERROR, {n_warn} WARN, {n_ok} OK"
    ]

    for r in reports:
        errors = [f for f in r.findings if f.severity == "ERROR"]
        warns = [f for f in r.findings if f.severity == "WARN"]
        if errors:
            codes = ", ".join(f.code for f in errors)
            lines.append(f"  ERROR  {r.category}/{r.slug}: {codes}")
        elif warns:
            codes = ", ".join(f.code for f in warns)
            lines.append(f"  WARN   {r.category}/{r.slug}: {codes}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: audit strategy artifacts and report discrepancies."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Audit Rigor strategy artifacts.")
    parser.add_argument("--root", type=Path, default=None, help="Repo root path.")
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also audit strategies/Baseline/ strategies.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    reports = audit_all(root=args.root, include_baseline=args.include_baseline)

    if args.emit_json:
        output = [
            {
                "slug": r.slug,
                "category": r.category,
                "path": r.path,
                "ok": r.ok,
                "worst_severity": r.worst_severity,
                "findings": [
                    {"code": f.code, "severity": f.severity, "message": f.message}
                    for f in r.findings
                ],
            }
            for r in reports
        ]
        print(json.dumps(output, indent=2))
    else:
        print(format_report(reports))

    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
