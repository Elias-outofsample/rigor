"""Per-strategy derived artefact generator — walk-forward, per-year, and risk/tail battery.

All computation is purely offline (reads only committed ``returns.csv``).  No API
key, no data fetch, no import of strategy.py.

Public API
----------
``derive_strategy(strategy_dir)``
    Write ``artifacts/<slug>_{wf,risk,peryear}.json`` for one strategy.
``derive_all(root=None)``
    Run over every production strategy (skips ``strategies/Baseline/``).
``main(argv=None)``
    CLI entry-point (argparse, ``--root``, ``--slug``, ``--json``).

Walk-forward implementation
---------------------------
``rigor.validation.robustness.walk_forward`` accepts a bare ``pd.Series`` of
returns — no callable needed.  We call it directly.

Per-year implementation
-----------------------
``rigor.validation.robustness.per_year`` also accepts a bare ``pd.Series``.
We call it directly but augment its output with ``max_drawdown`` and ``n_obs``
per year so downstream consumers have the full picture.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rigor import metrics as _m
from rigor.analysis.risk import risk_summary as _risk_summary
from rigor.analysis.tail_risk import fit_gpd_tail, kupiec_pof_test
from rigor.data.config import find_repo_root
from rigor.project import layout
from rigor.validation.robustness import per_year as _per_year
from rigor.validation.robustness import walk_forward as _walk_forward

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _iso(value) -> str:
    """ISO string for a datetime-like index value; fall back to str() otherwise."""
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


def _load_returns(strategy_dir: Path, slug: str) -> pd.Series | None:
    """Load ``artifacts/<slug>_returns.csv`` from *strategy_dir*.

    Handles two formats:
    * Named date column:  ``date,returns``
    * Unnamed index:      ``,returns``  (first column has no header, second is "returns")

    Returns ``None`` if the file is absent, empty, or unparseable.
    """
    paths = layout.artifact_paths(strategy_dir, slug)
    csv_path = paths["returns"]
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        if df.empty:
            return None
        # If the index is already named "date" it's fine; otherwise it might be
        # the unnamed index column or a different name — just use it directly.
        df.index = pd.to_datetime(df.index, utc=False)
        # Find the returns column: prefer "returns", otherwise first numeric column.
        if "returns" in df.columns:
            sr = df["returns"].sort_index().dropna()
        else:
            sr = df.iloc[:, 0].sort_index().dropna()
        sr = sr.astype(float)
        sr.name = slug
        if len(sr) < 2:
            return None
        return sr
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse returns for %s: %s", slug, exc)
        return None


def _to_json_safe(obj: Any) -> Any:  # noqa: ANN401
    """Recursively cast numpy scalars to Python floats/ints; replace NaN/inf with None."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _to_json_safe(obj.tolist())
    return obj


# --------------------------------------------------------------------------- #
# Derive functions                                                             #
# --------------------------------------------------------------------------- #

def derive_walk_forward(returns: pd.Series, n_folds: int = 5) -> dict:
    """Sequential fold consistency over a returns Series.

    Delegates to ``rigor.validation.robustness.walk_forward`` (accepts a plain
    ``pd.Series`` — no callable required).  We override ``n_folds`` from the
    function's default of 8 to 5 so the API surface is explicit.

    Returns a dict with keys: ``n_folds``, ``fold_sharpes``, ``full_sharpe``,
    ``efficiency``, ``consistency``, ``cv``, ``verdict``, ``fold_boundaries``.
    """
    r = returns.dropna()
    result = _walk_forward(r, n_folds=n_folds)

    # Annotate fold date boundaries for human readability. Slice the Series' own
    # index in contiguous folds (matching np.array_split sizing: the first n%k
    # folds get one extra) — never call np.array_split on a Series, whose return
    # type (Series vs bare ndarray) is numpy-version-dependent.
    n, k = len(r), int(result["n_folds"])
    boundaries = []
    if n >= 2 and k > 0:
        idx = r.index
        base_sz, rem = divmod(n, k)
        start = 0
        for i in range(k):
            sz = base_sz + (1 if i < rem else 0)
            if sz <= 0:
                continue
            seg = idx[start:start + sz]
            boundaries.append({
                "start": _iso(seg[0]),
                "end": _iso(seg[-1]),
                "n_obs": int(sz),
            })
            start += sz
    result["fold_boundaries"] = boundaries
    return result


def derive_per_year(returns: pd.Series) -> dict:
    """Per-calendar-year statistics.

    Calls ``rigor.validation.robustness.per_year`` for Sharpe + return, then
    augments each year bucket with ``max_drawdown`` and ``n_obs`` using the
    same ``rigor.metrics`` helpers.

    Returns a dict with keys: ``years`` (dict keyed by int year), ``n_years``,
    ``positive_fraction``, ``worst_sharpe``.  Each year value adds
    ``max_drawdown`` and ``n_obs`` to the base ``{sharpe, return}`` pair.
    """
    base = _per_year(returns)
    # Augment each year with max_drawdown and n_obs.
    r = returns.dropna()
    if isinstance(r.index, pd.DatetimeIndex) and base["years"]:
        for year, grp in r.groupby(r.index.year):
            key = int(year)
            if key in base["years"]:
                arr = grp.to_numpy()
                base["years"][key]["max_drawdown"] = round(float(_m.compute_max_drawdown(arr)), 4)
                base["years"][key]["n_obs"] = int(len(arr))
    return base


def derive_risk(returns: pd.Series) -> dict:
    """Full risk / tail battery from a returns Series.

    Merges:
    * ``rigor.analysis.risk.risk_summary`` — VaR/CVaR/omega/drawdown ratios
    * ``rigor.analysis.tail_risk.fit_gpd_tail`` — generalised Pareto EVT
    * ``rigor.analysis.tail_risk.kupiec_pof_test`` — VaR backtest

    All numpy scalars are cast to plain Python floats; NaN/inf become None so
    the dict is safe for ``json.dump``.
    """
    r = returns.dropna()
    summary = _risk_summary(r)
    gpd = fit_gpd_tail(r)
    kupiec = kupiec_pof_test(r.to_numpy())
    combined: dict = {
        **summary,
        "gpd_tail": gpd,
        "kupiec_pof": kupiec,
    }
    return _to_json_safe(combined)


# --------------------------------------------------------------------------- #
# Strategy-level driver                                                        #
# --------------------------------------------------------------------------- #

def derive_strategy(strategy_dir: Path | str, *, n_folds: int = 5) -> dict:
    """Derive and write artefacts for one strategy directory.

    Reads ``config.json`` to find the slug, loads ``artifacts/<slug>_returns.csv``,
    computes the three batteries, and writes:

    * ``extra/<slug>_wf.json``       — walk-forward folds
    * ``extra/<slug>_risk.json``     — risk / tail battery
    * ``extra/<slug>_peryear.json``  — per-calendar-year stats

    Returns a summary dict: ``{slug, written, skipped, reason}``.
    """
    strategy_dir = Path(strategy_dir)
    config_path = strategy_dir / "config.json"

    # Resolve slug.
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = {}
    else:
        cfg = {}
    slug = cfg.get("slug") or strategy_dir.name

    returns = _load_returns(strategy_dir, slug)
    if returns is None:
        log.info("Skipping %s — no usable returns.csv", slug)
        return {"slug": slug, "written": [], "skipped": True, "reason": "no_returns"}

    if len(returns) < 10:
        return {"slug": slug, "written": [], "skipped": True, "reason": "degenerate_returns"}

    extra = layout.extra_dir(strategy_dir)        # auxiliary diagnostics -> extra/
    extra.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    def _write(name: str, data: dict) -> None:
        path = extra / f"{slug}_{name}.json"
        path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")
        written.append(str(path))

    _write("wf", derive_walk_forward(returns, n_folds=n_folds))
    _write("risk", derive_risk(returns))
    _write("peryear", derive_per_year(returns))

    log.info("Derived %d artefacts for %s", len(written), slug)
    return {"slug": slug, "written": written, "skipped": False, "reason": ""}


# --------------------------------------------------------------------------- #
# Bulk driver                                                                  #
# --------------------------------------------------------------------------- #

def derive_all(root: Path | None = None) -> list[dict]:
    """Run ``derive_strategy`` over every production strategy.

    Enumerates ``strategies/*/*/config.json`` beneath *root* (or the repo root
    found via ``find_repo_root()``).  Skips ``strategies/Baseline/``.
    """
    if root is None:
        root = find_repo_root()
    root = Path(root)
    strat_root = root / layout.STRATEGIES_ROOT
    baseline_marker = layout.BASELINE_ROOT

    results: list[dict] = []
    for cfg_path in sorted(strat_root.glob("*/*/config.json")):
        rel = cfg_path.relative_to(strat_root)
        # Skip Baseline subtree.
        if rel.parts[0] == baseline_marker:
            continue
        results.append(derive_strategy(cfg_path.parent))
    return results


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    """Argparse entry-point.  Exit 0 on success."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="python -m rigor.project.derive",
        description="Generate derived artefacts (walk-forward, risk, per-year) from returns.csv.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root (default: auto-detect via .git).",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Run for a single strategy slug (must live under strategies/*/*/<slug>).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary to stdout instead of human-readable text.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    root = args.root or find_repo_root()
    results: list[dict]

    if args.slug:
        strat_root = root / layout.STRATEGIES_ROOT
        matches = [
            p.parent
            for p in strat_root.glob(f"*/{ args.slug}/config.json")
            if p.parent.parent.name != layout.BASELINE_ROOT
        ]
        if not matches:
            print(f"ERROR: strategy '{args.slug}' not found under {strat_root}")
            return 1
        results = [derive_strategy(matches[0])]
    else:
        results = derive_all(root)

    n_ok = sum(1 for r in results if not r["skipped"])
    n_skip = len(results) - n_ok

    if args.json:
        print(json.dumps({"n_derived": n_ok, "n_skipped": n_skip, "results": results}, indent=2))
    else:
        print(f"Derived: {n_ok}  Skipped: {n_skip}  Total: {len(results)}")
        for r in results:
            status = "SKIP" if r["skipped"] else "OK  "
            reason = f"  ({r['reason']})" if r["skipped"] else ""
            print(f"  {status} {r['slug']}{reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
