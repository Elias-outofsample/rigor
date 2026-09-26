"""Unified optimisation-pipeline orchestrator with calibration profiles.

The framework already ships every optimisation *stage* (search, select,
CPCV-PBO, deflated Sharpe, holdout, falsification, walk-forward, enrichment) and
a ``robust`` selection ladder with floors. What it lacked was the *one-command*
research→selection→deployment-readiness flow under a **named calibration
profile** — the discovery-vs-deployment split. This module adds exactly that
orchestration layer and nothing else: it introduces no new statistics, it
chains the existing stages and reads the existing verdict.

``run_pipeline`` does four things, in order:

  1. **Search** the strategy's ``param_grid()`` (reusing :mod:`rigor.optimize.grid`
     to enumerate / Sobol-sample) and backtest every config on the cache the
     strategy builds once.
  2. **Validate the search.** It runs CPCV-PBO over the configs' return matrix,
     grades the winner through the shared :func:`rigor.validation.validate` (so the
     deflated Sharpe accounts for the number of trials searched), and runs the
     holdout + falsification stages on the winner.
  3. **Select under the profile.** The winner is chosen by the existing
     ``robust`` ladder (:func:`rigor.optimize.select.select`) using the *profile's*
     floors, so a permissive ``discovery`` profile keeps configs a strict
     ``deployment`` profile rejects.
  4. **Gate for deployment.** It checks the selected winner against the profile's
     beyond-the-ladder thresholds (deflated-Sharpe floor, activity floor) *and*
     the framework's own promotion verdict — the single source of truth
     (:mod:`rigor.validation.verdict`). The gate's verdict ban-list mirrors the
     enforced promotion gate (:mod:`rigor.project.promotion_gate`).

The result is a single ``outcome`` — ``PROMOTE`` / ``REJECT`` for ``deployment``,
advisory ``ADMIT`` / ``WITHHOLD`` for ``discovery`` — plus per-stage detail.

It is deterministic (every stage it calls is) and non-destructive: it never
writes ``config.json``; the per-stage JSON artifact is written only when
``write=True``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import metrics as _m
from ..data import DataConfig, DataLoader
from ..engine import BacktestResult, result_from_returns
from ..validation import validate
from ..validation.accel import active_backend
from ..validation.cpcv import cpcv_pbo
from . import grid as _grid
from .calibration import CalibrationProfile, get_profile
from .falsification import run_falsification
from .holdout import holdout_validate
from .select import config_metrics, select

__all__ = ["PipelineResult", "run_pipeline"]

# Outcome vocabulary. ``deployment`` speaks the promotion language
# (PROMOTE/REJECT); ``discovery`` is advisory and never blocks, so it uses a
# softer pair that cannot be mistaken for a promotion decision.
_DEPLOY_PASS, _DEPLOY_FAIL = "PROMOTE", "REJECT"
_DISCOVER_PASS, _DISCOVER_FAIL = "ADMIT", "WITHHOLD"


@dataclass
class PipelineResult:
    """Structured result of one profiled optimisation run.

    ``outcome`` is the single headline decision; ``gate`` carries the per-check
    detail (which floors passed / failed and why); ``stages`` is the raw
    per-stage report for full traceability.
    """

    slug: str
    profile: str
    outcome: str                       # PROMOTE/REJECT (deployment) | ADMIT/WITHHOLD (discovery)
    deploy_ready: bool                 # True iff outcome is the passing value
    selected_params: dict | None       # the chosen config (None when the grid is REJECTED)
    selection_priority: str            # ROBUST_PERF | REJECTED
    gate: dict[str, Any] = field(default_factory=dict)
    profile_thresholds: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return _m.compute_sharpe(r.to_numpy(dtype="float64"), _m.periods_per_year_of(r.index))


def _as_series(ret: Any) -> pd.Series:
    if isinstance(ret, BacktestResult):
        ret = ret.returns
    return ret if isinstance(ret, pd.Series) else pd.Series(np.asarray(ret, dtype="float64"))


def _search(strategy, combos: list[dict]) -> list[dict]:
    """Backtest every config on the shared cache; return rows ranked by Sharpe.

    Each row is ``{"params", "sharpe", "returns"}``. The ordering is the single
    authoritative one (Sharpe desc, NaN last) — the same key the standalone
    optimiser uses, so the winner is consistent across entry points.
    """
    cache = strategy.cache
    rows = [
        {"params": c, "sharpe": _sharpe(r), "returns": r}
        for c in combos
        for r in (_as_series(strategy.run_backtest(cache, c)),)
    ]
    rows.sort(key=lambda x: (x["sharpe"] if x["sharpe"] == x["sharpe"] else -1e18), reverse=True)
    return rows


def _winner_active_bars(report: dict) -> int | None:
    """The selected winner's active-bar count, if the ladder recorded it."""
    met = (report.get("selection") or {}).get("selected_metrics") or {}
    bars = met.get("active_bars")
    return int(bars) if isinstance(bars, (int, float)) else None


def _evaluate_gate(report: dict, profile: CalibrationProfile) -> dict[str, Any]:
    """Score the selected winner against the profile's deployment-readiness gate.

    The selection ladder already enforced the per-config / grid floors
    (Sharpe / PSR / CAGR / drawdown / PBO) under the profile. This gate adds the
    checks the ladder does not cover and ties the final call to the shared
    promotion verdict:

      * **selection** — a config survived the ladder (a REJECTED grid fails);
      * **deflated_sharpe** — the winner's DSR P(skill) clears ``dsr_min``;
      * **activity** — the winner trades on >= ``min_active_bars`` bars;
      * **verdict** — the shared promotion verdict is not in the profile's
        ``blocking_verdicts`` (REJECT always blocks; deployment also blocks on
        CONDITIONAL).

    A check whose input is unavailable is recorded as ``neutral`` and excluded
    from the reduction, so the gate never fails on missing evidence.
    """
    checks: list[dict[str, Any]] = []

    selection = report.get("selection") or {}
    priority = selection.get("priority", "REJECTED")
    selected = priority != "REJECTED"
    checks.append({
        "name": "selection",
        "passed": selected,
        "detail": (selection.get("reason") if not selected else f"selected via {priority}"),
    })

    dsr = (report.get("overfit") or {}).get("dsr")
    if dsr is None:
        checks.append({"name": "deflated_sharpe", "neutral": True, "detail": "no DSR recorded"})
    else:
        checks.append({
            "name": "deflated_sharpe", "passed": float(dsr) >= profile.dsr_min,
            "value": round(float(dsr), 4), "threshold": profile.dsr_min,
        })

    bars = _winner_active_bars(report)
    if bars is None:
        checks.append({"name": "activity", "neutral": True,
                       "detail": "no active-bar count recorded"})
    else:
        checks.append({
            "name": "activity", "passed": bars >= profile.min_active_bars,
            "value": bars, "threshold": profile.min_active_bars,
        })

    verdict = (report.get("verdict") or {}).get("verdict")
    if verdict is None:
        checks.append({"name": "verdict", "neutral": True, "detail": "no verdict recorded"})
    else:
        checks.append({
            "name": "verdict", "passed": verdict not in profile.blocking_verdicts,
            "value": verdict, "threshold": f"not in {sorted(profile.blocking_verdicts)}",
        })

    scored = [c for c in checks if not c.get("neutral")]
    passed = all(c.get("passed", False) for c in scored)
    return {
        "passed": passed,
        "n_checks": len(scored),
        "n_failed": sum(1 for c in scored if not c.get("passed", False)),
        "checks": checks,
    }


def run_pipeline(
    strategy_path: str | Path, *, profile: str | CalibrationProfile = "discovery",
    as_of: str | None = None, max_combos: int = 512, offline: bool = False,
    data: DataLoader | None = None, write: bool = False,
) -> PipelineResult:
    """Run the optimisation pipeline under a named calibration *profile*.

    One command, research→selection→deployment-readiness: search the grid,
    validate the winner through the shared anti-overfit battery (CPCV-PBO,
    deflated Sharpe, holdout, falsification), **select under the profile's
    floors** (so ``discovery`` admits more than ``deployment``), then **gate**
    the winner against the shared promotion verdict and the profile's
    deployment-readiness thresholds.

    Returns a :class:`PipelineResult`. ``write=True`` persists the per-stage
    report to ``artifacts/<slug>_pipeline_<profile>.json`` (gitignored); the
    pipeline never edits ``config.json``. Raises if the strategy has no
    multi-value ``param_grid()`` — there is nothing to optimise.
    """
    prof = profile if isinstance(profile, CalibrationProfile) else get_profile(profile)
    floors = prof.robust_floors()

    from ..project import layout
    from ..project.run import load_config, load_strategy_module

    path = Path(strategy_path)
    config, raw = load_config(path)
    slug = raw.get("slug") or path.name
    data = data or DataLoader(DataConfig.from_env(as_of=as_of, offline=offline))
    strategy = load_strategy_module(path, slug).build(config, data)

    full_grid = strategy.param_grid()
    grid_combos = _grid.expand(full_grid)
    if len(grid_combos) < 2:
        raise ValueError(
            f"{slug} has no multi-value param_grid() — nothing to optimise. Declare a "
            "param_grid (e.g. {'lookback': [20, 50, 100]}) on the strategy to opt in."
        )

    # --- Stage 1: search --------------------------------------------------
    default = strategy.default_params()
    combos = _grid.sample(grid_combos, max_combos, default=default)
    rows = _search(strategy, combos)
    n_evaluated = len(rows)

    # --- Stage 2: validate the search (CPCV-PBO + shared verdict) ---------
    matrix = pd.concat([x["returns"].rename(i) for i, x in enumerate(rows)], axis=1).dropna()
    cpcv = None
    if len(matrix) >= 20:
        cpcv = cpcv_pbo(matrix.to_numpy().T, ppy=_m.periods_per_year_of(matrix.index))

    # --- Stage 3: select under the profile's floors -----------------------
    selection = select(rows, mode="robust", floors=floors, pbo=(cpcv or {}).get("pbo"))
    winner_idx = selection["index"] if selection["index"] is not None else 0
    winner = rows[winner_idx]

    val = validate(result_from_returns(winner["returns"]), n_trials=n_evaluated, cpcv=cpcv)
    ppy = _m.periods_per_year_of(winner["returns"].dropna().index)

    default_sharpe = next((x["sharpe"] for x in rows if x["params"] == default), float("nan"))
    report: dict[str, Any] = {
        "slug": slug, "as_of": data.cfg.as_of_date, "profile": prof.name,
        "backend": active_backend("auto"),
        "grid_size": len(grid_combos), "configs_evaluated": n_evaluated,
        "select_mode": "robust", "selection": selection,
        "default_params": default, "default_sharpe": round(float(default_sharpe), 4),
        "default_in_grid": any(x["params"] == default for x in rows),
        "best_params": winner["params"], "best_sharpe": round(float(winner["sharpe"]), 4),
        "ranking": [{"params": x["params"], "sharpe": round(float(x["sharpe"]), 4)}
                    for x in rows[:25]],
        "overfit": val["overfit"], "cpcv": val["cpcv"], "verdict": val["verdict"],
        # --- Stage 4: robustness checks on the winner ---------------------
        "holdout": holdout_validate(winner["returns"], ppy),
        "falsification": run_falsification(winner["returns"], ppy),
    }
    # Backfill the active-bar count for the activity gate when the ladder
    # didn't record selected_metrics (e.g. the grid was REJECTED and we fell
    # back to the top-Sharpe row): compute it from the winner directly.
    if _winner_active_bars(report) is None:
        bars = config_metrics(winner["returns"])["active_bars"]
        report.setdefault("selection", {}).setdefault("selected_metrics", {})["active_bars"] = bars

    gate = _evaluate_gate(report, prof)
    report["pipeline_gate"] = gate
    report["profile_thresholds"] = prof.as_dict()

    if write:
        art = layout.artifacts_dir(path)
        art.mkdir(parents=True, exist_ok=True)
        import json
        (art / f"{slug}_pipeline_{prof.name}.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")

    deploy_ready = bool(gate["passed"])
    if prof.name == "deployment":
        outcome = _DEPLOY_PASS if deploy_ready else _DEPLOY_FAIL
    else:
        outcome = _DISCOVER_PASS if deploy_ready else _DISCOVER_FAIL

    return PipelineResult(
        slug=slug,
        profile=prof.name,
        outcome=outcome,
        deploy_ready=deploy_ready,
        selected_params=(selection.get("params") if selection.get("priority") != "REJECTED"
                         else None),
        selection_priority=selection.get("priority", "REJECTED"),
        gate=gate,
        profile_thresholds=prof.as_dict(),
        stages=report,
    )
