"""Opt-in parameter optimisation — orchestrator.

Search a strategy's ``param_grid()`` for the best configuration and report it
through the anti-overfit machinery the framework already ships, plus the
data-snooping tests and search-generalisation diagnostics ported from the prior
research library. Nothing here duplicates the verdict/scoring — the winner is
graded by ``rigor.validation`` exactly like any single backtest.

Pipeline (every stage after the search is opt-in via flags):

  1. **Search** — enumerate the grid (sampled if huge), or run a **two-pass**
     coarse->fine search that visits a fraction of a large grid on modest
     hardware. The data cache is built once and reused for every config.
  2. **Select** — ``sharpe`` (top Sharpe) or ``robust`` (robustness floors then
     performance). See ``select``.
  3. **Anti-overfit** — grade the winner through ``rigor.validation`` with
     ``n_trials`` = configs searched, so the **deflated Sharpe** and **CPCV-PBO**
     account for the search.
  4. **Data snooping** — White Reality Check / Hansen SPA / Romano-Wolf StepM:
     is the winner's edge over the *family* of configs real, or luck-of-the-max?
  5. **Walk-forward** — re-select per in-sample fold, measure out-of-sample
     (expanding or rolling), so you see whether the *search itself* generalises.
  6. **Ensemble** (optional) — average a diverse few top configs; if the
     ensemble holds up, the edge is broad rather than a single lucky point.

It is **non-destructive**: it writes a local ``artifacts/<slug>_optimization.json``
report (gitignored) and never edits ``config.json``. You decide whether to adopt
the winning params and bump the ``version``.

GPU/CUDA grid-search routing (additive, default-identical)
----------------------------------------------------------
The search loop (``_run``) can optionally route the *whole* combo list through a
strategy-supplied batched kernel (``StrategyBase.grid_kernel`` — see its
docstring) instead of the per-combo CPU loop. This is a pure speed optimisation,
gated so the default behaviour is **byte-identical**:

  * The kernel is used ONLY when ALL of these hold — the strategy defines a
    ``grid_kernel()`` (most do not, so they are unaffected); the resolved backend
    is CUDA and CuPy/a device is actually available (``active_backend`` says
    ``cuda``); and the grid is large enough to amortise a GPU launch
    (``len(combos) >= _KERNEL_MIN_COMBOS``). Otherwise the existing CPU loop runs.
  * **Determinism is preserved end to end**: the kernel only *produces the
    per-combo bar returns faster*. Every authoritative ``sharpe`` is still
    computed by the same ``_sharpe`` helper (``rigor.metrics``) on those returns,
    and the ranking uses the identical sort key, so a CPU run and a kernel run on
    the same combos yield identical ``rows``. The kernel never supplies a Sharpe.
  * The whole kernel attempt is wrapped in ``try/except``: any failure (raise,
    ``None``, or a wrong-shaped tensor) falls back to the CPU loop. The
    optimisation never crashes because of the kernel.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as _m
from ..data import DataConfig, DataLoader
from ..engine import BacktestResult, result_from_returns
from ..validation import (
    hansen_spa_test,
    stepm_test,
    validate,
    white_reality_check,
)
from ..validation.accel import active_backend, backends_available
from ..validation.cpcv import cpcv_pbo
from . import grid as _grid
from .clustering import cluster_parameters
from .enrichment import enrich_equity
from .ensemble import build_ensemble
from .falsification import run_falsification
from .holdout import holdout_validate
from .regime import optimize_per_regime
from .scoring import score_configs
from .select import RobustFloors, select
from .walkforward import walk_forward_opt

__all__ = ["optimize_strategy"]

# Minimum combo count for the GPU kernel path to be worth a launch; below this the
# CPU loop wins (kernel transfer/launch overhead dominates). Below this threshold
# the CPU loop is used even when a kernel and a CUDA device are both available.
_KERNEL_MIN_COMBOS = 1000


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return _m.compute_sharpe(r.to_numpy(dtype="float64"), _m.periods_per_year_of(r.index))


def _as_series(ret) -> pd.Series:
    if isinstance(ret, BacktestResult):
        ret = ret.returns
    return ret if isinstance(ret, pd.Series) else pd.Series(np.asarray(ret, dtype="float64"))


def _dedup(combos: list[dict]) -> list[dict]:
    """First-seen-order de-duplication of the combo list (shared by both paths)."""
    out, seen = [], set()
    for c in combos:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _rank(rows: list[dict]) -> list[dict]:
    """Sort rows by Sharpe desc (NaN last) — the single authoritative ordering."""
    rows.sort(key=lambda x: (x["sharpe"] if x["sharpe"] == x["sharpe"] else -1e18), reverse=True)
    return rows


def _use_kernel(strategy, combos: list[dict], backend: str) -> bool:
    """Route to the GPU kernel ONLY when a kernel exists, CUDA is truly available,
    and the grid is large enough to amortise a launch — otherwise the CPU loop."""
    if len(combos) < _KERNEL_MIN_COMBOS:
        return False
    if getattr(strategy, "grid_kernel", None) is None or strategy.grid_kernel() is None:
        return False
    # active_backend resolves to "cuda" only when CuPy + a real device are present
    # (see rigor.validation.accel), so this is the single source of CUDA truth.
    return active_backend(backend) == "cuda" and backends_available().get("cuda", False)


def _run_kernel(strategy, cache, combos: list[dict]) -> list[dict]:
    """Evaluate the de-duplicated combos via the strategy's batched grid kernel.

    The kernel only *produces* the (n_combos, n_bars) returns tensor faster; every
    Sharpe and the ranking are still computed here from those returns, so the rows
    are byte-identical to the CPU loop. Raises on any contract violation so the
    caller's try/except can fall back to the CPU path.
    """
    kernel = strategy.grid_kernel()
    tensor = kernel(cache, combos)
    if tensor is None:
        raise ValueError("grid_kernel returned None")
    tensor = np.asarray(tensor, dtype="float64")
    if tensor.ndim != 2 or tensor.shape[0] != len(combos):
        raise ValueError(f"grid_kernel returned shape {tensor.shape}, expected ({len(combos)}, n)")
    dates = cache.get("dates") if isinstance(cache, dict) else None
    index = None
    if dates is not None:
        # Preserve the cache index object (and its freq/name metadata) when it is
        # already a DatetimeIndex, so a kernel row is byte-identical to a CPU row
        # whose Series carries that same index. Only coerce for non-Index dates.
        idx = dates if isinstance(dates, pd.DatetimeIndex) else pd.DatetimeIndex(np.asarray(dates))
        if len(idx) >= tensor.shape[1]:
            index = idx[-tensor.shape[1]:]  # align oldest-first, like the bare-array CPU path
    rows = []
    for i, c in enumerate(combos):
        r = pd.Series(tensor[i], index=index)
        rows.append({"params": c, "sharpe": _sharpe(r), "returns": r})
    return _rank(rows)


def _run(strategy, cache, combos: list[dict], *, backend: str = "auto") -> list[dict]:
    """Backtest each config (reusing the prebuilt cache); return ranked rows.

    Default path is the per-combo CPU loop. When the strategy ships a CUDA grid
    kernel, a device is available, and the grid is large enough, the whole combo
    list is routed through the kernel instead (see the module note). Any kernel
    failure falls back to the identical CPU loop.
    """
    combos = _dedup(combos)
    if _use_kernel(strategy, combos, backend):
        try:
            return _run_kernel(strategy, cache, combos)
        except Exception:  # noqa: BLE001 - any kernel failure -> CPU fallback, never crash
            pass
    rows = [
        {"params": c, "sharpe": _sharpe(r), "returns": r}
        for c in combos
        for r in (_as_series(strategy.run_backtest(cache, c)),)
    ]
    return _rank(rows)


def _snoop(rows: list[dict], benchmark, *, max_family: int, n_bootstrap: int,
           backend: str = "auto") -> dict:
    """Data-snooping tests: does the best of the searched family beat the benchmark
    (the untuned default), accounting for having searched the whole family?"""
    family = [r["returns"] for r in rows[:max_family]]
    if len(family) < 5:
        return {"note": "too few configs for data-snooping tests (need >= 5)"}
    return {
        "n_configs": len(family),
        "benchmark": "default_config" if benchmark is not None else "cash",
        "white_reality_check": white_reality_check(family, benchmark, n_bootstrap=n_bootstrap,
                                                   backend=backend),
        "hansen_spa": hansen_spa_test(family, benchmark, n_bootstrap=n_bootstrap, backend=backend),
        "romano_wolf_stepm": stepm_test(family, benchmark, n_bootstrap=n_bootstrap,
                                        backend=backend),
    }


def _read_trial_log(log_path: Path) -> dict:
    """Return the persisted trial ledger, or a zeroed default if absent/corrupt."""
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "cumulative_n_trials" in data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema_version": 1, "runs": [], "cumulative_n_trials": 0}


def _update_trial_log(log_path: Path, slug: str, n_trials: int, grid_dims: dict) -> None:
    """Append a run entry and persist the updated ledger atomically (temp + os.replace)."""
    ledger = _read_trial_log(log_path)
    ledger.setdefault("schema_version", 1)
    ledger["slug"] = slug
    ledger.setdefault("runs", [])
    ledger["runs"].append({
        "ts": datetime.now(UTC).isoformat(),
        "n_trials": int(n_trials),
        "grid_dims": grid_dims,
    })
    ledger["cumulative_n_trials"] = int(ledger.get("cumulative_n_trials", 0)) + int(n_trials)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=log_path.parent, prefix=log_path.name + ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=2, default=str)
        os.replace(tmp, log_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def optimize_strategy(
    strategy_path: str | Path, *, as_of: str | None = None, max_combos: int = 512,
    n_wf_folds: int = 5, wf_mode: str = "expanding", offline: bool = False,
    select_mode: str = "sharpe", two_pass: bool = False, coarse_combos: int = 64,
    refine_span: int = 1, ensemble: int | None = None, snoop_max_alt: int = 64,
    snoop_bootstrap: int = 500, full: bool = False, enrich: bool = False,
    holdout: bool = False, falsify: bool = False, score: bool = False,
    regime: bool = False, cluster: bool = False, backend: str = "auto",
    data: DataLoader | None = None, write: bool = True,
    cumulative_trials: bool = True,
) -> dict:
    """Search ``strategy_path``'s param grid and return an overfit-aware report.

    The base pipeline (search, select, deflated-Sharpe + CPCV-PBO verdict, data
    snooping, walk-forward) always runs. The remaining stages are opt-in and add
    compute; ``full=True`` enables them all: ``enrich`` (stage-3 robustness
    scalars on the winner), ``holdout`` (final-20% slice), ``falsify`` (noise
    injection), ``score`` (5-pillar composite selection), ``regime`` (per-regime
    winners), ``cluster`` (parameter clusters).

    Writes ``artifacts/<slug>_optimization.json`` unless ``write=False``. Raises
    if the strategy has no multi-value ``param_grid()``.

    When ``cumulative_trials=True`` (default), prior optimization campaigns are
    tracked in ``extra/<slug>_n_trials_log.json`` and the DSR/deflation uses
    the cumulative trial count across all runs (``effective_n_trials``). Set
    ``cumulative_trials=False`` to revert to today-only accounting (no ledger
    read/write).
    """
    if full:
        enrich = holdout = falsify = score = regime = cluster = True
        if ensemble is None:
            ensemble = 5
    from ..project import layout  # lazy: avoids a project<->optimize import cycle
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
    default = strategy.default_params()
    cache = strategy.cache  # built once, reused across every configuration

    search = "one_pass"
    if two_pass and len(grid_combos) > coarse_combos:
        coarse = _grid.expand(_grid.decimate(full_grid, coarse_combos))
        coarse_rows = _run(strategy, cache, coarse, backend=backend)
        fine_grid = _grid.refine_around(full_grid, coarse_rows[0]["params"], span=refine_span)
        fine = _grid.expand(fine_grid)
        rows = _run(strategy, cache, [*coarse, *fine], backend=backend)
        search = "two_pass"
    else:
        combos = _grid.sample(grid_combos, max_combos, default=default)
        rows = _run(strategy, cache, combos, backend=backend)
    n_evaluated = len(rows)

    # cumulative multiple-testing exposure across optimization campaigns.
    # The ledger is only read/written when write=True (a persisted run); when
    # write=False (dry-run / test), prior_cumulative=0 so behaviour is identical
    # to pre-feature runs.
    art = layout.artifacts_dir(path)
    extra = layout.extra_dir(path)
    log_path = extra / f"{slug}_n_trials_log.json"          # auxiliary diagnostic -> extra/
    if cumulative_trials and write:
        _prior = _read_trial_log(log_path)
        if _prior.get("cumulative_n_trials", 0) == 0:
            # backward-compat: read a pre-existing ledger from the old artifacts/ home
            _legacy = art / f"{slug}_n_trials_log.json"
            if _legacy.exists():
                _prior = _read_trial_log(_legacy)
        prior_cumulative = int(_prior.get("cumulative_n_trials", 0))
    else:
        prior_cumulative = 0
    # effective_n_trials is the count passed to DSR/deflation; it accounts for all
    # prior campaigns so the multiple-testing penalty reflects cumulative exposure.
    effective_n_trials = prior_cumulative + n_evaluated
    cumulative_n_trials = prior_cumulative + n_evaluated  # same value, named for the report

    # anti-overfit verdict on the winner, deflated for the number of configs searched
    matrix = pd.concat([x["returns"].rename(i) for i, x in enumerate(rows)], axis=1).dropna()
    cpcv = None
    if len(matrix) >= 20:
        cpcv = cpcv_pbo(matrix.to_numpy().T, ppy=_m.periods_per_year_of(matrix.index))

    selection = select(rows, mode=select_mode, floors=RobustFloors(),
                       pbo=(cpcv or {}).get("pbo"))
    sel = selection["index"] if selection["index"] is not None else 0
    best = rows[sel]

    val = validate(result_from_returns(best["returns"]), n_trials=effective_n_trials, cpcv=cpcv)
    default_returns = next((x["returns"] for x in rows if x["params"] == default), None)
    snoop = _snoop(rows, default_returns, max_family=snoop_max_alt,
                   n_bootstrap=snoop_bootstrap, backend=backend)
    wf = walk_forward_opt(matrix, n_wf_folds, mode=wf_mode)

    default_sharpe = next((x["sharpe"] for x in rows if x["params"] == default), float("nan"))
    report = {
        "slug": slug, "as_of": data.cfg.as_of_date, "metric": "sharpe", "search": search,
        "backend": active_backend(backend), "backends_available": backends_available(),
        "grid_size": len(grid_combos), "configs_evaluated": n_evaluated,
        "effective_n_trials": effective_n_trials,
        "cumulative_n_trials": cumulative_n_trials,
        "select_mode": select_mode, "selection": selection,
        "default_params": default, "default_sharpe": round(default_sharpe, 4),
        "default_in_grid": default_returns is not None,
        "best_params": best["params"], "best_sharpe": round(best["sharpe"], 4),
        "improvement_vs_default": round(best["sharpe"] - default_sharpe, 4),
        "ranking": [{"params": x["params"], "sharpe": round(x["sharpe"], 4)} for x in rows[:25]],
        "overfit": val["overfit"], "cpcv": val["cpcv"], "verdict": val["verdict"],
        "data_snooping": snoop, "walk_forward_opt": wf,
    }
    if ensemble:
        report["ensemble"] = build_ensemble(rows, n_members=ensemble)
    ppy = _m.periods_per_year_of(best["returns"].dropna().index)
    if enrich:
        report["enrichment"] = enrich_equity(best["returns"], ppy)
    if holdout:
        report["holdout"] = holdout_validate(best["returns"], ppy)
    if falsify:
        report["falsification"] = run_falsification(best["returns"], ppy)
    if score:
        report["selection_score_5pillar"] = score_configs(rows, cpcv=cpcv)
    if regime:
        report["regime"] = optimize_per_regime(rows)
    if cluster:
        report["clustering"] = cluster_parameters(rows)
    if write:
        art.mkdir(parents=True, exist_ok=True)
        (art / f"{slug}_optimization.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    if cumulative_trials and write:
        extra.mkdir(parents=True, exist_ok=True)
        _update_trial_log(log_path, slug, n_evaluated, {k: len(v) for k, v in full_grid.items()})
    return report
