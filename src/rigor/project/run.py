"""Run a standardized strategy folder -> standardized artifacts + report card."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ..data import DataConfig, DataLoader, build_loader
from ..report import write_report_card
from ..strategy import StrategyConfig
from . import layout

_CONFIG_FIELDS = {
    "name", "start_date", "end_date", "initial_capital", "commission_bps",
    "long_short", "max_positions", "rebalance_freq", "role", "status", "version",
    "ledger_exempt", "optimize_exempt_reason", "extra",
}


def load_config(strategy_path: Path) -> tuple[StrategyConfig, dict]:
    """Parse config.json -> (StrategyConfig, raw dict with slug/category/...)."""
    raw = json.loads((Path(strategy_path) / "config.json").read_text(encoding="utf-8"))
    kwargs = {k: v for k, v in raw.items() if k in _CONFIG_FIELDS}
    return StrategyConfig(**kwargs), raw


def load_strategy_module(strategy_path: Path, slug: str):
    """Import the folder's strategy.py in isolation and return the module."""
    path = Path(strategy_path) / "strategy.py"
    spec = importlib.util.spec_from_file_location(f"rigor_strategy__{slug}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_strategy(
    strategy_path: str | Path, *, as_of: str | None = None,
    params: dict | None = None, data: DataLoader | None = None,
    thesis: bool = True, validate: bool = True, offline: bool = False,
) -> dict:
    """Load, backtest, validate, and write artifacts (returns.csv, summary.json,
    report.html, and — unless ``thesis=False`` — the thesis PDF).

    ``offline=True`` reads only the frozen snapshot (no live fetch) for guaranteed
    reproducibility."""
    path = Path(strategy_path)
    config, raw = load_config(path)
    slug = raw.get("slug") or path.name
    category = raw.get("category", "other")

    # Default loader is the best-available-data superset: local Databento 1-minute bars
    # for catalogued CME futures/crypto (ES/NQ/RTY/CL/GC/SI/BTC/ETH), EODHD-delegated for
    # everything else. Equity/ETF strategies behave identically; futures strategies get the
    # deepest, highest-resolution series. An explicit data= still overrides.
    data = data or build_loader(DataConfig.from_env(as_of=as_of, offline=offline))
    module = load_strategy_module(path, slug)
    if not hasattr(module, "build"):
        raise AttributeError(f"{path/'strategy.py'} must define build(config, data)")
    strategy = module.build(config, data)
    result = strategy.backtest(params)

    # Expected excess return: subtract the period-average risk-free (3-month T-bill,
    # FRED DGS3MO) from the annualized expected return. compute_core_metrics defaults
    # to rf=0 (so ev_excess_annual == ev_annual); here, where the data layer is
    # available, we refine it to a TRUE excess. Degrades gracefully to rf=0 if the
    # series is unavailable (e.g. offline without it cached) — never breaks a run.
    try:
        _rf = data.fred("DGS3MO")
        _idx = result.returns.index
        _win = _rf[(_rf.index >= _idx[0]) & (_rf.index <= _idx[-1])]
        _rf_annual = float(_win.mean()) / 100.0 if len(_win) else 0.0
        _ev_annual = result.metrics.get("ev_annual")
        if _ev_annual is not None and _rf_annual:
            result.metrics["risk_free_annual"] = _rf_annual
            result.metrics["ev_excess_annual"] = _ev_annual - _rf_annual
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: risk-free (DGS3MO) unavailable; ev_excess uses rf=0 "
              f"({type(exc).__name__}: {exc})")

    validation = None
    if validate:
        # Validation must never break a run; degrade gracefully.
        try:
            from ..validation import validate_strategy
            validation = validate_strategy(strategy, result)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: validation skipped ({type(exc).__name__}: {exc})")

    art_dir = layout.artifacts_dir(path)
    art_dir.mkdir(parents=True, exist_ok=True)
    paths = layout.artifact_paths(path, slug)
    result.returns.to_csv(paths["returns"], header=True)
    # Position ledger reconstructed from the engine's target weights (per-symbol
    # entry/exit spells). Skipped for returns-only strategies (no weights). Must
    # never break a run.
    try:
        from .trades import write_position_ledger
        write_position_ledger(path, slug, result)
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: position ledger skipped ({type(exc).__name__}: {exc})")
    write_report_card(result, art_dir, slug, name=config.name, as_of=as_of, validation=validation)

    if thesis:
        # The thesis PDF is a standard output of every run. Its deps (reportlab/
        # matplotlib/svglib/lxml) are CORE, so this should always succeed; a failure
        # must never lose the backtest, but it is surfaced loudly (not a quiet warn).
        try:
            from ..thesis_card import write_thesis
            thesis_path = write_thesis(
                result, art_dir, slug, name=config.name, category=category, as_of=as_of
            )
            paths["thesis"] = thesis_path
        except Exception as exc:  # noqa: BLE001
            bar = "!" * 72
            missing = isinstance(exc, ImportError)
            print(f"  {bar}")
            print(f"  WARNING: thesis PDF was NOT generated  ({type(exc).__name__}: {exc})")
            if missing:
                print("  A required dependency is missing. Reinstall the framework:")
                print("      pip install -e .        (thesis deps are included)")
            else:
                print("  The backtest succeeded, but the thesis step errored above.")
                print("  The thesis is a required output — fix the cause and re-run.")
            print(f"  {bar}")

    # Refresh the shared strategy catalog (must never break a run).
    try:
        from .index import write_index
        write_index()
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: catalog update skipped ({type(exc).__name__}: {exc})")

    return {"slug": slug, "result": result, "artifacts": paths, "validation": validation}
