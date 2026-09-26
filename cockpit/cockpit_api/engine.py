"""Adapter between the API and the shared ``portfolio_engine`` engine.

Discovers the strategy book once per process (refreshable), resolves request slugs
to candidates, runs the shared ``build_portfolio`` pipeline, and serialises the
result bundle (BacktestResult / DataFrames / Series) into JSON-safe primitives the
browser can render. No FastAPI imports here — pure adaptation, easy to unit-test.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pandas as pd

from portfolio_engine import (
    build_from_weights,
    build_portfolio,
    list_saved,
    save_portfolio,
    scan_strategies,
    search_portfolios,
)

from .schemas import BuildRequest, Constraints, DetailRequest, SaveRequest, SearchRequest
from .settings import get_settings

_lock = threading.Lock()
_cands: list | None = None

_METRIC_KEYS = ("sharpe", "cagr", "max_drawdown", "volatility", "sortino", "calmar",
                "var_95", "cvar_95", "ulcer_index", "n_obs")


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #
def candidates(refresh: bool = False) -> list:
    """The discovered strategy candidates (scanned once, then cached)."""
    global _cands
    with _lock:
        if _cands is None or refresh:
            _cands = scan_strategies(get_settings().strategy_root)
        return _cands


def rescan() -> int:
    return len(candidates(refresh=True))


def list_strategies() -> list[dict]:
    return [_strategy_row(c) for c in candidates()]


def _start_year(c) -> int | None:
    idx = getattr(c, "returns", None)
    if idx is None or len(idx.index) == 0:
        return None
    return int(idx.index.min().year)


def _strategy_row(c) -> dict:
    return clean({
        "slug": c.slug, "name": c.name, "category": c.category,
        "sharpe": c.sharpe, "cagr": c.cagr, "max_dd": c.max_drawdown,
        "pbo": c.pbo, "verdict": c.verdict, "n_obs": int(len(c.returns)),
        "start_year": _start_year(c),
    })


def _resolve(slugs: list[str]) -> list:
    by_slug = {c.slug: c for c in candidates()}
    return [by_slug[s] for s in slugs if s in by_slug]


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def _constraints_to_dict(c: Constraints) -> dict:
    return {
        "min_sharpe": c.min_sharpe, "min_cagr": c.min_cagr / 100.0, "max_beta": c.max_beta,
        "max_corr_bench": c.max_corr_bench, "max_corr_strat": c.max_corr_strat,
        "min_divers": c.min_divers, "max_risk": c.max_risk / 100.0,
        "max_dd": c.max_dd / 100.0, "max_vol": c.max_vol / 100.0,
    }


def _require(slugs: list[str]) -> list:
    selected = _resolve(slugs)
    if len(selected) < 2:
        raise ValueError("need at least two known strategies (check the slugs)")
    return selected


def _filter_by_year(selected: list, min_start_year: int) -> list:
    """Keep only strategies with data going back to ``min_start_year`` or earlier."""
    if not min_start_year:
        return selected
    out = [c for c in selected if (_start_year(c) or 9999) <= min_start_year]
    if len(out) < 2:
        raise ValueError(f"fewer than 2 strategies have data since {min_start_year} or earlier")
    return out


def search(req: SearchRequest) -> dict:
    """Grid-search every portfolio meeting the criteria; return the ranked candidates.

    Weights come back keyed by **slug** (the id the browser uses), each with headline
    metrics, so the UI can list them and open one via :func:`detail`.
    """
    selected = _filter_by_year(_require(req.slugs), req.min_start_year)
    max_w = (req.max_weight / 100.0) or None
    cands = search_portfolios(selected, _constraints_to_dict(req.constraints),
                              max_weight=max_w, use_cpcv=req.use_cpcv, n_samples=req.n_samples,
                              min_strats=req.min_strats, max_strats=req.max_strats,
                              common_window=req.common_window)
    name_to_slug = {c.name: c.slug for c in selected}
    for cand in cands:
        cand["weights"] = {name_to_slug.get(n, n): w for n, w in cand["weights"].items()}
    return clean({"candidates": cands, "n": len(cands),
                  "min_sharpe": req.constraints.min_sharpe,
                  "common_window": req.common_window})


def detail(req: DetailRequest) -> dict:
    """Open one chosen candidate (slug→weight) and return the full dossier."""
    selected = _require(req.slugs)
    slug_to_name = {c.slug: c.name for c in selected}
    weights_by_name = {slug_to_name.get(s, s): float(w) for s, w in req.weights.items()}
    bundle = build_from_weights(selected, weights_by_name, _constraints_to_dict(req.constraints),
                                common_window=req.common_window)
    return serialize(bundle)


def build(req: BuildRequest) -> dict:
    """Direct single allocation (a chosen method) → full dossier. Kept for parity."""
    selected = _require(req.slugs)
    max_w = (req.max_weight / 100.0) or None
    bundle = build_portfolio(selected, req.method, max_w, req.leverage, req.use_cpcv,
                             _constraints_to_dict(req.constraints))
    return serialize(bundle)


def save(req: SaveRequest) -> dict:
    """Persist a chosen candidate to ``portfolios/<name>/`` (definition + artifacts + report)."""
    selected = _require(req.slugs)
    slug_to_name = {c.slug: c.name for c in selected}
    weights_by_name = {slug_to_name.get(s, s): float(w) for s, w in req.weights.items()}
    bundle = build_from_weights(selected, weights_by_name, _constraints_to_dict(req.constraints),
                                common_window=req.common_window)
    folder = save_portfolio(
        req.name, bundle["selected"], bundle["result"].weights,
        mode="custom", result=bundle["result"],
        portfolio_root=get_settings().strategy_root.parent / "portfolios")
    report = next(folder.glob("artifacts/*_report.html"), None)
    return {"saved": folder.name, "report": report.name if report else None}


def saved() -> list[str]:
    return list_saved(get_settings().strategy_root.parent / "portfolios")


# --------------------------------------------------------------------------- #
# Serialisation                                                               #
# --------------------------------------------------------------------------- #
def _equity_points(returns: pd.Series, n: int = 400) -> list[dict]:
    r = returns.dropna()
    if r.empty:
        return []
    eq = (1.0 + r).cumprod()
    if len(eq) > n:
        idx = np.linspace(0, len(eq) - 1, n).astype(int)
        eq = eq.iloc[idx]
    return [{"t": d.strftime("%Y-%m-%d"), "v": float(v)} for d, v in eq.items()]


def _df_rows(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records")) if df is not None and len(df) else []


def _corr(corr: pd.DataFrame) -> dict:
    if corr is None or corr.empty:
        return {"names": [], "matrix": []}
    return {"names": [str(c) for c in corr.columns], "matrix": clean(corr.to_numpy())}


def serialize(b: dict) -> dict:
    """Turn a ``build_portfolio`` bundle into JSON-safe primitives."""
    res = b["result"]
    port = res.portfolio_returns
    equity = {"portfolio": _equity_points(port)}
    bench = b.get("benchmark")
    if bench is not None and not bench.empty:
        window = bench.loc[(bench.index >= port.index.min()) & (bench.index <= port.index.max())]
        equity["benchmark"] = _equity_points(window)
    # Per-leg equity (over the portfolio's window) for the toggleable chart lines.
    legs = []
    for c in b.get("selected", []):
        r = c.returns
        r = r.loc[(r.index >= port.index.min()) & (r.index <= port.index.max())].dropna()
        if not r.empty:
            legs.append({"name": c.name, "points": _equity_points(r)})
    equity["strategies"] = legs
    out: dict[str, Any] = {
        "method": b["method"],
        "verdict": b["verdict"],
        "metrics": {k: res.metrics.get(k) for k in _METRIC_KEYS},
        "diversification": res.diversification.get("ratio", 1.0),
        "avg_active_share": res.avg_active_share,
        "pbo": b["pbo"],
        "permutation": b["permutation"],
        "gates": b["gates"],
        "equity": equity,
        "weights": [{"name": k, "weight": v}
                    for k, v in sorted(res.weights.items(), key=lambda kv: -kv[1])],
        "attribution": _df_rows(b["attribution"]),
        "components": _df_rows(b["components"]),
        "correlation": _corr(b["corr_matrix"]),
        "tail": b["tail"],
        "regime": b["regime"],
        "stress": b["stress"],
        "factor": b["factor"],
    }
    return clean(out)


def clean(obj: Any) -> Any:
    """Recursively coerce numpy / pandas scalars to JSON-safe values (NaN/Inf → None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
