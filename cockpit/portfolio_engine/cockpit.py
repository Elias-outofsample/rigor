"""Headless portfolio-build pipeline — the single source for "build a book".

This is the headless orchestration behind the ★ Build action: it takes a set of
discovered strategies plus a weighting method and constraints, and runs the whole
dossier — allocate → backtest → validate out-of-sample → gate against the
constraints → analyse (attribution, components, correlation, tail dependence,
regimes, stress, factors). It returns a plain dict of results.

The FastAPI web cockpit (``cockpit_api/``) calls this one function, so the API and
any other consumer share a single implementation and can never disagree on what a
portfolio's numbers are. Every metric comes from the shared ``rigor`` engine /
``portfolio_engine`` primitives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rigor import metrics as _m

from . import analysis, stress
from .allocator import AllocationConfig, PortfolioAllocator
from .backtester import PortfolioBacktester
from .data import build_factor_proxies, load_benchmark_returns
from .portfolio_validation import permutation_test, portfolio_pbo
from .search import subset_grid_search, weight_grid_search

__all__ = ["build_portfolio", "build_from_weights", "search_portfolios",
           "eval_gates", "components", "tail_dependence"]


def eval_gates(res, attrib: pd.DataFrame, cons: dict, extra: dict) -> list[dict]:
    """Pass/fail each set constraint against the built book (strict).

    ``cons`` keys (any may be absent / 0 = "off"): ``min_sharpe``, ``min_cagr``
    (fraction), ``max_beta``, ``max_corr_bench``, ``max_corr_strat``,
    ``min_divers``, ``max_risk`` (fraction), ``max_dd`` (fraction),
    ``max_vol`` (fraction). Returns one ``{label, ok, actual}`` row per set gate.
    """
    m = res.metrics
    beta = res.beta.get("beta", float("nan")) if res.beta else float("nan")
    max_risk = float(attrib["risk_contrib_pct"].max()) if len(attrib) else 0.0
    div = res.diversification.get("ratio", float("nan"))
    gates: list[dict] = []

    def add(label: str, ok: bool, actual: str) -> None:
        gates.append({"label": label, "ok": bool(ok), "actual": actual})

    def isnum(x) -> bool:
        return x == x  # noqa: PLR0124 (NaN check)

    if cons.get("min_sharpe"):
        add(f"min Sharpe ≥ {cons['min_sharpe']:.2f}", m["sharpe"] >= cons["min_sharpe"],
            f"{m['sharpe']:.2f}")
    if cons.get("min_cagr"):
        add(f"min CAGR ≥ {cons['min_cagr']:.0%}", m["cagr"] >= cons["min_cagr"],
            f"{m['cagr']:.1%}")
    if cons.get("max_beta") and isnum(beta):
        add(f"max β ≤ {cons['max_beta']:.2f}", abs(beta) <= cons["max_beta"], f"{beta:.2f}")
    if cons.get("max_corr_bench") and isnum(extra.get("corr_bench", float("nan"))):
        cb = extra["corr_bench"]
        add(f"max ρ bench ≤ {cons['max_corr_bench']:.2f}", abs(cb) <= cons["max_corr_bench"],
            f"{cb:.2f}")
    if cons.get("max_corr_strat"):
        mc = extra.get("max_pair_corr", 0.0)
        add(f"max ρ strats ≤ {cons['max_corr_strat']:.2f}", mc <= cons["max_corr_strat"],
            f"{mc:.2f}")
    if cons.get("min_divers") and isnum(div):
        add(f"min divers. ≥ {cons['min_divers']:.2f}", div >= cons["min_divers"], f"{div:.2f}")
    if cons.get("max_risk"):
        add(f"max risk/strat ≤ {cons['max_risk']:.0%}", max_risk <= cons["max_risk"],
            f"{max_risk:.0%}")
    if cons.get("max_dd"):
        add(f"max |DD| ≤ {cons['max_dd']:.0%}", abs(m["max_drawdown"]) <= cons["max_dd"],
            f"{m['max_drawdown']:.1%}")
    if cons.get("max_vol"):
        add(f"max vol ≤ {cons['max_vol']:.0%}", m["volatility"] <= cons["max_vol"],
            f"{m['volatility']:.1%}")
    return gates


def components(common: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Per-strategy stats on the common (all-overlapping) window."""
    cols = ["name", "weight", "cagr", "sharpe", "vol", "max_dd"]
    if common.empty:
        return pd.DataFrame(columns=cols)
    ppy = _m.periods_per_year_of(common.index)
    rows = []
    for name in common.columns:
        arr = common[name].to_numpy(dtype="float64")
        rows.append({"name": name, "weight": float(weights.get(name, 0.0)),
                     "cagr": float(_m.compute_cagr(arr, ppy)),
                     "sharpe": float(_m.compute_sharpe(arr, ppy)),
                     "vol": float(_m.compute_volatility(arr, ppy)),
                     "max_dd": float(_m.compute_max_drawdown(arr))})
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)


def tail_dependence(port: pd.Series, bench, res) -> dict:
    """Equity beta + overall vs worst-decile (tail) correlation to the benchmark."""
    if bench is None:
        return {}
    a = pd.concat({"p": port, "b": bench}, axis=1).dropna()
    if len(a) < 30:
        return {}
    overall = float(a["p"].corr(a["b"]))
    thresh = a["b"].quantile(0.10)
    tail = a[a["b"] <= thresh]
    tail_corr = float(tail["p"].corr(tail["b"])) if len(tail) > 5 else float("nan")
    beta = res.beta.get("beta", float("nan")) if res.beta else float("nan")
    return {"beta": beta, "overall_corr": overall, "tail_corr": tail_corr}


def build_portfolio(selected, method: str, max_weight: float | None, leverage_cap: float,
                    use_cpcv: bool, cons: dict, *, common_window: bool = False) -> dict:
    """Allocate → backtest → validate → gate → analyse. Returns the result bundle.

    Parameters
    ----------
    selected:
        List of ``StrategyCandidate`` (≥ 2) to combine.
    method:
        ``"robust (grid)"`` (composite grid search) or any allocation mode in
        ``portfolio_engine.weight_methods.METHODS`` plus ``kelly`` /
        ``regime_dependent``.
    max_weight:
        Per-strategy weight cap as a fraction (``None`` = uncapped).
    leverage_cap:
        Gross-exposure cap (≥ 1.0).
    use_cpcv:
        Use combinatorial purged CV for the grid search's OOS estimate.
    cons:
        Constraint dict (see :func:`eval_gates`).

    Returns
    -------
    dict with keys: ``result`` (BacktestResult), ``verdict``, ``benchmark``,
    ``selected``, ``method``, ``attribution``/``components``/``corr_matrix``
    (DataFrames), ``tail``, ``regime``, ``stress``, ``pbo``, ``permutation``,
    ``factor``, ``gates``, ``extra``.
    """
    if len(selected) < 2:
        raise ValueError("select at least two strategies")

    if method == "robust (grid)":
        rm_full = pd.concat({c.name: c.returns for c in selected}, axis=1)
        df = weight_grid_search(rm_full, n_samples=1500, top_k=1, max_weight=max_weight,
                                use_cpcv=use_cpcv)
        if df.empty:
            raise ValueError("grid search found no viable allocation")
        wmap = df.iloc[0]["weights"]
        weights = np.array([float(wmap.get(c.name, 0.0)) for c in selected])
        if weights.sum() <= 0:
            weights = np.ones(len(selected)) / len(selected)
    else:
        cfg = AllocationConfig(mode=method, max_weight=max_weight, leverage_cap=leverage_cap)
        weights = np.asarray(PortfolioAllocator(selected, cfg).allocate(), dtype=float)

    return _dossier(selected, weights, cons, method, common_window=common_window)


def build_from_weights(selected, weights_by_name: dict, cons: dict, *,
                       common_window: bool = False) -> dict:
    """Build the full dossier for an *explicit* weight vector (a chosen candidate
    from :func:`search_portfolios`). Weights are keyed by ``StrategyCandidate.name``;
    missing legs default to 0, and an all-zero vector falls back to equal weight.
    """
    if len(selected) < 2:
        raise ValueError("select at least two strategies")
    weights = np.array([float(weights_by_name.get(c.name, 0.0)) for c in selected])
    if weights.sum() <= 0:
        weights = np.ones(len(selected)) / len(selected)
    return _dossier(selected, weights, cons, "custom", common_window=common_window)


def search_portfolios(selected, cons: dict, *, max_weight: float | None = None,
                      n_samples: int = 4000, use_cpcv: bool = False, top_k: int = 120,
                      min_strats: int = 0, max_strats: int = 0,
                      common_window: bool = False) -> list[dict]:
    """Grid-search the allocation simplex and return **every** candidate that clears
    the criteria, ranked by the composite anti-overfit score.

    Criteria applied here are the cheap, grid-computed ones — ``min_sharpe`` (full
    Sharpe), ``min_cagr`` and ``max_dd`` from ``cons``, plus a **cardinality** band
    (``min_strats`` / ``max_strats`` legs carrying weight; ``0`` = unbounded). The
    full strict gates (β / ρ / diversification / risk-per-strat / vol) are evaluated
    when a candidate is opened via :func:`build_from_weights`. Each candidate dict
    carries its weight vector (keyed by strategy name) plus headline metrics.
    """
    if len(selected) < 2:
        raise ValueError("select at least two strategies")
    rm = pd.concat({c.name: c.returns for c in selected}, axis=1)
    # Focused subset search — find the best *k-of-N* combination, not a closet index.
    # Default to a 1..10-leg band when no cardinality is given, so a whole-book search
    # still returns investable books (the old dense search produced ~N-leg portfolios).
    lo = min_strats if min_strats else 1
    hi = max_strats if max_strats else min(10, len(selected))
    df = subset_grid_search(rm, min_k=lo, max_k=hi, n_samples=n_samples, top_k=top_k,
                            max_weight=max_weight, use_cpcv=use_cpcv,
                            common_window=common_window)
    min_sharpe = cons.get("min_sharpe") or 0.0
    min_cagr = cons.get("min_cagr") or 0.0
    max_dd = cons.get("max_dd") or 0.0
    out = []
    for rank, (_, row) in enumerate(df.iterrows()):
        if row["sharpe_full"] < min_sharpe or row["cagr"] < min_cagr:
            continue
        if max_dd and abs(row["max_drawdown"]) > max_dd:
            continue
        n_strats = int(sum(1 for w in row["weights"].values() if w > 1e-6))
        if (min_strats and n_strats < min_strats) or (max_strats and n_strats > max_strats):
            continue
        out.append({
            "rank": rank, "weights": dict(row["weights"]),
            "sharpe": float(row["sharpe_full"]), "sharpe_oos": float(row["sharpe_oos_mean"]),
            "cagr": float(row["cagr"]), "max_dd": float(row["max_drawdown"]),
            "dsr": float(row["dsr"]), "composite": float(row["composite_score"]),
            "n_strats": n_strats, "n_obs": int(row.get("n_obs", 0)),
        })
    return out


def _dossier(selected, weights: np.ndarray, cons: dict, method: str, *,
             common_window: bool = False) -> dict:
    """Backtest a weight vector and assemble the full result bundle.

    ``common_window=True`` restricts the book to its *live* legs and backtests on the
    bars where they all overlap (inner-join), so the headline metrics/equity reflect
    only the period every strategy was running — not the early stretch where younger
    legs hadn't started.
    """
    # A portfolio IS its weight-bearing legs — drop zero-weight strategies so the
    # whole dossier (weights, attribution, components, correlation, equity) is about
    # the actual book, not every strategy that was on the selection list.
    weights = np.asarray(weights, dtype=float)
    keep = np.flatnonzero(np.abs(weights) > 1e-9)
    if 1 <= keep.size < len(weights):
        selected = [selected[i] for i in keep]
        weights = weights[keep]
    total = weights.sum()
    if total > 0:
        weights = weights / total

    bench = load_benchmark_returns("SPY")
    align = "none" if common_window else "native"
    res = PortfolioBacktester(selected, weights, benchmark_returns=bench,
                              align_frequency=align).run()
    port = res.portfolio_returns

    from rigor.engine import result_from_returns
    from rigor.validation import validate
    verdict = validate(result_from_returns(port))["verdict"]

    rm = pd.concat({c.name: c.returns for c in selected}, axis=1)
    if common_window:
        rm = rm.dropna(how="any")
    common = rm.dropna(how="any")
    w_vec = [res.weights.get(c.name, 0.0) for c in selected]
    attrib = analysis.attribution(rm, w_vec)
    regimes = analysis.market_regimes(port, bench)

    corr_matrix = common.corr() if common.shape[1] >= 2 else pd.DataFrame()
    max_pair_corr = 0.0
    if not corr_matrix.empty:
        iu = np.triu_indices_from(corr_matrix.to_numpy(), k=1)
        max_pair_corr = (float(np.nanmax(np.abs(corr_matrix.to_numpy()[iu])))
                         if iu[0].size else 0.0)
    corr_bench = float("nan")
    if bench is not None:
        paired = pd.concat({"p": port, "b": bench}, axis=1).dropna()
        if len(paired) >= 2:
            corr_bench = float(paired.corr().iloc[0, 1])
    extra = {"corr_bench": corr_bench, "max_pair_corr": max_pair_corr}

    return {
        "result": res, "verdict": verdict, "benchmark": bench, "selected": list(selected),
        "method": method, "attribution": attrib,
        "components": components(common, res.weights),
        "corr_matrix": corr_matrix,
        "tail": tail_dependence(port, bench, res),
        "regime": analysis.regime_performance(port, regimes),
        "stress": stress.run_all_scenarios(port),
        "pbo": portfolio_pbo(rm),
        "permutation": permutation_test(rm, w_vec, n_perm=1000),
        "factor": (analysis.factor_regression(port, build_factor_proxies())
                   if bench is not None else {"note": "factor data unavailable offline"}),
        "gates": eval_gates(res, attrib, cons, extra),
        "extra": extra,
    }
