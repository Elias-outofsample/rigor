"""Scenario-response analysis — how a strategy REACTS to market regimes.

Conditions strategy returns on explicit, interpretable regimes and reports, per
regime: annualised return, hit-rate, conditional beta and ``n_obs``. It answers
"how does it behave when the market trends up/down, when vol is high/low, when a
sector rotates" — and derives a behavioural verdict (defensive / long-vol vs
directional / needs-trend), going *beyond a single binary beta*.

  * ``scenario_response`` — the conditional response across all regimes + verdict.
  * ``market_drawdown_response`` — strategy response across the *distribution* of
    market drawdown episodes (every threshold, "every crash is different").
  * ``market_return_buckets`` — the conditional response *curve* per benchmark
    quantile bucket, revealing convexity/concavity a single beta hides.
  * ``realized_vol`` — a benchmark-derived vol proxy (always available, no VIX).
  * ``scenario_verdict`` — the behavioural label from a response dict.

Pairs with :mod:`rigor.analysis.regime` (regime labelling) and
:mod:`rigor.analysis.exposure`. Dependency-light: pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import TRADING_DAYS, periods_per_year_of

__all__ = [
    "scenario_response",
    "scenario_verdict",
    "market_drawdown_response",
    "market_return_buckets",
    "realized_vol",
]


def _ppy(returns: pd.Series) -> int:
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        return periods_per_year_of(returns.index)
    return TRADING_DAYS


def realized_vol(benchmark_returns: pd.Series, window: int | None = None,
                 annualise: bool = True) -> pd.Series:
    """Realized-volatility proxy from the benchmark.

    Always available across the full history (and for commodities/FX where VIX is
    irrelevant), so vol-regime analysis never depends on VIX coverage. ``window``
    defaults to a ~21-trading-day equivalent scaled to the series frequency.
    """
    b = pd.Series(benchmark_returns).astype("float64")
    ppy = _ppy(b)
    w = window or max(4, int(round(21 * ppy / 252)))
    vol = b.rolling(w, min_periods=max(3, w // 2)).std()
    return vol * np.sqrt(ppy) if annualise else vol


def market_drawdown_response(
    returns: pd.Series,
    benchmark_close: pd.Series,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict:
    """Strategy response across the DISTRIBUTION of market drawdown episodes.

    Instead of a single beta or a few named crashes (2008/COVID), this finds every
    period where the benchmark sits in a drawdown ≥ each threshold from its running
    peak, and reports the strategy's response — with dispersion — because "every
    crash is different". Per threshold bucket: number of drawdown bars, strategy
    mean/median/worst/best per-bar return, cumulative in-regime return, and hit
    rate. Buckets with < 5 bars report only ``n_days``.
    """
    r = pd.Series(returns).dropna().astype("float64")
    bc = pd.Series(benchmark_close).reindex(r.index).ffill()
    peak = bc.cummax()
    dd = bc / peak - 1.0  # <= 0
    out: dict = {}
    for th in thresholds:
        in_dd = dd <= -abs(th)
        sub = r[in_dd].dropna()
        n = int(len(sub))
        key = f"mkt_dd_ge_{int(abs(th) * 100)}pct"
        if n < 5:
            out[key] = {"n_days": n}
            continue
        out[key] = {
            "n_days": n,
            "strat_mean": float(sub.mean()),
            "strat_median": float(sub.median()),
            "strat_worst": float(sub.min()),
            "strat_best": float(sub.max()),
            "strat_cum_return_in_regime": float((1.0 + sub).prod() - 1.0),
            "strat_hit_rate": float((sub > 0).mean()),
        }
    return out


def market_return_buckets(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    n_buckets: int = 7,
) -> dict:
    """Conditional response CURVE: strategy mean return per benchmark-return bucket.

    Reveals NON-LINEARITY (convexity/concavity) concretely — e.g. positive in both
    tails = long-gamma-like — where a single beta hides it. Returns ``{}`` when
    there is too little data to form the buckets.
    """
    r = pd.Series(returns).dropna().astype("float64")
    b = pd.Series(benchmark_returns).reindex(r.index).astype("float64")
    df = pd.DataFrame({"r": r, "b": b}).dropna()
    if len(df) < n_buckets * 10:
        return {}
    try:
        df["bucket"] = pd.qcut(df["b"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return {}
    curve = {}
    for bk, grp in df.groupby("bucket"):
        curve[int(bk)] = {
            "bench_mean": float(grp["b"].mean()),
            "strat_mean": float(grp["r"].mean()),
            "n": int(len(grp)),
        }
    return {"buckets": curve, "n_buckets": len(curve)}


def _cond_stats(r: pd.Series, mask: pd.Series, ppy: int,
                bench: pd.Series | None = None) -> dict:
    sub = r[mask].dropna()
    n = int(len(sub))
    nan = float("nan")
    if n < 5:
        return {"n_obs": n, "ann_return": nan, "hit_rate": nan, "beta": nan}
    ann = float(sub.mean()) * ppy
    hit = float((sub > 0).mean())
    beta = nan
    if bench is not None:
        bsub = bench.reindex(sub.index).dropna()
        common = sub.index.intersection(bsub.index)
        if len(common) >= 5 and float(bsub.loc[common].var()) > 0:
            beta = float(np.cov(sub.loc[common], bsub.loc[common])[0, 1]
                         / bsub.loc[common].var())
    return {"n_obs": n, "ann_return": ann, "hit_rate": hit, "beta": beta}


def scenario_response(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    benchmark_close: pd.Series | None = None,
    vix: pd.Series | None = None,
    sector_returns: pd.DataFrame | None = None,
    trend_window: int = 200,
) -> dict:
    """Conditional response of ``returns`` across market regimes, plus a verdict.

    Regimes:
      * market direction — benchmark up vs down periods.
      * sustained trend — benchmark above/below its ``trend_window`` SMA (uses
        ``benchmark_close`` if given, else cumprods the benchmark returns).
      * volatility — VIX (or a benchmark rolling-vol proxy) high vs low (median).
      * sector rotation — per sector, periods where it outperforms the benchmark
        (only when ``sector_returns`` is provided).

    Also attaches the drawdown-episode distribution and the per-bucket response
    curve (concrete, generalisable replacements for an abstract beta), then a
    behavioural ``verdict``. Returns NaN-filled neutral stats for any regime with
    too few observations rather than raising.
    """
    r = pd.Series(returns).dropna().astype("float64")
    b = pd.Series(benchmark_returns).reindex(r.index).astype("float64")
    ppy = _ppy(r)
    out: dict = {"periods_per_year": ppy, "n_obs": int(len(r))}

    # 1) Market direction.
    out["market_up"] = _cond_stats(r, b > 0, ppy, b)
    out["market_down"] = _cond_stats(r, b < 0, ppy, b)

    # 2) Sustained trend — benchmark vs its SMA.
    if benchmark_close is not None:
        bc = pd.Series(benchmark_close).reindex(r.index).ffill()
    else:
        bc = (1.0 + b.fillna(0.0)).cumprod()
    sma = bc.rolling(trend_window, min_periods=max(20, trend_window // 4)).mean()
    out["trend_up"] = _cond_stats(r, bc > sma, ppy, b)
    out["trend_down"] = _cond_stats(r, bc <= sma, ppy, b)

    # 3) Volatility regime.
    if vix is not None:
        vol = pd.Series(vix).reindex(r.index).ffill()
    else:
        vol = b.rolling(max(4, int(round(21 * ppy / 252)))).std() * np.sqrt(ppy)
    vol_valid = vol.dropna()
    med = float(vol_valid.median()) if not vol_valid.empty else float("nan")
    out["vol_low"] = _cond_stats(r, vol <= med, ppy, b)
    out["vol_high"] = _cond_stats(r, vol > med, ppy, b)
    out["vol_proxy"] = "VIX" if vix is not None else "benchmark_rolling_std"

    # 4) Sector rotation (optional).
    if sector_returns is not None and not sector_returns.empty:
        sec: dict = {}
        for col in sector_returns.columns:
            s = pd.Series(sector_returns[col]).reindex(r.index).astype("float64")
            st = _cond_stats(r, s > b, ppy, b)  # sector beats the broad benchmark
            if st["n_obs"] >= 20:
                sec[str(col)] = st["ann_return"]
        if sec:
            ranked = sorted(sec.items(),
                            key=lambda kv: (kv[1] if kv[1] == kv[1] else -1e18),
                            reverse=True)
            out["sector_response"] = dict(ranked)
            out["best_sector"] = ranked[0][0]
            out["worst_sector"] = ranked[-1][0]

    # 5) Concrete, generalizable market-move response (replaces abstract beta):
    # the distribution of the response across ALL drawdown episodes of each
    # magnitude, plus the per-bucket response curve (nonlinearity a beta hides).
    out["drawdown_response"] = market_drawdown_response(r, bc)
    out["return_buckets"] = market_return_buckets(r, b)

    out["verdict"] = scenario_verdict(out)
    return out


def scenario_verdict(resp: dict) -> str:
    """Behavioural label from a :func:`scenario_response` dict (NaN → 0)."""
    def _ar(k: str) -> float:
        v = resp.get(k, {}).get("ann_return", float("nan"))
        return v if v == v else 0.0  # NaN -> 0

    up, down = _ar("market_up"), _ar("market_down")
    t_up, t_down = _ar("trend_up"), _ar("trend_down")
    v_low, v_high = _ar("vol_low"), _ar("vol_high")

    tags = []
    if down > up and down > 0:
        tags.append("defensive (outperforms in down markets)")
    elif up > down + abs(down) and t_up > t_down:
        tags.append("directional/pro-cyclical (needs up-trend)")
    if v_high > v_low and v_high > 0:
        tags.append("long-vol / crisis-alpha (better in high-vol)")
    elif v_low > v_high + abs(v_high):
        tags.append("short-vol / calm-market (decays in high-vol)")
    if t_up > t_down + abs(t_down):
        tags.append("trend-following (needs sustained up-trend)")
    elif t_down > t_up and t_down > 0:
        tags.append("counter-trend / mean-reverting")
    return "; ".join(tags) if tags else "regime-neutral"
