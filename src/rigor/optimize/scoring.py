"""Five-pillar composite selection score (V4.2) — ported from the research library.

An alternative to ranking by raw Sharpe: score every config on five normalised
0-10 pillars and take their weighted geometric mean, so a config has to be good
on *several* axes (risk-adjusted return, drawdown, trend stability, consistency,
tail) rather than spiking on one. Reuses the robust pillars when the enrichment
metrics are available (PSR / UPI / rolling-consistency / Sortino), else the
classic ones (Sharpe / Calmar / ROC310 / win-rate); stability is common to both.

This does *not* replace Rigor's verdict — it is reported alongside it. The weighted
geomean up-weights the risk-adjusted pillar (x3) and lightly favours absolute
CAGR (x1.5) and activity (x1.2) to break near-ties, matching the source's
calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m
from .enrichment import enrich_equity
from .numba_accel import weighted_geomean

__all__ = ["score_configs", "RANGES"]

# (lo, hi) calibration ranges -> each metric clipped then rescaled to [0, 10].
RANGES = {
    "sharpe": (-0.5, 2.5), "calmar": (0.0, 3.0), "stability": (0.0, 1.0),
    "roc310": (0.0, 10.0), "winrate": (0.3, 0.7), "psr": (0.5, 0.99),
    "upi": (0.0, 5.0), "rolling_consistency": (0.0, 1.0), "sortino": (-1.0, 5.0),
}
_W_RISK, _W_CAGR, _W_TRADES = 3.0, 1.5, 1.2  # calibration weights


def _norm(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo:
        return pd.Series(5.0, index=s.index)
    return (s.clip(lo, hi) - lo) / (hi - lo) * 10.0


def _norm_nan(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return _norm(s.fillna(0.5 * (lo + hi)) if s.hasnans else s, lo, hi)


def _score_trades(n: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (np.log10(1.0 + n) / np.log10(1.0 + 10_000.0) * 10.0).clip(0.0, 10.0)


def _config_metrics(returns: pd.Series, ppy: int) -> dict:
    r = returns.dropna()
    arr = r.to_numpy(dtype="float64")
    n = len(arr)
    if n < 2:
        return {}
    from scipy.stats import kurtosis, skew
    m = {
        "sharpe": _m.compute_sharpe(arr, ppy), "cagr": _m.compute_cagr(arr, ppy),
        "max_dd": _m.compute_max_drawdown(arr), "win_rate": _m.compute_win_rate(arr),
        "trades": int(np.count_nonzero(arr)),
        "skewness": float(skew(arr, bias=False)) if n > 3 else 0.0,
        "kurtosis": float(kurtosis(arr, fisher=True, bias=False)) if n > 3 else 0.0,
    }
    m.update(enrich_equity(r, ppy))
    return m


def score_configs(rows: list[dict], *, cpcv: dict | None = None) -> dict:
    """Compute the 5-pillar composite score for every config; return the ranking
    and the winner. ``rows`` is the ``{"params", "returns"}`` list."""
    ppy = _m.periods_per_year_of(rows[0]["returns"].dropna().index)
    recs = []
    for row in rows:
        m = _config_metrics(row["returns"], ppy)
        if m:
            m["params"] = row["params"]
            recs.append(m)
    if not recs:
        return {"note": "no scorable configs"}
    df = pd.DataFrame(recs)

    df["calmar"] = df.apply(lambda r: r["cagr"] / abs(r["max_dd"]) if r["max_dd"] else 0.0, axis=1)
    df["score_sharpe"] = _norm(df["sharpe"], *RANGES["sharpe"])
    df["score_calmar"] = _norm(df["calmar"], *RANGES["calmar"])
    df["score_stability"] = _norm_nan(df["stability"], *RANGES["stability"])
    df["score_roc310"] = _norm_nan(df["roc310"], *RANGES["roc310"])
    df["score_winrate"] = _norm_nan(df["win_rate"], *RANGES["winrate"])
    df["score_psr"] = _norm_nan(df["psr"], *RANGES["psr"])
    df["score_upi"] = _norm_nan(df["upi"], *RANGES["upi"])
    df["score_rolling_consistency"] = _norm_nan(
        df["rolling_consistency"], *RANGES["rolling_consistency"])
    df["score_sortino"] = _norm_nan(df["sortino"], *RANGES["sortino"])
    df["score_trades"] = _score_trades(df["trades"])
    cagr_num = pd.to_numeric(df["cagr"], errors="coerce").fillna(0.0)
    recency_num = pd.to_numeric(df["recency"], errors="coerce").fillna(0.0)
    df["score_cagr_abs"] = (cagr_num / 0.20 * 10.0).clip(0.0, 10.0)
    df["score_recency"] = ((recency_num + 1.0) / 3.0 * 10.0).clip(0.0, 10.0)
    calmar_raw = df["calmar"].clip(lower=0.0)
    df["score_calmar_robust"] = (np.sqrt(calmar_raw.clip(0.0, 2.0) / 2.0) * 10.0).clip(0.0, 10.0)

    # Robust pillars swap in for the classic ones when present (they always are here).
    has = {p: df[f"score_{p}"].notna().any()
           for p in ("psr", "upi", "rolling_consistency", "sortino")}
    pillar_risk = "score_psr" if has["psr"] else "score_sharpe"
    score_cols = [
        pillar_risk,
        "score_upi" if has["upi"] else "score_calmar",
        "score_stability",
        "score_rolling_consistency" if has["rolling_consistency"] else "score_roc310",
        "score_sortino" if has["sortino"] else "score_winrate",
        "score_calmar_robust", "score_trades", "score_cagr_abs", "score_recency",
    ]
    # Grid-level rank-stability pillar (constant across rows) from CPCV Kendall tau.
    if cpcv and cpcv.get("rank_correlation") is not None:
        rho = float(cpcv["rank_correlation"])
        df["score_rank_stability"] = np.clip((rho + 1.0) / 2.0 * 10.0, 0.0, 10.0)
        score_cols.append("score_rank_stability")

    eps = 0.1
    pillars = df[score_cols].to_numpy(dtype="float64") + eps
    np.maximum(pillars, eps, out=pillars)
    weights = np.ones(len(score_cols))
    weights[score_cols.index(pillar_risk)] = _W_RISK
    weights[score_cols.index("score_cagr_abs")] = _W_CAGR
    weights[score_cols.index("score_trades")] = _W_TRADES
    # Per-config weighted geomean — the optimiser's composite-score hot path.
    # Optional Numba CPU acceleration (opt-in via RIGOR_OPTIMIZE_ACCEL); the default
    # is the byte-identical pure-NumPy fallback, so CI (no numba) is unchanged.
    df["composite_score"] = weighted_geomean(pillars, weights)

    df = df.sort_values(["composite_score", "sharpe", "cagr"],
                        ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    return {
        "best_params": best["params"],
        "best_composite": round(float(best["composite_score"]), 4),
        "best_sharpe": round(float(best["sharpe"]), 4),
        "pillars_used": score_cols,
        "winner_pillars": {c: round(float(best[c]), 2) for c in score_cols},
        "ranking": [{"params": r["params"], "composite": round(float(r["composite_score"]), 4),
                     "sharpe": round(float(r["sharpe"]), 4)} for _, r in df.head(25).iterrows()],
    }
