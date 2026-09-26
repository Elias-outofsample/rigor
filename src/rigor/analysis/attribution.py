"""Signal-vs-ranker attribution & Sharpe-difference tests (#11).

Answers "where does the alpha come from?" — the *entry signal* (e.g. ``IBS<0.2``)
or the *ranker / SetupScore* that selects and orders candidates (e.g. rank by
NATR)? A "mean-reversion" strategy whose return is really driven by the NATR
ranker is a volatility strategy in disguise.

  * ``decompose_signal_vs_setup`` — Brinson-Hood-Beebower contribution split
    over four parallel backtests (signal-only / setup-only / combined / baseline).
  * ``test_signal_orthogonality`` — Spearman ρ between signal and setup scores
    (high |ρ| ⇒ the two components are redundant).
  * ``attribution_sharpe_breakdown`` — Sharpe-level shares + Memmel pairwise tests.
  * ``memmel_sharpe_difference_test`` — Memmel (2003) z-test for two correlated
    Sharpe ratios.
  * ``conditional_signal_test`` — paired t-test: does the setup add daily return?
  * ``brinson_fachler`` — classic sector × period allocation/selection/interaction.

scipy only; Sharpe reuses ``rigor.metrics``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .. import metrics as _m

__all__ = [
    "decompose_signal_vs_setup", "test_signal_orthogonality",
    "attribution_sharpe_breakdown", "memmel_sharpe_difference_test",
    "conditional_signal_test", "brinson_fachler",
]


def _to_array(x) -> np.ndarray:
    return np.asarray(x.to_numpy() if isinstance(x, pd.Series) else x, dtype="float64")


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and np.isfinite(b) else default


def _sharpe(r: np.ndarray, ppy: int) -> float:
    return float(_m.compute_sharpe(r, ppy)) if len(r) >= 2 else 0.0


def decompose_signal_vs_setup(returns_signal_only, returns_setup_only, returns_combined,
                              returns_baseline=None, ppy: int = _m.TRADING_DAYS) -> dict:
    """Brinson-Hood-Beebower (1986) decomposition adapted to a single strategy.

    Given four parallel backtests (signal-only, setup-only, combined, baseline)::

        R_combined = R_baseline + (R_signal-R_base) + (R_setup-R_base) + interaction
        interaction = R_combined - R_signal - R_setup + R_baseline
    """
    rs = _to_array(returns_signal_only)
    ru = _to_array(returns_setup_only)
    rc = _to_array(returns_combined)
    rb = _to_array(returns_baseline) if returns_baseline is not None else np.zeros_like(rc)
    n = min(len(rs), len(ru), len(rc), len(rb))
    rs, ru, rc, rb = rs[:n], ru[:n], rc[:n], rb[:n]

    mean_s, mean_u, mean_c, mean_b = (
        float(np.mean(x)) if len(x) else 0.0 for x in (rs, ru, rc, rb))
    contrib_signal = mean_s - mean_b
    contrib_setup = mean_u - mean_b
    interaction = mean_c - mean_s - mean_u + mean_b
    total_excess = mean_c - mean_b
    if abs(total_excess) > 1e-12:
        share_signal, share_setup, share_inter = (contrib_signal / total_excess,
                                                   contrib_setup / total_excess,
                                                   interaction / total_excess)
    else:
        share_signal = share_setup = share_inter = 0.0

    if abs(share_signal) >= max(abs(share_setup), abs(share_inter)):
        dominant = "SIGNAL"
    elif abs(share_setup) >= abs(share_inter):
        dominant = "SETUP_SCORE"
    else:
        dominant = "INTERACTION"

    return {
        "sharpe_combined": _sharpe(rc, ppy), "sharpe_signal_only": _sharpe(rs, ppy),
        "sharpe_setup_only": _sharpe(ru, ppy), "sharpe_baseline": _sharpe(rb, ppy),
        "mean_return_combined": mean_c,
        "contribution": {"signal": float(contrib_signal), "setup": float(contrib_setup),
                         "interaction": float(interaction),
                         "total_excess_vs_baseline": float(total_excess)},
        "share_of_excess": {"signal": float(share_signal), "setup": float(share_setup),
                            "interaction": float(share_inter)},
        "dominant_source": dominant,
    }


def test_signal_orthogonality(signal_score, setup_score) -> dict:
    """Spearman rank correlation between signal and setup scores; |ρ|>0.5 ⇒ redundant."""
    s, u = _to_array(signal_score), _to_array(setup_score)
    n = min(len(s), len(u))
    s, u = s[:n], u[:n]
    mask = np.isfinite(s) & np.isfinite(u)
    s, u = s[mask], u[mask]
    if len(s) < 20:
        return {"rho": 0.0, "p_value": 1.0, "redundant": False, "n_obs": int(len(s))}
    rho, p = stats.spearmanr(s, u)
    rho = float(rho) if np.isfinite(rho) else 0.0
    verdict = ("ORTHOGONAL" if abs(rho) < 0.1 else "WEAKLY_CORRELATED" if abs(rho) < 0.3
               else "MODERATELY_CORRELATED" if abs(rho) < 0.5 else "REDUNDANT")
    return {"rho": rho, "p_value": float(p) if np.isfinite(p) else 1.0,
            "redundant": abs(rho) > 0.5, "verdict": verdict, "n_obs": int(len(s))}


# Pytest-discovery suppression — `test_` prefix kept for API consistency.
test_signal_orthogonality.__test__ = False  # type: ignore[attr-defined]


def memmel_sharpe_difference_test(returns_a, returns_b, ppy: int = _m.TRADING_DAYS) -> dict:
    """Memmel (2003) z-test for the difference between two *correlated* Sharpe ratios.

    ``z_stat`` / ``p_value`` are computed on the **per-period** Sharpe difference (the
    correct scale for the test); ``sharpe_a`` / ``sharpe_b`` / ``sharpe_diff`` are
    reported **annualised** (×√ppy) for readability. Don't divide ``sharpe_diff`` by
    ``z_stat`` — they're in different scales.
    """
    a, b = _to_array(returns_a), _to_array(returns_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    bad = {"sharpe_diff": 0.0, "z_stat": 0.0, "p_value": 1.0, "is_different": False}
    if n < 30:
        return bad
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    if var_a <= 0 or var_b <= 0:
        return bad
    sr_a, sr_b = mu_a / np.sqrt(var_a), mu_b / np.sqrt(var_b)
    rho = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    var_diff = (2.0 - 2.0 * rho
                + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * rho**2)) / n
    se = np.sqrt(max(var_diff, 0.0))
    diff = sr_a - sr_b
    z = _safe_div(diff, se)
    p = 2.0 * float(stats.norm.sf(abs(z)))
    scale = np.sqrt(ppy)
    return {"sharpe_a": float(sr_a * scale), "sharpe_b": float(sr_b * scale),
            "sharpe_diff": float(diff * scale), "z_stat": z, "p_value": p,
            "is_different": p < 0.05, "correlation": rho}


def attribution_sharpe_breakdown(returns_signal_only, returns_setup_only,
                                 returns_combined, ppy: int = _m.TRADING_DAYS) -> dict:
    """Sharpe-level attribution: each pillar's share of combined Sharpe + Memmel tests."""
    rs = _to_array(returns_signal_only)
    ru = _to_array(returns_setup_only)
    rc = _to_array(returns_combined)
    n = min(len(rs), len(ru), len(rc))
    rs, ru, rc = rs[:n], ru[:n], rc[:n]
    sc, ss, su = _sharpe(rc, ppy), _sharpe(rs, ppy), _sharpe(ru, ppy)
    mem_cs = memmel_sharpe_difference_test(rc, rs, ppy=ppy)
    mem_cu = memmel_sharpe_difference_test(rc, ru, ppy=ppy)
    return {
        "sharpe_combined": sc, "sharpe_signal_only": ss, "sharpe_setup_only": su,
        "share_signal": _safe_div(ss, sc), "share_setup": _safe_div(su, sc),
        "memmel_combined_vs_signal": mem_cs, "memmel_combined_vs_setup": mem_cu,
        "setup_adds_value": mem_cs["p_value"] < 0.05 and mem_cs["sharpe_diff"] > 0,
        "signal_adds_value": mem_cu["p_value"] < 0.05 and mem_cu["sharpe_diff"] > 0,
    }


def conditional_signal_test(returns_signal_only, returns_combined,
                            ppy: int = _m.TRADING_DAYS) -> dict:
    """Paired one-sided t-test: does the combined strategy beat signal-only daily?"""
    rs, rc = _to_array(returns_signal_only), _to_array(returns_combined)
    n = min(len(rs), len(rc))
    rs, rc = rs[:n], rc[:n]
    if n < 30:
        return {"diff_mean": 0.0, "t_stat": 0.0, "p_value": 1.0, "setup_adds_value": False}
    diff = rc - rs
    t_stat, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
    return {"diff_mean": float(diff.mean()), "diff_mean_annualised": float(diff.mean() * ppy),
            "t_stat": float(t_stat), "p_value_two_sided": float(p_two),
            "p_value_one_sided": p_one,
            "setup_adds_value": p_one < 0.05 and diff.mean() > 0, "n_obs": n}


def brinson_fachler(strategy_weights: pd.DataFrame, strategy_returns: pd.DataFrame,
                    benchmark_weights: pd.DataFrame, benchmark_returns: pd.DataFrame,
                    periods: Sequence | pd.DatetimeIndex | None = None) -> dict:
    """Brinson-Fachler attribution by sector and period.

    All four DataFrames share one column per sector, one row per period. Per period::

        allocation_k  = (w_k^p - w_k^b) * (r_k^b - R_b)
        selection_k   =  w_k^b         * (r_k^p - r_k^b)
        interaction_k = (w_k^p - w_k^b) * (r_k^p - r_k^b)
    """
    idx = strategy_weights.index if periods is None else pd.Index(periods)
    cols = strategy_weights.columns
    sw = strategy_weights.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64)
    sr = strategy_returns.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64)
    bw = benchmark_weights.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64)
    br = benchmark_returns.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64)
    n_periods = len(idx)

    port_total = (sw.values * sr.values).sum(axis=1)
    bench_total = (bw.values * br.values).sum(axis=1)

    alloc = (sw.values - bw.values) * (br.values - bench_total[:, None])
    selec = bw.values * (sr.values - br.values)
    inter = (sw.values - bw.values) * (sr.values - br.values)

    by_sector = {}
    for k, sector in enumerate(cols):
        a, s, i = float(alloc[:, k].mean()), float(selec[:, k].mean()), float(inter[:, k].mean())
        by_sector[sector] = {"allocation": a, "selection": s, "interaction": i, "total": a + s + i}

    by_period = [{"period": str(p), "allocation": float(alloc[t].sum()),
                  "selection": float(selec[t].sum()), "interaction": float(inter[t].sum()),
                  "total": float(alloc[t].sum() + selec[t].sum() + inter[t].sum()),
                  "excess_return": float(port_total[t] - bench_total[t])}
                 for t, p in enumerate(idx)]

    ta = float(alloc.sum(axis=1).mean())
    ts = float(selec.sum(axis=1).mean())
    ti = float(inter.sum(axis=1).mean())
    te = float((port_total - bench_total).mean())
    effects = {"allocation": abs(ta), "selection": abs(ts), "interaction": abs(ti)}
    dominant = max(effects, key=lambda k: effects[k])
    return {
        "by_sector": by_sector, "by_period": by_period,
        "total": {"allocation": ta, "selection": ts, "interaction": ti,
                  "excess_return": te, "attributed_excess": ta + ts + ti,
                  "residual": te - (ta + ts + ti)},
        "dominant_effect": dominant, "sectors": list(cols), "n_periods": n_periods,
        "summary": (f"Brinson-Fachler: excess={te:.4%} | alloc={ta:.4%} | "
                    f"selec={ts:.4%} | inter={ti:.4%} | dominant={dominant.upper()}"),
    }
