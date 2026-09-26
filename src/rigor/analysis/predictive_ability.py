"""Predictive-ability tests (#7, part).

Distinct from our White-RC/Hansen-SPA/StepM (which compare a pool's best vs a
benchmark): these test forecast-loss superiority and forecast skill.

  * ``diebold_mariano`` — equal predictive accuracy (Harvey-Leybourne-Newbold
    small-sample correction).
  * ``giacomini_white`` — *conditional* predictive ability given instruments.
  * ``model_confidence_set`` — Hansen-Lunde-Nason: the smallest set containing the
    best model at confidence 1−α (stationary-bootstrap range statistic).
  * ``pesaran_timmermann`` — non-parametric directional-accuracy test.
  * ``stochastic_dominance`` — Davidson-Duclos FSD/SSD/TSD vs a benchmark.

scipy only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "diebold_mariano", "giacomini_white", "model_confidence_set",
    "pesaran_timmermann", "stochastic_dominance",
]


def _a(x) -> np.ndarray:
    return np.asarray(x.to_numpy() if isinstance(x, pd.Series) else x, dtype="float64")


def _hac_var(x: np.ndarray, lag: int | None = None) -> float:
    n = len(x)
    if n < 3:
        return float(np.var(x, ddof=1)) if n > 1 else 0.0
    lag = lag or max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    xc = x - x.mean()
    s = float(xc @ xc / n)
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float(xc[k:] @ xc[:-k] / n)
    return max(s, 1e-12)


def diebold_mariano(loss_a, loss_b, *, h: int = 1, small_sample: bool = True) -> dict:
    """Diebold-Mariano equal-predictive-accuracy test on two loss series."""
    a, b = _a(loss_a), _a(loss_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 10:
        return {"dm_stat": 0.0, "p_value": 1.0, "is_different": False}
    d = a - b
    se = np.sqrt(_hac_var(d, lag=max(h - 1, 1)) / n)
    if se <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "is_different": False}
    dm = float(d.mean() / se)
    if small_sample:
        dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
        return {"dm_stat": dm, "p_value": p, "n_obs": n, "is_different": p < 0.05,
                "correction": "HLN-1997"}
    p = 2.0 * float(stats.norm.sf(abs(dm)))
    return {"dm_stat": dm, "p_value": p, "n_obs": n, "is_different": p < 0.05,
            "correction": "asymptotic"}


def giacomini_white(loss_a, loss_b, instruments) -> dict:
    """Giacomini-White conditional predictive ability (Wald χ² on h·(L_a−L_b))."""
    a, b = _a(loss_a), _a(loss_b)
    h = (instruments.to_numpy() if isinstance(instruments, pd.DataFrame)
         else np.asarray(instruments, dtype="float64"))
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    n = min(len(a), len(b), len(h))
    a, b, h = a[:n], b[:n], h[:n]
    m = np.isfinite(a) & np.isfinite(b) & np.all(np.isfinite(h), axis=1)
    a, b, h = a[m], b[m], h[m]
    n, q = len(a), h.shape[1]
    if n < 5 * q:
        return {"gw_stat": 0.0, "p_value": 1.0, "is_different": False}
    z = h * (a - b)[:, None]
    zbar = z.mean(axis=0)
    omega = np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
    try:
        oi = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        oi = np.linalg.pinv(omega)
    stat = float(n * zbar @ oi @ zbar)
    p = float(stats.chi2.sf(stat, df=q))
    return {"gw_stat": stat, "p_value": p, "n_obs": n, "n_instruments": q,
            "is_different": p < 0.05}


def _stationary_boot_idx(n_boot: int, T: int, block: int, rng) -> np.ndarray:
    p = 1.0 / max(block, 1)
    jump = rng.random((n_boot, T)) < p
    starts = rng.integers(0, T, size=(n_boot, T), dtype="int64")
    out = np.empty((n_boot, T), dtype="int64")
    for b in range(n_boot):
        idx = np.empty(T, dtype="int64")
        idx[0] = starts[b, 0]
        for t in range(1, T):
            idx[t] = starts[b, t] if jump[b, t] else (idx[t - 1] + 1) % T
        out[b] = idx
    return out


def model_confidence_set(losses, *, alpha: float = 0.10, n_bootstrap: int = 500,
                         block_size: int = 10, seed: int = 42) -> dict:
    """Hansen-Lunde-Nason Model Confidence Set: the models you can't statistically
    distinguish from the best, at confidence 1−α."""
    L = (losses.to_numpy() if isinstance(losses, pd.DataFrame)
         else np.asarray(losses, dtype="float64"))
    if L.ndim != 2:
        raise ValueError("losses must be 2-D (T, K)")
    T, K = L.shape
    if K < 2 or T < 30:
        return {"survivors": list(range(K)), "p_values": {}, "alpha": alpha}
    rng = np.random.default_rng(seed)
    bidx = _stationary_boot_idx(n_bootstrap, T, block_size, rng)
    survivors, p_values = list(range(K)), {}
    while len(survivors) > 1:
        sub = L[:, survivors]
        dmean = sub.mean(axis=0)
        bmeans = np.stack([sub[bidx[b]].mean(axis=0) for b in range(n_bootstrap)])
        k = len(survivors)
        var_d = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    var_d[i, j] = float(np.var(bmeans[:, i] - bmeans[:, j]
                                               - (dmean[i] - dmean[j]), ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            tobs = float(np.max(np.abs(np.where(var_d > 0,
                         (dmean[:, None] - dmean[None, :]) / np.sqrt(var_d / T + 1e-30), 0.0))))
        bootT = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            bm = bmeans[b] - dmean
            with np.errstate(divide="ignore", invalid="ignore"):
                bootT[b] = float(np.max(np.abs(np.where(var_d > 0,
                                 (bm[:, None] - bm[None, :]) / np.sqrt(var_d / T + 1e-30), 0.0))))
        p = float(np.mean(bootT >= tobs))
        if p >= alpha:
            break
        worst = int(np.argmax(dmean))
        p_values[survivors[worst]] = p
        survivors.pop(worst)
    return {"survivors": survivors, "p_values": p_values, "alpha": alpha,
            "n_models": K, "n_survivors": len(survivors)}


def pesaran_timmermann(predictions, actuals) -> dict:
    """Pesaran-Timmermann directional-accuracy test (sign skill vs independence)."""
    p, a = _a(predictions), _a(actuals)
    m = np.isfinite(p) & np.isfinite(a)
    p, a = p[m], a[m]
    n = len(p)
    if n < 30:
        return {"hit_rate": 0.0, "da_stat": 0.0, "p_value": 1.0, "verdict": "insufficient_data"}
    sp, sa = (np.sign(p) > 0).astype("float64"), (np.sign(a) > 0).astype("float64")
    P, Q = float(sa.mean()), float(sp.mean())
    phat = float((sp == sa).mean())
    pstar = P * Q + (1 - P) * (1 - Q)
    var_diff = (pstar * (1 - pstar) / n
                - ((2 * P - 1) ** 2 * Q * (1 - Q) / n + (2 * Q - 1) ** 2 * P * (1 - P) / n
                   + 4 * P * Q * (1 - P) * (1 - Q) / (n * n)))
    if var_diff <= 0:
        return {"hit_rate": phat, "da_stat": 0.0, "p_value": 1.0, "verdict": "degenerate"}
    da = (phat - pstar) / np.sqrt(var_diff)
    pv = 2.0 * (1.0 - stats.norm.cdf(abs(da)))
    return {"hit_rate": phat, "expected_hit_rate": pstar, "da_stat": float(da),
            "p_value": float(pv),
            "verdict": ("DIRECTIONAL_SKILL" if pv < 0.05 and da > 0
                        else "ANTI_DIRECTIONAL" if pv < 0.05 and da < 0
                        else "NO_DIRECTIONAL_SKILL")}


def stochastic_dominance(strategy_returns, benchmark_returns, *, order: int = 2,
                         n_grid: int = 80, n_bootstrap: int = 150, seed: int = 42) -> dict:
    """Davidson-Duclos stochastic-dominance test (order 1 FSD / 2 SSD / 3 TSD)
    of a strategy vs a benchmark, with a bootstrap critical value."""
    s, b = _a(strategy_returns), _a(benchmark_returns)
    s, b = s[np.isfinite(s)], b[np.isfinite(b)]
    if len(s) < 50 or len(b) < 50:
        return {"order": order, "verdict": "insufficient_data", "dominates": False}
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")
    grid = np.linspace(min(s.min(), b.min()), max(s.max(), b.max()), n_grid)
    fact = float(math.factorial(order - 1))

    def D(r, x):
        if order == 1:
            d = float((r < x).mean())
            return d, d * (1 - d) / len(r)
        arr = np.maximum(x - r, 0.0) ** (order - 1) / fact
        d = float(arr.mean())
        return d, float((np.mean(arr**2) - d**2) / len(r))

    Ds = [D(s, x) for x in grid]
    Db = [D(b, x) for x in grid]
    tmax_dom = tmax_anti = -np.inf
    for (d_s, v_s), (d_b, v_b) in zip(Ds, Db, strict=False):
        se = np.sqrt(max(v_s + v_b, 1e-12))
        tmax_dom = max(tmax_dom, (d_s - d_b) / se)
        tmax_anti = max(tmax_anti, (d_b - d_s) / se)
    rng = np.random.default_rng(seed)
    bd, ba = np.empty(n_bootstrap), np.empty(n_bootstrap)
    for ib in range(n_bootstrap):
        sb = s[rng.integers(0, len(s), len(s))]
        bb = b[rng.integers(0, len(b), len(b))]
        md = ma = -np.inf
        for gi, x in enumerate(grid):
            d_sb, v_sb = D(sb, x)
            d_bb, v_bb = D(bb, x)
            se = np.sqrt(max(v_sb + v_bb, 1e-12))
            md = max(md, ((d_sb - d_bb) - (Ds[gi][0] - Db[gi][0])) / se)
            ma = max(ma, ((d_bb - d_sb) - (Db[gi][0] - Ds[gi][0])) / se)
        bd[ib], ba[ib] = md, ma
    cv_dom, cv_anti = float(np.percentile(bd, 95)), float(np.percentile(ba, 95))
    dominates = bool(tmax_dom < cv_dom and tmax_anti > cv_anti)
    is_dominated = bool(tmax_anti < cv_anti and tmax_dom > cv_dom)
    return {"order": order, "dominates": dominates, "is_dominated": is_dominated,
            "t_max_dominates": float(tmax_dom), "t_max_dominated": float(tmax_anti),
            "p_value_dominates": float(np.mean(bd >= tmax_dom)),
            "verdict": (f"STRATEGY_DOMINATES_SD{order}" if dominates
                        else f"BENCHMARK_DOMINATES_SD{order}" if is_dominated
                        else "NO_DOMINANCE")}
