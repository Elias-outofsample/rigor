"""Portfolio analytics — the deeper diagnostics the Backtest/Robustness tabs show.

Everything here operates on an already-computed portfolio return series (and,
where relevant, a benchmark). Significance reuses ``rigor.validation`` (bootstrap
Sharpe CI, PSR, deflated Sharpe); the rest (beta decomposition, capture ratios,
regimes, Monte-Carlo, ruin, frontier, rebalance sweep) is ported from the prior
research library, on plain NumPy + ``rigor.metrics``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rigor import metrics as _m
from scipy.stats import kurtosis, pearsonr, skew

__all__ = [
    "beta_decomposition", "capture_ratios", "drawdown_beta", "conditional_correlation",
    "rolling_beta", "rolling_correlation", "significance", "market_regimes",
    "regime_performance", "monte_carlo", "ruin_probability", "efficient_frontier",
    "rebalance_sweep", "portfolio_sector_exposure", "attribution", "factor_regression",
]


def _safe_div(a, b, default=0.0):
    return a / b if b not in (0, 0.0) and np.isfinite(b) else default


def _drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1.0 + returns.fillna(0)).cumprod()
    return eq / eq.cummax() - 1.0


# --- beta / correlation family --------------------------------------------

def beta_decomposition(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Full / up-market / down-market beta + Jensen's alpha (annualised)."""
    a = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(a) < 10:
        return {"beta": 0.0, "up_beta": 0.0, "down_beta": 0.0, "alpha": 0.0}
    ppy = _m.periods_per_year_of(a.index)
    r, b = a["r"].to_numpy(), a["b"].to_numpy()

    def _beta(mask):
        if mask.sum() <= 5 or b[mask].var() == 0:
            return 0.0
        return _safe_div(np.cov(r[mask], b[mask], ddof=1)[0, 1], np.var(b[mask], ddof=1))

    full = _beta(np.ones(len(b), dtype=bool))
    return {
        "beta": float(full), "up_beta": float(_beta(b > 0)), "down_beta": float(_beta(b <= 0)),
        "alpha": float((r.mean() - full * b.mean()) * ppy),
    }


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Morningstar up/down capture ratios."""
    a = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(a) < 20:
        return {"up_capture": 0.0, "down_capture": 0.0, "capture_ratio": 0.0}
    up, down = a[a["b"] > 0], a[a["b"] < 0]
    uc = _safe_div(up["r"].mean(), up["b"].mean()) if len(up) else 0.0
    dc = _safe_div(down["r"].mean(), down["b"].mean()) if len(down) else 0.0
    return {"up_capture": float(uc), "down_capture": float(dc),
            "capture_ratio": float(_safe_div(uc, dc))}


def drawdown_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Beta of the portfolio drawdown path vs the benchmark drawdown path."""
    a = pd.DataFrame({"ds": _drawdown_series(returns), "db": _drawdown_series(benchmark)}).dropna()
    if len(a) < 10 or a["db"].var() == 0:
        return 0.0
    return float(_safe_div(np.cov(a["ds"], a["db"], ddof=1)[0, 1], a["db"].var()))


def conditional_correlation(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Correlation with the benchmark on up vs down benchmark days."""
    a = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(a) < 20:
        return {"up_corr": 0.0, "down_corr": 0.0, "asymmetry": 0.0}
    up, down = a[a["b"] > 0], a[a["b"] <= 0]
    uc = float(pearsonr(up["r"], up["b"])[0]) if len(up) > 5 else 0.0
    dc = float(pearsonr(down["r"], down["b"])[0]) if len(down) > 5 else 0.0
    return {"up_corr": uc, "down_corr": dc, "asymmetry": dc - uc}


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    a = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(a) < window:
        return pd.Series(dtype="float64")
    cov = a["r"].rolling(window).cov(a["b"])
    var = a["b"].rolling(window).var()
    return (cov / var).dropna()


def rolling_correlation(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    a = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(a) < window:
        return pd.Series(dtype="float64")
    return a["r"].rolling(window).corr(a["b"]).dropna()


# --- significance (reuses rigor.validation) ---------------------------------

def significance(returns: pd.Series, *, n_trials: int = 1) -> dict:
    """Bootstrap Sharpe CI + PSR + deflated Sharpe for the portfolio."""
    from rigor.validation.overfit import deflated_sharpe_ratio, probabilistic_sharpe_ratio
    from rigor.validation.robustness import bootstrap_sharpe_ci
    r = returns.dropna()
    arr = r.to_numpy(dtype="float64")
    n = len(arr)
    if n < 30:
        return {"note": "too few observations"}
    ppy = _m.periods_per_year_of(r.index)
    sharpe_pp = _m.compute_sharpe(arr, ppy) / np.sqrt(ppy)
    sk = float(skew(arr, bias=False))
    ku = float(kurtosis(arr, fisher=True, bias=False))
    return {
        "bootstrap_ci": bootstrap_sharpe_ci(r),
        "psr": float(probabilistic_sharpe_ratio(sharpe_pp, 0.0, n, sk, ku)),
        "dsr": float(deflated_sharpe_ratio(sharpe_pp, n_trials, n, sk, ku)),
    }


# --- regimes ---------------------------------------------------------------

def market_regimes(returns: pd.Series, benchmark: pd.Series | None = None,
                   *, vol_window: int = 63, ma_window: int = 200) -> pd.Series:
    """6-bucket Trend x Vol regime labels (Bull/Bear x Calm/Mid/Stress), lagged 1 bar."""
    ppy = _m.periods_per_year_of(returns.index)
    base = benchmark if benchmark is not None else returns
    prices = (1.0 + base.reindex(returns.index).fillna(0)).cumprod()
    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    rv = (prices.pct_change().fillna(0).rolling(vol_window, min_periods=vol_window).std()
          * np.sqrt(ppy))
    vol = pd.Series("Mid", index=returns.index, dtype=object)
    vol[rv <= 0.12] = "Calm"
    vol[rv >= 0.20] = "Stress"
    trend = pd.Series(np.where((prices > ma).fillna(False), "Bull", "Bear"),
                      index=returns.index, dtype=object)
    labels = trend + "-" + vol
    labels[ma.isna()] = "Bear-Mid"
    return labels.shift(1).fillna("Bear-Mid")


def regime_performance(returns: pd.Series, regimes: pd.Series) -> dict:
    """Per-regime Sharpe / CAGR / MaxDD / win-rate / time share."""
    a = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    out = {}
    for reg in sorted(a["regime"].unique()):
        r = a.loc[a["regime"] == reg, "ret"]
        if len(r) < 5:
            continue
        arr = r.to_numpy(dtype="float64")
        ppy = _m.periods_per_year_of(r.index)
        out[str(reg)] = {
            "sharpe": float(_m.compute_sharpe(arr, ppy)), "cagr": float(_m.compute_cagr(arr, ppy)),
            "max_dd": float(_m.compute_max_drawdown(arr)), "win_rate": float((arr > 0).mean()),
            "n_obs": len(r), "pct_time": float(len(r) / len(a)),
        }
    return out


# --- Monte-Carlo / ruin ----------------------------------------------------

def monte_carlo(returns: pd.Series, *, n_sims: int = 1000, seed: int = 42) -> dict:
    """Bootstrap equity-path simulation: loss/double probabilities + DD distribution."""
    r = returns.dropna().to_numpy(dtype="float64")
    n = len(r)
    if n < 30:
        return {"note": "too few observations"}
    rng = np.random.default_rng(seed)
    paths = np.cumprod(1.0 + r[rng.integers(0, n, size=(n_sims, n))], axis=1)
    final = paths[:, -1]
    dd = np.array([(p / np.maximum.accumulate(p) - 1.0).min() for p in paths])
    return {
        "prob_loss": float(np.mean(final < 1.0)), "prob_double": float(np.mean(final >= 2.0)),
        "median_return": float(np.median(final) - 1.0),
        "mean_max_dd": float(dd.mean()), "p5_max_dd": float(np.percentile(dd, 5)),
        "p95_max_dd": float(np.percentile(dd, 95)), "worst_max_dd": float(dd.min()),
        "n_sims": n_sims,
    }


def ruin_probability(returns: pd.Series, *, target_drawdown: float = -0.20,
                     horizon: int | None = None, n_sims: int = 5000, seed: int = 42) -> dict:
    """P(hit ``target_drawdown`` within ``horizon`` bars), via bootstrap resampling."""
    r = returns.dropna().to_numpy(dtype="float64")
    n = len(r)
    if n < 30:
        return {"ruin_prob": float("nan")}
    h = max(10, min(int(horizon or n), n))
    rng = np.random.default_rng(seed)
    ruin = 0
    for _ in range(n_sims):
        eq = np.cumprod(1.0 + r[rng.integers(0, n, size=h)])
        if (eq / np.maximum.accumulate(eq) - 1.0).min() <= target_drawdown:
            ruin += 1
    return {"ruin_prob": ruin / n_sims, "target_drawdown": target_drawdown,
            "horizon": h, "n_sims": n_sims}


# --- frontier / rebalance / sector ----------------------------------------

def efficient_frontier(returns_matrix: pd.DataFrame, *, n_samples: int = 4000,
                       seed: int = 0) -> pd.DataFrame:
    """Long-only efficient frontier (dependency-free): sample the simplex, keep the
    upper hull of the (vol, return) cloud — max return per volatility bucket."""
    rm = returns_matrix.dropna(how="any")
    n = rm.shape[1]
    if n < 2 or len(rm) < 30:
        return pd.DataFrame()
    ppy = _m.periods_per_year_of(rm.index)
    mu = rm.mean().to_numpy() * ppy
    cov = rm.cov().to_numpy() * ppy
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n), size=n_samples)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w).clip(0))
    df = pd.DataFrame({"ret": ret, "vol": vol})
    df["bucket"] = pd.cut(df["vol"], 30)
    hull = df.loc[df.groupby("bucket", observed=True)["ret"].idxmax()].dropna()
    hull = hull.sort_values("vol")
    hull["sharpe"] = hull["ret"] / hull["vol"].where(hull["vol"] > 1e-9)
    return hull[["vol", "ret", "sharpe"]].reset_index(drop=True)


def rebalance_sweep(returns_matrix: pd.DataFrame, target_weights: np.ndarray) -> pd.DataFrame:
    """Portfolio metrics under different rebalance cadences (weights drift between
    rebalances). Reports Sharpe / CAGR / MaxDD / average turnover per cadence."""
    rm = returns_matrix.dropna(how="any")
    if len(rm) < 252:
        return pd.DataFrame()
    ppy = _m.periods_per_year_of(rm.index)
    freqs = {"Never": None, "Annual": "YE", "Quarterly": "QE", "Monthly": "ME", "Weekly": "W"}
    target = np.asarray(target_weights, dtype=float)
    rows = {}
    for label, freq in freqs.items():
        rebal = ({rm.index[0]} if freq is None
                 else set(rm.resample(freq).first().dropna(how="all").index))
        w = target.copy()
        port = np.zeros(len(rm))
        turn, n_reb = 0.0, 0
        for i, (date, row) in enumerate(rm.iterrows()):
            if date in rebal and i > 0:
                turn += float(np.abs(w - target).sum())
                n_reb += 1
                w = target.copy()
            vals = row.to_numpy()
            port[i] = float((vals * w).sum())
            w = w * (1.0 + vals)
            s = w.sum()
            if s > 0:
                w = w / s
        rows[label] = {"sharpe": float(_m.compute_sharpe(port, ppy)),
                       "cagr": float(_m.compute_cagr(port, ppy)),
                       "max_dd": float(_m.compute_max_drawdown(port)),
                       "turnover": turn / max(n_reb, 1)}
    return pd.DataFrame(rows).T


def attribution(returns_matrix: pd.DataFrame, weights: np.ndarray,
                *, ppy: int | None = None) -> pd.DataFrame:
    """Per-leg return *and* risk attribution, sorted by risk share.

    Return contribution is ``wᵢ · μᵢ`` (annualised, linearised); risk contribution
    is the Euler share ``RCᵢ/σ_p`` from :func:`risk.marginal_risk_contributions`
    (Σ pct_risk = 1). The two columns answer "which leg drives the return" and
    "which leg drives the risk" — they rarely coincide, which is the point.
    """
    from . import risk as _risk
    rm = returns_matrix.dropna(how="any")
    names = list(rm.columns)
    n = len(names)
    cols = ["name", "weight", "ret_contrib", "ret_contrib_pct", "risk_contrib_pct", "sharpe"]
    w = np.asarray(weights, dtype=float)
    if n == 0 or len(rm) < 2 or w.size != n or w.sum() <= 0:
        return pd.DataFrame(columns=cols)
    ppy = ppy or _m.periods_per_year_of(rm.index)
    wn = w / w.sum()
    mu = rm.mean().to_numpy()
    sd = rm.std(ddof=1).to_numpy()
    ret_contrib = wn * mu * ppy                       # annualised return contribution
    total = ret_contrib.sum()
    ret_pct = ret_contrib / total if total != 0 else np.zeros(n)
    rc = _risk.marginal_risk_contributions(rm, wn, ppy=ppy)
    risk_pct = np.asarray(rc["pct_risk"], dtype=float)
    sharpe = np.where(sd > 1e-12, mu / sd * np.sqrt(ppy), 0.0)
    df = pd.DataFrame({
        "name": names, "weight": wn, "ret_contrib": ret_contrib,
        "ret_contrib_pct": ret_pct, "risk_contrib_pct": risk_pct, "sharpe": sharpe,
    })
    return df.sort_values("risk_contrib_pct", ascending=False).reset_index(drop=True)


def factor_regression(returns: pd.Series, factors: dict[str, pd.Series],
                      *, ppy: int | None = None) -> dict:
    """OLS of portfolio returns on factor returns with Newey-West (HAC) t-stats.

    ``factors`` is a name→return-series map (e.g. ETF-proxy market/size/value/…).
    Returns annualised alpha + its t-stat, each factor's beta + t-stat, and R².
    Degrades to a ``note`` when there's no factor data (offline) or too little
    overlap — the app never requires a live data connection.
    """
    if not factors:
        return {"note": "no factor data (offline or unavailable)"}
    data = pd.concat({"__y__": returns, **factors}, axis=1).dropna()
    names = list(factors.keys())
    n = len(data)
    if n < 60:
        return {"note": "too few overlapping observations for a factor fit"}
    y = data["__y__"].to_numpy()
    X = np.column_stack([np.ones(n)] + [data[name].to_numpy() for name in names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {"note": "collinear factors — regression is singular"}
    lag = max(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), 1)   # Newey-West rule
    xe = X * resid[:, None]
    s = xe.T @ xe
    for k in range(1, lag + 1):
        g = xe[k:].T @ xe[:-k]
        s += (1.0 - k / (lag + 1.0)) * (g + g.T)
    se = np.sqrt(np.clip(np.diag(xtx_inv @ s @ xtx_inv), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, np.nan)
    ppy = ppy or _m.periods_per_year_of(data.index)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "alpha_ann": float(beta[0] * ppy), "alpha_t": float(tvals[0]),
        "r_squared": float(r2), "n_obs": n, "nw_lag": lag,
        "factors": {name: {"beta": float(beta[i + 1]), "t_stat": float(tvals[i + 1])}
                    for i, name in enumerate(names)},
    }


def portfolio_sector_exposure(selected, weights: dict[str, float]) -> dict[str, float]:
    """Aggregate weighted sector exposure across the selected strategies."""
    out: dict[str, float] = {}
    for cand in selected:
        w = weights.get(cand.name, 0.0)
        if abs(w) < 1e-9:
            continue
        for sector, frac in (cand.sector_exposure or {"Unknown": 1.0}).items():
            out[sector] = out.get(sector, 0.0) + w * frac
    total = sum(out.values())
    return {k: v / total for k, v in sorted(out.items(), key=lambda kv: -kv[1])} if total else out
