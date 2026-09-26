"""Portfolio risk metrics not covered by ``rigor.metrics`` (VaR/CVaR/ulcer/beta).

Everything Sharpe/CAGR/MaxDD/Sortino/Calmar-related comes from ``rigor.metrics`` —
these are the portfolio-specific extras the builder reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "value_at_risk", "conditional_var", "ulcer_index",
    "diversification_ratio", "beta_to_benchmark",
    "marginal_risk_contributions", "risk_concentration_hhi", "risk_hhi",
    "autocorrelation_profile",
]


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR: the ``alpha`` quantile of the return distribution (negative)."""
    r = returns.dropna().to_numpy()
    return float(np.percentile(r, alpha * 100)) if len(r) else 0.0


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of returns at or below the VaR threshold."""
    r = returns.dropna().to_numpy()
    if not len(r):
        return 0.0
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index: RMS of the percent drawdown path (depth *and* duration)."""
    r = returns.dropna().to_numpy()
    if not len(r):
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100.0
    return float(np.sqrt(np.mean(dd ** 2)))


def diversification_ratio(returns_matrix: pd.DataFrame, weights: np.ndarray) -> dict:
    """Weighted-average vol / portfolio vol. >1 means correlation gives a
    diversification benefit. Also returns the average pairwise correlation."""
    rm = returns_matrix.dropna(how="any")
    if rm.shape[1] < 2 or len(rm) < 2:
        return {"ratio": 1.0, "avg_pairwise_corr": 0.0, "portfolio_vol": 0.0}
    w = np.abs(weights)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    cov = rm.cov().to_numpy()
    vols = np.sqrt(np.clip(np.diag(cov), 0, None))
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    weighted_vol = float(w @ vols)
    corr = rm.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    avg_corr = float(np.nanmean(corr[iu])) if len(iu[0]) else 0.0
    ratio = weighted_vol / port_vol if port_vol > 0 else 1.0
    return {"ratio": ratio, "avg_pairwise_corr": avg_corr, "portfolio_vol": port_vol}


def marginal_risk_contributions(
    returns_matrix: pd.DataFrame,
    weights: np.ndarray,
    *,
    annualise: bool = True,
    ppy: int = 252,
) -> dict:
    """Euler risk decomposition: marginal + percentage risk contributions per leg.

    Parameters
    ----------
    returns_matrix:
        Per-leg daily returns (columns = strategy/leg symbols).
    weights:
        Portfolio weights aligned to ``returns_matrix`` columns.
    annualise:
        When True, scale all vol figures by ``sqrt(ppy)``.
    ppy:
        Periods per year used for annualisation (default 252 business days).

    Returns
    -------
    dict with keys: ``portfolio_vol``, ``assets``, ``weight``, ``mrc``,
    ``risk_contribution``, ``pct_risk``, ``pct_weight``, ``concentration_flag``.

    Math
    ----
    Let Σ = sample covariance of leg returns.  Portfolio vol σ_p = sqrt(wᵀΣw).

    * MRCᵢ  = (Σw)ᵢ / σ_p          (marginal risk contribution)
    * RCᵢ   = wᵢ · MRCᵢ            (risk contribution; Euler: Σ RCᵢ = σ_p)
    * pct_i = RCᵢ / σ_p             (risk share; Σ pct_i = 1)

    ``concentration_flag[i]`` is True when pct_riskᵢ - pct_weightᵢ > 0.15 —
    a leg carrying materially more risk than its weight suggests.
    """
    rm = returns_matrix.dropna(how="any")
    n = rm.shape[1]
    symbols = list(rm.columns)

    w = np.asarray(weights, dtype=float)
    if len(w) != n:
        msg = f"weights length {len(w)} != returns_matrix columns {n}"
        raise ValueError(msg)

    zeros: dict = {
        "portfolio_vol": 0.0,
        "assets": symbols,
        "weight": w.tolist(),
        "mrc": [0.0] * n,
        "risk_contribution": [0.0] * n,
        "pct_risk": [0.0] * n,
        "pct_weight": (w / w.sum()).tolist() if w.sum() != 0 else [0.0] * n,
        "concentration_flag": [False] * n,
    }

    if n == 0 or len(rm) < 2:
        return zeros

    # Covariance matrix (per-period); guard against singular/degenerate case.
    try:
        cov = rm.cov().to_numpy()
        var_p = float(w @ cov @ w)
        if var_p <= 0.0 or not np.isfinite(var_p):
            return zeros
        sigma_p = float(np.sqrt(var_p))
    except Exception:  # noqa: BLE001
        return zeros

    scale = float(np.sqrt(ppy)) if annualise else 1.0
    sigma_p_ann = sigma_p * scale

    sigma_w = cov @ w  # (Σw)ᵢ — vector
    mrc = sigma_w / sigma_p  # per-period MRC; annualise below
    rc = w * mrc             # risk contributions (sum to sigma_p per period)

    mrc_ann = (mrc * scale).tolist()
    rc_ann = (rc * scale).tolist()
    pct_risk = (rc / sigma_p).tolist()  # dimensionless; no scale needed

    w_sum = w.sum()
    pct_weight = (w / w_sum).tolist() if w_sum != 0 else [0.0] * n

    concentration_flag = [
        bool((pr - pw) > 0.15) for pr, pw in zip(pct_risk, pct_weight, strict=True)
    ]

    return {
        "portfolio_vol": sigma_p_ann,
        "assets": symbols,
        "weight": w.tolist(),
        "mrc": mrc_ann,
        "risk_contribution": rc_ann,
        "pct_risk": pct_risk,
        "pct_weight": pct_weight,
        "concentration_flag": concentration_flag,
    }


def risk_concentration_hhi(weights: np.ndarray) -> float:
    """Herfindahl–Hirschman Index on portfolio weights.

    Returns the sum of squared normalised weights.  Ranges from ``1/n``
    (perfectly equal) to ``1`` (fully concentrated in one leg).
    """
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum == 0 or len(w) == 0:
        return 1.0
    w_norm = w / w_sum
    return float(np.dot(w_norm, w_norm))


def risk_hhi(returns_matrix: pd.DataFrame, weights: np.ndarray) -> float:
    """Herfindahl–Hirschman Index on *risk shares* (pct_risk).

    Uses the Euler decomposition to get each leg's share of total portfolio
    volatility, then returns ``Σ pct_riskᵢ²``.  Ranges from ``1/n`` (risk
    spread evenly) to ``1`` (all risk in one leg).
    """
    result = marginal_risk_contributions(returns_matrix, weights, annualise=False)
    pct = np.asarray(result["pct_risk"])
    if pct.sum() == 0:
        n = returns_matrix.shape[1]
        return 1.0 / n if n > 0 else 1.0
    return float(np.dot(pct, pct))


def beta_to_benchmark(port_returns: pd.Series, bench_returns: pd.Series) -> dict:
    """OLS beta + up/down beta + correlation of portfolio vs benchmark on aligned bars."""
    df = pd.concat({"p": port_returns, "b": bench_returns}, axis=1).dropna()
    if len(df) < 30 or df["b"].var() == 0:
        return {"beta": 0.0, "up_beta": 0.0, "down_beta": 0.0, "corr": 0.0, "n_obs": len(df)}
    p, b = df["p"].to_numpy(), df["b"].to_numpy()

    def _beta(mask):
        if mask.sum() < 5 or b[mask].var() == 0:
            return 0.0
        return float(np.cov(p[mask], b[mask])[0, 1] / b[mask].var())

    return {
        "beta": _beta(np.ones(len(b), dtype=bool)),
        "up_beta": _beta(b > 0),
        "down_beta": _beta(b < 0),
        "corr": float(np.corrcoef(p, b)[0, 1]),
        "n_obs": len(df),
    }


def autocorrelation_profile(
    returns: pd.Series | np.ndarray,
    *,
    max_lag: int = 10,
) -> dict:
    """Sample ACF leakage profile for a single return series.

    Parameters
    ----------
    returns:
        Daily (or other frequency) return series.  NaNs are dropped.
    max_lag:
        Number of lags to compute (lags 1 … max_lag).

    Returns
    -------
    dict with keys:

    ``lags``
        List of integers ``[1, …, max_lag]``.
    ``acf``
        Sample autocorrelation at each lag.
    ``ljung_box_p``
        Ljung-Box joint p-value for lags 1 … max_lag (via
        ``scipy.stats.acf`` / ``statsmodels``-free implementation).
        ``None`` if scipy is unavailable or ``n < 20``.
    ``leakage_flag``
        ``True`` when lag-1 ACF > 0.20 — signature of return smoothing,
        stale prices, or illiquidity.  Always ``False`` when ``n < 20``.

    Notes
    -----
    Ljung-Box statistic: Q = n(n+2) Σ_{k=1}^{max_lag} ρ̂_k² / (n-k).
    Under H₀ of white noise, Q ~ χ²(max_lag).
    """
    if isinstance(returns, pd.Series):
        arr = returns.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]

    n = len(arr)
    lags = list(range(1, max_lag + 1))
    short_series = n < 20

    if short_series or n == 0:
        return {
            "lags": lags,
            "acf": [0.0] * max_lag,
            "ljung_box_p": None,
            "leakage_flag": False,
        }

    # Centre the series
    mu = arr.mean()
    x = arr - mu
    var = float(np.dot(x, x))  # n * sample variance (unnormalised)

    acf_vals: list[float] = []
    for k in lags:
        if var == 0:
            acf_vals.append(0.0)
        else:
            cov_k = float(np.dot(x[: n - k], x[k:]))
            acf_vals.append(cov_k / var)  # normalise by var (= n * σ²)

    # Ljung-Box Q-statistic
    ljung_box_p: float | None = None
    try:
        from scipy.stats import chi2  # noqa: PLC0415

        q_stat = float(
            n * (n + 2) * sum(rho**2 / (n - k) for k, rho in zip(lags, acf_vals, strict=True))
        )
        ljung_box_p = float(1.0 - chi2.cdf(q_stat, df=max_lag))
    except Exception:  # noqa: BLE001
        ljung_box_p = None

    leakage_flag = bool(acf_vals[0] > 0.20) if acf_vals else False

    return {
        "lags": lags,
        "acf": acf_vals,
        "ljung_box_p": ljung_box_p,
        "leakage_flag": leakage_flag,
    }
