"""Factor attribution (#2) — Fama-French / Carhart factor regressions.

Decomposes a strategy's return into factor exposures + residual alpha, with
Newey-West HAC t-stats and a progressive (nested-model) alpha that shows how much
alpha survives as you add factors (CAPM → FF3 → FF5 → FF6 → FF9):

  * ``compute_factor_regression`` — OLS of returns on a factor panel, HAC t-stats,
    R²/adj-R², VIF, invested-only filter, optional cost-drag net alpha.
  * ``compute_progressive_alpha`` — alpha + absorbed-% across nested models.
  * ``compute_vif`` — variance-inflation factors (multicollinearity diagnostic).
  * ``residualize_by_sector`` — sector-neutralise a cross-sectional signal.
  * ``load_french_factors`` — fetch FF5 daily via ``pandas-datareader`` (optional);
    returns an empty frame offline, so callers pass their own factor panel.

HAC and VIF are implemented in pure numpy here, so the core regression needs **no**
optional dependency — only ``load_french_factors`` does (network + pandas-datareader).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t

from .. import metrics as _m

__all__ = [
    "FACTOR_MODELS", "compute_factor_regression", "compute_progressive_alpha",
    "compute_vif", "residualize_by_sector", "load_french_factors",
]

FACTOR_MODELS: dict[str, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    "FF9": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "BAB", "ST_Rev", "LT_Rev"],
    "Q5": ["Mkt-RF", "ME", "IA", "ROE", "EG"],
}


def _ppy(s: pd.Series) -> int:
    if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
        return _m.periods_per_year_of(s.index)
    return _m.TRADING_DAYS


def _nw_auto_lags(n: int) -> int:
    """Newey-West (1994) automatic lag selection."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _resample_factors_to_returns_freq(returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Compound factor returns to the (coarser) strategy frequency so betas stay on-scale."""
    if not (isinstance(returns.index, pd.DatetimeIndex)
            and isinstance(factors.index, pd.DatetimeIndex)):
        return factors
    if len(factors) < 50 or len(returns) < 10:
        return factors
    ret_ppy = _ppy(returns)
    fac_ppy = _m.periods_per_year_of(factors.index)
    if ret_ppy >= fac_ppy * 0.5:
        return factors
    target = {12: "ME", 52: "W-FRI", 4: "QE", 1: "YE"}.get(int(round(ret_ppy)), "ME")
    return (1.0 + factors).resample(target).prod() - 1.0


def _ols_hac(X: np.ndarray, y: np.ndarray, use_hac: bool,
             maxlags: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients + standard errors (HAC/Newey-West when ``use_hac``).

    Returns (coeffs, se, resid). The HAC sandwich is computed in pure numpy.
    """
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    coeffs = XtX_inv @ (X.T @ y)
    resid = y - X @ coeffs
    if not use_hac:
        sigma2 = float(resid @ resid) / max(n - k, 1)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        return coeffs, se, resid
    lags = maxlags if maxlags is not None else _nw_auto_lags(n)
    lags = max(0, min(lags, n - 1))
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        gamma = u[L:].T @ u[:-L]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    return coeffs, np.sqrt(np.maximum(np.diag(cov), 0.0)), resid


def compute_vif(factors: pd.DataFrame) -> dict[str, float]:
    """Variance Inflation Factor per column (>5 moderate, >10 severe collinearity)."""
    X = factors.dropna()
    names = list(X.columns)
    vif: dict[str, float] = {}
    if len(X) < len(names) + 2:
        return {n: float("nan") for n in names}
    arr = X.to_numpy(dtype=np.float64)
    for i, name in enumerate(names):
        y = arr[:, i]
        others = np.delete(arr, i, axis=1)
        Z = np.column_stack([np.ones(len(y)), others])
        try:
            beta = np.linalg.lstsq(Z, y, rcond=None)[0]
            resid = y - Z @ beta
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
            vif[name] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
        except (np.linalg.LinAlgError, ValueError):
            vif[name] = float("nan")
    return vif


def compute_factor_regression(returns: pd.Series, factors: pd.DataFrame | None = None,
                              invested_only: bool = False, use_hac: bool = False,
                              hac_maxlags: int | None = None,
                              cost_drag_annual: float = 0.0,
                              resample_factors: bool = True) -> dict:
    """OLS factor regression with optional Newey-West HAC and invested-only filter.

    ``factors`` is a panel with one column per factor plus an optional ``RF`` column
    (excess returns are formed when present). ``returns`` are total returns. Returns
    annualised alpha, betas, HAC/OLS t-stats and p-values, R², VIF and exposure.
    """
    if factors is None:
        factors = load_french_factors()
    if factors is None or factors.empty:
        return {"error": "no factor data"}
    if resample_factors:
        factors = _resample_factors_to_returns_freq(returns, factors)

    aligned = pd.DataFrame({"ret": returns}).join(factors, how="inner").dropna()
    if len(aligned) < 30:
        return {"error": "insufficient aligned data"}
    n_total = len(aligned)
    invested_mask = aligned["ret"] != 0
    n_invested = int(invested_mask.sum())
    exposure = n_invested / n_total if n_total > 0 else 0.0
    if invested_only:
        aligned = aligned[invested_mask]
        if len(aligned) < 30:
            return {"error": "insufficient invested-only data"}

    y = aligned["ret"].to_numpy(dtype=np.float64)
    if "RF" in aligned.columns:
        y = y - aligned["RF"].to_numpy(dtype=np.float64)
    factor_cols = [c for c in factors.columns if c != "RF"]
    X = np.column_stack([np.ones(len(y)), aligned[factor_cols].to_numpy(dtype=np.float64)])

    coeffs, se_ols, resid = _ols_hac(X, y, use_hac=False, maxlags=None)
    ppy = _ppy(returns)
    alpha = float(coeffs[0]) * ppy
    betas = {col: float(coeffs[i + 1]) for i, col in enumerate(factor_cols)}

    n, k = X.shape
    df = max(n - k, 1)
    t_ols = coeffs / np.where(se_ols > 0, se_ols, 1e-12)
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / df if n > k else 0.0
    p_alpha_ols = float(2.0 * scipy_t.sf(abs(t_ols[0]), df))
    beta_t_ols = {col: float(t_ols[i + 1]) for i, col in enumerate(factor_cols)}
    beta_p_ols = {col: float(2.0 * scipy_t.sf(abs(t_ols[i + 1]), df))
                  for i, col in enumerate(factor_cols)}

    t_alpha, p_alpha, beta_t, beta_p, se_type = (float(t_ols[0]), p_alpha_ols,
                                                 beta_t_ols, beta_p_ols, "OLS")
    if use_hac:
        coeffs_h, se_hac, _ = _ols_hac(X, y, use_hac=True, maxlags=hac_maxlags)
        t_hac = coeffs_h / np.where(se_hac > 0, se_hac, 1e-12)
        t_alpha = float(t_hac[0])
        p_alpha = float(2.0 * scipy_t.sf(abs(t_hac[0]), df))
        beta_t = {col: float(t_hac[i + 1]) for i, col in enumerate(factor_cols)}
        beta_p = {col: float(2.0 * scipy_t.sf(abs(t_hac[i + 1]), df))
                  for i, col in enumerate(factor_cols)}
        se_type = "HAC"

    vif = compute_vif(aligned[factor_cols]) if len(factor_cols) >= 2 else {}
    alpha_net = alpha - float(cost_drag_annual)
    alpha_net_t = t_alpha * (alpha_net / alpha) if abs(alpha) > 1e-12 else t_alpha
    alpha_net_p = float(2.0 * scipy_t.sf(abs(alpha_net_t), df))
    sigma = float(np.sqrt(ss_res / df))
    return {
        "alpha": alpha, "alpha_gross": alpha, "alpha_net": alpha_net,
        "alpha_net_t_stat": float(alpha_net_t), "alpha_net_p_value": alpha_net_p,
        "cost_drag_annual": float(cost_drag_annual), "alpha_t_stat": t_alpha,
        "alpha_p_value": p_alpha, "alpha_t_stat_ols": float(t_ols[0]),
        "alpha_p_value_ols": p_alpha_ols, "betas": betas, "beta_t_stats": beta_t,
        "beta_p_values": beta_p, "beta_t_stats_ols": beta_t_ols, "beta_p_values_ols": beta_p_ols,
        "r_squared": float(r2), "adj_r_squared": float(adj_r2), "residual_std": sigma,
        "n_obs": n, "n_invested": n_invested, "n_total": n_total, "exposure": float(exposure),
        "se_type": se_type, "vif": vif,
    }


def compute_progressive_alpha(returns: pd.Series, factors: pd.DataFrame,
                              invested_mask: pd.Series | None = None,
                              models: dict[str, list[str]] | None = None,
                              hac_maxlags: int = 10) -> dict[str, dict]:
    """Run each nested factor model; report alpha, HAC t/p, R², and absorbed-%."""
    if models is None:
        models = FACTOR_MODELS
    ppy = _ppy(returns)
    out: dict[str, dict] = {}
    for model_name, cols in models.items():
        available = [c for c in cols if c in factors.columns]
        if not available:
            continue
        if invested_mask is not None:
            idx = returns[invested_mask].index
            X_data = factors[available].reindex(idx)
            y = returns[invested_mask].to_numpy(dtype=np.float64)
            valid = X_data.notna().all(axis=1)
            y = y[valid.to_numpy()]
            X_data = X_data[valid]
            exposure = float(invested_mask.mean())
        else:
            y = returns.to_numpy(dtype=np.float64)
            X_data = factors[available]
            exposure = 1.0
        if len(y) < 30:
            continue
        X = np.column_stack([np.ones(len(y)), X_data.to_numpy(dtype=np.float64)])
        coeffs, se_hac, resid = _ols_hac(X, y, use_hac=True, maxlags=hac_maxlags)
        t_hac = coeffs / np.where(se_hac > 0, se_hac, 1e-12)
        n_obs, n_k = len(y), len(available) + 1
        ss_res = float(resid @ resid)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n_obs - 1) / max(n_obs - n_k, 1)
        df = max(n_obs - n_k, 1)
        alpha_cond = float(coeffs[0]) * ppy
        mean_excess = float(np.mean(y)) * ppy
        factor_explained = mean_excess - alpha_cond
        absorbed = factor_explained / mean_excess * 100 if abs(mean_excess) > 1e-10 else 0.0
        out[model_name] = {
            "alpha_cond": float(alpha_cond), "alpha_eff": float(alpha_cond * exposure),
            "t": float(t_hac[0]), "p": float(2.0 * scipy_t.sf(abs(t_hac[0]), df)),
            "r2": float(r2), "adj_r2": float(adj_r2), "n_factors": len(available),
            "n_obs": n_obs, "exposure": exposure, "mean_excess_cond": float(mean_excess),
            "factor_explained": float(factor_explained), "absorbed_pct": float(absorbed),
            "loadings": {available[i]: float(coeffs[i + 1]) for i in range(len(available))},
            "loading_t": {available[i]: float(t_hac[i + 1]) for i in range(len(available))},
        }
    return out


def residualize_by_sector(series: pd.Series, sectors, method: str = "demean") -> pd.Series:
    """Sector-neutralise a cross-sectional signal (subtract sector mean, or OLS on dummies)."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if isinstance(sectors, dict):
        sectors = pd.Series(sectors, name="sector")
    aligned = pd.DataFrame({"x": series, "sector": sectors.reindex(series.index)})
    out = series.astype(float).copy()
    known = aligned["sector"].notna()
    if not known.any():
        return out
    if method == "demean":
        means = aligned.loc[known].groupby("sector")["x"].transform("mean")
        out.loc[known] = aligned.loc[known, "x"] - means
    elif method == "ols":
        sub = aligned.loc[known]
        X = pd.get_dummies(sub["sector"], drop_first=False, dtype=float).to_numpy()
        y = sub["x"].to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        out.loc[known] = y - X @ beta
    else:
        raise ValueError(f"unknown method {method!r}; expected 'demean' or 'ols'")
    return out


def load_french_factors(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fama-French 5 daily factors via ``pandas-datareader`` (optional, offline-safe).

    Returns an empty DataFrame if ``pandas-datareader`` is absent or the download
    fails — callers should pass their own factor panel in that case rather than rely
    on a network fetch.
    """
    start = start or "1963-07-01"
    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                            start=start, end=end)[0]
        return ff / 100
    except Exception:
        return pd.DataFrame()
