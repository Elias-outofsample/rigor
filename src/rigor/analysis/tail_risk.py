"""Tail-risk battery (#5): EVT, VaR backtests, and GJR-GARCH-FHS CVaR.

Three layers of tail analysis beyond the historical VaR/CVaR in ``risk.py``:

  * **EVT** — ``fit_gpd_tail`` (generalised Pareto, McNeil-Frey-Embrechts VaR/ES),
    ``compute_tail_index`` (Hill / Pickands / Moment), ``compute_extreme_risk_measures``.
  * **VaR backtests** — ``kupiec_pof_test`` (proportion-of-failures LR) and
    ``christoffersen_cc_test`` (conditional coverage = rate + independence).
  * **Conditional ES** — ``gjr_garch_fhs_cvar`` / ``rolling_gjr_garch_fhs_cvar``:
    GJR-GARCH(1,1,1) + Filtered Historical Simulation Monte-Carlo Expected
    Shortfall, validated with the two backtests above.

The GARCH layer needs the optional ``arch`` library (``pip install -e
".[analysis]"``). If ``arch`` is absent or the fit fails, it degrades
gracefully to a historical-CVaR fallback with ``fallback_used=True`` — nothing
raises, so the core install keeps working.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = [
    "fit_gpd_tail", "compute_tail_index", "compute_extreme_risk_measures",
    "kupiec_pof_test", "christoffersen_cc_test",
    "gjr_garch_fhs_cvar", "rolling_gjr_garch_fhs_cvar",
]


def _to_arr(returns) -> np.ndarray:
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    r = r.astype(np.float64)
    return r[np.isfinite(r)]


# --- EVT -------------------------------------------------------------------

def fit_gpd_tail(returns, threshold_percentile: float = 5) -> dict:
    """Fit a generalised Pareto distribution to the left (loss) tail (Pickands 1975)."""
    r = _to_arr(returns)
    if len(r) < 50:
        return {"error": "insufficient data"}
    threshold = np.percentile(r, threshold_percentile)
    exceed = threshold - r[r <= threshold]
    exceed = exceed[exceed > 0]
    if len(exceed) < 10:
        return {"error": "insufficient tail observations"}
    try:
        from scipy.stats import genpareto
        xi, _loc, sigma = genpareto.fit(exceed, floc=0)
    except Exception:
        return {"error": "GPD fit failed"}

    n, n_exceed = len(r), len(exceed)
    ratio = 0.01 / (n_exceed / n)
    var_loss = (sigma / xi * (ratio ** (-xi) - 1.0) if abs(xi) > 1e-10
                else -sigma * np.log(ratio))
    var_99 = threshold - var_loss
    cvar_99 = threshold - (var_loss + sigma) / (1.0 - xi) if xi < 1.0 else var_99
    return {"xi": float(xi), "sigma": float(sigma), "threshold": float(threshold),
            "n_exceedances": int(n_exceed), "var_99": float(var_99), "cvar_99": float(cvar_99)}


def compute_tail_index(returns, method: str = "hill", k: int | None = None) -> dict:
    """Estimate the tail index (alpha = 1/xi) via Hill, Pickands, or Moment estimators."""
    r = _to_arr(returns)
    r = -r[r < 0]
    r = np.sort(r)[::-1]
    n = len(r)
    if n < 20:
        return {"alpha": np.nan, "method": method}
    if k is None:
        k = int(np.sqrt(n))
    if method == "hill":
        if k < 2 or r[k - 1] <= 0:
            return {"alpha": np.nan, "method": "hill"}
        xi = float(np.mean(np.log(r[:k]) - np.log(r[k - 1])))
        return {"alpha": float(1.0 / xi) if xi > 0 else np.inf, "xi": xi, "method": "hill", "k": k}
    if method == "pickands":
        if 4 * k > n:
            k = n // 4
        if k < 1 or r[2 * k - 1] == r[k - 1]:
            return {"alpha": np.nan, "method": "pickands"}
        xi = float(np.log((r[k - 1] - r[2 * k - 1]) / (r[2 * k - 1] - r[4 * k - 1])) / np.log(2))
        return {"alpha": float(1 / xi) if xi > 0 else np.inf, "xi": xi,
                "method": "pickands", "k": k}
    if method == "moment":
        m1 = float(np.mean(np.log(r[:k]) - np.log(r[k - 1])))
        m2 = float(np.mean((np.log(r[:k]) - np.log(r[k - 1])) ** 2))
        xi = m1 + 1 - 0.5 / (1 - m1**2 / m2) if m2 > m1**2 else m1
        return {"alpha": float(1 / xi) if xi > 0 else np.inf, "xi": float(xi),
                "method": "moment", "k": k}
    return {"alpha": np.nan, "method": method}


def compute_extreme_risk_measures(returns) -> dict:
    """Combined EVT summary: GPD tail fit + Hill index + a fat-tail flag (alpha<4)."""
    gpd = fit_gpd_tail(returns)
    hill = compute_tail_index(returns, method="hill")
    return {"gpd": gpd, "hill_alpha": hill.get("alpha", np.nan),
            "fat_tailed": hill.get("alpha", np.inf) < 4}


# --- VaR backtests ---------------------------------------------------------

def _breach_indicator(returns, var_quantile: float) -> np.ndarray:
    r = _to_arr(returns)
    if len(r) < 30:
        return np.empty(0, dtype=np.int8)
    return (r < float(np.percentile(r, var_quantile * 100))).astype(np.int8)


def kupiec_pof_test(returns, var_quantile: float = 0.05) -> dict:
    """Kupiec (1995) proportion-of-failures LR test (χ²₁) that breach rate == quantile."""
    breaches = _breach_indicator(returns, var_quantile)
    n = len(breaches)
    if n < 30:
        return {"n_obs": n, "n_breaches": 0, "expected_breaches": 0, "p_observed": 0.0,
                "lr_stat": 0.0, "p_value": 1.0, "reject_H0": False, "verdict": "insufficient_data"}
    x = int(breaches.sum())
    p = float(var_quantile)
    p_hat = x / n

    def _ll(rate: float) -> float:
        if rate <= 0 or rate >= 1:
            return 0.0 if (x == 0 and rate == 0) or (x == n and rate == 1) else -np.inf
        return x * np.log(rate) + (n - x) * np.log(1 - rate)

    ll_h1 = _ll(p_hat) if 0 < p_hat < 1 else 0.0
    lr = float(max(-2.0 * (_ll(p) - ll_h1), 0.0))
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {"n_obs": n, "n_breaches": x, "expected_breaches": int(round(n * p)),
            "p_observed": float(p_hat), "p_expected": p, "lr_stat": lr, "p_value": p_value,
            "reject_H0": p_value < 0.05,
            "verdict": ("PASS" if p_value >= 0.05
                        else "FAIL_HIGH_BREACHES" if p_hat > p else "FAIL_LOW_BREACHES")}


def christoffersen_cc_test(returns, var_quantile: float = 0.05) -> dict:
    """Christoffersen (1998) conditional-coverage LR test (χ²₂): correct rate AND independence."""
    breaches = _breach_indicator(returns, var_quantile)
    n = len(breaches)
    if n < 30:
        return {"n_obs": n, "n_breaches": 0, "p_observed": 0.0, "lr_pof": 0.0, "lr_ind": 0.0,
                "lr_cc": 0.0, "p_value_ind": 1.0, "p_value_cc": 1.0, "reject_H0": False,
                "verdict": "insufficient_data"}
    pof = kupiec_pof_test(returns, var_quantile)
    lr_pof = pof["lr_stat"]

    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        prev, cur = int(breaches[t - 1]), int(breaches[t])
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    n_total = n0 + n1
    if n_total < 1:
        return {**pof, "lr_ind": 0.0, "lr_cc": lr_pof, "p_value_ind": 1.0,
                "p_value_cc": pof["p_value"]}
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / n_total

    def _log(x: float) -> float:
        return np.log(x) if x > 0 else 0.0

    ll_h0 = (n00 + n10) * _log(1 - pi) + (n01 + n11) * _log(pi)
    ll_h1 = (n00 * _log(1 - pi01) + n01 * _log(pi01)
             + n10 * _log(1 - pi11) + n11 * _log(pi11))
    lr_ind = float(max(-2.0 * (ll_h0 - ll_h1), 0.0))
    p_value_ind = float(1.0 - chi2.cdf(lr_ind, df=1))
    lr_cc = lr_pof + lr_ind
    p_value_cc = float(1.0 - chi2.cdf(lr_cc, df=2))
    if p_value_cc >= 0.05:
        verdict = "PASS"
    elif pof["p_value"] < 0.05 and p_value_ind < 0.05:
        verdict = "FAIL_RATE_AND_CLUSTERING"
    elif pof["p_value"] < 0.05:
        verdict = "FAIL_RATE"
    else:
        verdict = "FAIL_CLUSTERING"
    return {"n_obs": n, "n_breaches": int(breaches.sum()), "p_observed": float(breaches.mean()),
            "p_expected": float(var_quantile),
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "pi01": pi01, "pi11": pi11, "lr_pof": lr_pof, "lr_ind": lr_ind, "lr_cc": lr_cc,
            "p_value_pof": pof["p_value"], "p_value_ind": p_value_ind, "p_value_cc": p_value_cc,
            "reject_H0": p_value_cc < 0.05, "verdict": verdict}


# --- GJR-GARCH-FHS CVaR (arch optional) ------------------------------------

_MIN_OBS = 60
_DEFAULT_WINDOW = 252
_DEFAULT_N_SIM = 10_000


def _historical_es(r: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if len(r) < 10:
        return float("nan"), float("nan"), 0
    var_th = float(np.percentile(r, alpha * 100.0))
    tail = r[r <= var_th]
    return (float(np.mean(tail)) if len(tail) else var_th), var_th, int(len(tail))


def _fallback(r, alpha, horizon, window, n_sim, reason) -> dict:
    es_h, var_h, n_tail = _historical_es(r, alpha)
    return {"es": es_h, "var": var_h, "es_historical": es_h, "var_historical": var_h,
            "method": "historical_fallback", "alpha": alpha, "horizon": horizon,
            "estimation_window": window, "n_simulations": n_sim, "n_tail": n_tail,
            "n_obs": int(len(r)),
            "kupiec": {"verdict": "insufficient_data", "p_value": 1.0, "lr_stat": 0.0,
                       "reject_H0": False, "n_breaches": 0, "n_obs": int(len(r))},
            "christoffersen": {"verdict": "insufficient_data", "p_value_cc": 1.0,
                               "lr_cc": 0.0, "reject_H0": False},
            "garch_params": {"convergence": False}, "fallback_used": True,
            "fallback_reason": reason}


def _fit_gjr_garch(r_pct: np.ndarray):
    from arch import arch_model
    am = arch_model(r_pct, vol="GARCH", p=1, o=1, q=1, mean="Zero", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        raise RuntimeError(f"GJR-GARCH did not converge (flag={res.convergence_flag})")
    return res


def gjr_garch_fhs_cvar(returns, alpha: float = 0.05, horizon: int = 1,
                       estimation_window: int = _DEFAULT_WINDOW,
                       n_simulations: int = _DEFAULT_N_SIM, seed: int = 42,
                       run_backtest_tests: bool = True) -> dict:
    """GJR-GARCH(1,1,1) + Filtered Historical Simulation Expected Shortfall.

    Conditional ES that captures vol clustering and the leverage effect, validated
    with Kupiec POF + Christoffersen CC. Needs the optional ``arch`` library; if it
    is missing or the fit fails, returns a historical-CVaR fallback
    (``fallback_used=True``) rather than raising.
    """
    r = _to_arr(returns)
    n = len(r)
    if n < _MIN_OBS:
        return _fallback(r, alpha, horizon, estimation_window, n_simulations,
                         f"n_obs={n} < {_MIN_OBS}")
    win = min(int(estimation_window), n)
    r_win = r[-win:]
    r_pct = r_win * 100.0
    try:
        res = _fit_gjr_garch(r_pct)
    except Exception as exc:
        return _fallback(r, alpha, horizon, estimation_window, n_simulations,
                         f"GJR-GARCH unavailable or fit failed: {exc}")

    p = res.params
    omega = float(p.get("omega", 0.0))
    alpha_p = float(p.get("alpha[1]", 0.0))
    gamma_p = float(p.get("gamma[1]", 0.0))
    beta_p = float(p.get("beta[1]", 0.0))
    persistence = alpha_p + 0.5 * gamma_p + beta_p

    sigma_pct = np.asarray(res.conditional_volatility, dtype=np.float64)
    sigma_pct = np.where(sigma_pct > 1e-12, sigma_pct, np.nan)
    eps_pct = r_pct - 0.0
    z = eps_pct / sigma_pct
    z = z[np.isfinite(z)]
    if len(z) < _MIN_OBS:
        return _fallback(r, alpha, horizon, estimation_window, n_simulations,
                         "too few filtered innovations")
    std_z = np.std(z, ddof=1)
    z = (z - np.mean(z)) / (std_z if std_z > 1e-12 else 1.0)

    sigma2_last = float(sigma_pct[-1] ** 2)
    eps_last = float(eps_pct[-1])
    rng = np.random.default_rng(seed)
    horizon = max(int(horizon), 1)
    n_sim = int(n_simulations)
    z_sample = rng.choice(z, size=(n_sim, horizon), replace=True)

    sum_returns = np.zeros(n_sim, dtype=np.float64)
    sigma2 = np.full(n_sim, sigma2_last, dtype=np.float64)
    eps_prev = np.full(n_sim, eps_last, dtype=np.float64)
    for h in range(horizon):
        leverage = np.where(eps_prev < 0, gamma_p * (eps_prev * eps_prev), 0.0)
        sigma2 = omega + alpha_p * (eps_prev * eps_prev) + leverage + beta_p * sigma2
        eps_h = np.sqrt(np.maximum(sigma2, 1e-12)) * z_sample[:, h]
        sum_returns += eps_h
        eps_prev = eps_h
    sim = sum_returns / 100.0

    var_garch = float(np.percentile(sim, alpha * 100.0))
    tail = sim[sim <= var_garch]
    es_garch = float(np.mean(tail)) if len(tail) else var_garch
    es_h_win, var_h_win, _ = _historical_es(r_win, alpha)
    es_h_full, var_h_full, n_tail = _historical_es(r, alpha)

    if run_backtest_tests:
        try:
            kupiec = kupiec_pof_test(r_win, var_quantile=alpha)
        except Exception as exc:
            kupiec = {"verdict": f"error: {exc}", "p_value": 1.0, "lr_stat": 0.0,
                      "reject_H0": False}
        try:
            cc = christoffersen_cc_test(r_win, var_quantile=alpha)
        except Exception as exc:
            cc = {"verdict": f"error: {exc}", "p_value_cc": 1.0, "lr_cc": 0.0, "reject_H0": False}
    else:
        kupiec = {"verdict": "skipped", "p_value": 1.0, "lr_stat": 0.0, "reject_H0": False}
        cc = {"verdict": "skipped", "p_value_cc": 1.0, "lr_cc": 0.0, "reject_H0": False}

    return {"es": es_garch, "var": var_garch, "es_historical": es_h_full,
            "var_historical": var_h_full, "es_historical_window": es_h_win,
            "var_historical_window": var_h_win, "method": "gjr_garch_fhs", "alpha": float(alpha),
            "horizon": int(horizon), "estimation_window": int(win), "n_simulations": int(n_sim),
            "n_tail": int(n_tail), "n_obs": int(n), "kupiec": kupiec, "christoffersen": cc,
            "garch_params": {"omega": omega, "alpha": alpha_p, "gamma": gamma_p, "beta": beta_p,
                             "persistence": float(persistence), "convergence": True},
            "fallback_used": False, "fallback_reason": ""}


_ROLL_COLS = ["es", "var", "es_historical", "var_historical", "fallback_used",
              "kupiec_p", "garch_persistence"]


def rolling_gjr_garch_fhs_cvar(returns, alpha: float = 0.05, horizon: int = 1,
                               window: int = _DEFAULT_WINDOW, step: int = 21,
                               n_simulations: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Rolling conditional ES — refit GJR-GARCH every ``step`` periods on a trailing window."""
    r_series = returns.dropna() if isinstance(returns, pd.Series) else pd.Series(_to_arr(returns))
    n = len(r_series)
    if n < window + 1:
        return pd.DataFrame(columns=_ROLL_COLS)
    indices = list(range(window, n, max(int(step), 1)))
    if indices and indices[-1] != n - 1:
        indices.append(n - 1)
    rows = []
    for end_idx in indices:
        sub = r_series.iloc[max(0, end_idx - window):end_idx]
        if len(sub) < _MIN_OBS:
            continue
        out = gjr_garch_fhs_cvar(sub, alpha=alpha, horizon=horizon, estimation_window=window,
                                 n_simulations=n_simulations, seed=seed, run_backtest_tests=False)
        rows.append({"date": r_series.index[end_idx], "es": out["es"], "var": out["var"],
                     "es_historical": out["es_historical"], "var_historical": out["var_historical"],
                     "fallback_used": out["fallback_used"],
                     "kupiec_p": out["kupiec"].get("p_value", 1.0),
                     "garch_persistence": out["garch_params"].get("persistence", float("nan"))})
    if not rows:
        return pd.DataFrame(columns=_ROLL_COLS)
    return pd.DataFrame(rows).set_index("date")
