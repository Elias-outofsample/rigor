"""Hidden-options exposure detection + crisis-period decomposition (#13, part).

Equity strategies can carry implicit option-like payoffs (short-vol carry, hidden
gamma/theta). These functions surface them, and stress an edge across historical
crashes:

  * ``detect_vol_exposure`` — correlation of returns with a vol proxy (short/long vol).
  * ``detect_gamma_exposure`` — convexity via a quadratic-in-benchmark regression.
  * ``detect_theta_exposure`` — income vs burst-gain profile from win-rate/autocorr.
  * ``compute_payoff_analysis`` / ``compute_greeks_summary`` — combine into a label.
  * ``decompose_by_crisis`` — per-crisis-window Sharpe; a crash-Sharpe ratio that
    distinguishes vol-risk-premium from convexity from genuinely robust alpha.

scipy only; data frequency inferred from the return index.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .. import metrics as _m

__all__ = [
    "detect_vol_exposure", "detect_gamma_exposure", "detect_theta_exposure",
    "compute_payoff_analysis", "compute_greeks_summary",
    "DEFAULT_CRISIS_WINDOWS", "decompose_by_crisis",
]


def _ppy(returns: pd.Series) -> int:
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        return _m.periods_per_year_of(returns.index)
    return _m.TRADING_DAYS


def _ppy_thresholds(ppy: int) -> tuple[float, float]:
    """(t_threshold, p_threshold) tuned for data frequency — looser CI when sparse."""
    return (1.96, 0.05) if ppy >= 60 else (1.28, 0.20)


def detect_vol_exposure(returns: pd.Series, benchmark_returns: pd.Series,
                        vix_data: pd.Series | None = None) -> dict:
    """Detect vol exposure (short_vol / long_vol / neutral) via correlation with a vol proxy."""
    ppy = _ppy(returns)
    _t, p_thresh = _ppy_thresholds(ppy)
    vol_window = max(4, int(round(21 * ppy / 252)))
    if vix_data is not None:
        vol_proxy = vix_data.reindex(returns.index).ffill().dropna()
    else:
        vol_proxy = benchmark_returns.rolling(vol_window).std() * np.sqrt(ppy)

    aligned = pd.DataFrame({"r": returns, "vol": vol_proxy}).dropna()
    if len(aligned) < 50:
        return {"exposure": "neutral", "correlation": 0.0}

    corr, p = pearsonr(aligned["r"], aligned["vol"])
    corr_thresh = 0.15 if ppy >= 60 else 0.10
    if corr < -corr_thresh and p < p_thresh:
        exposure = "short_vol"
    elif corr > corr_thresh and p < p_thresh:
        exposure = "long_vol"
    else:
        exposure = "neutral"

    median_vol = aligned["vol"].median()
    low = aligned[aligned["vol"] <= median_vol]["r"]
    high = aligned[aligned["vol"] > median_vol]["r"]
    vol_premium = float(np.mean(low)) * ppy - float(np.mean(high)) * ppy
    return {"exposure": exposure, "correlation": float(corr), "p_value": float(p),
            "vol_risk_premium": float(vol_premium)}


def detect_gamma_exposure(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Detect gamma (convexity) via ``r = a + b·r_b + c·r_b²``; c>0 long, c<0 short gamma."""
    aligned = pd.DataFrame({"r": returns, "b": benchmark_returns}).dropna()
    if len(aligned) < 50:
        return {"gamma": 0.0, "exposure": "neutral"}
    ppy = _ppy(returns)
    t_thresh, _ = _ppy_thresholds(ppy)
    y = aligned["r"].to_numpy()
    x = aligned["b"].to_numpy()
    X = np.column_stack([np.ones(len(y)), x, x**2])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coeffs
    n = len(y)
    sigma = np.sqrt(np.sum(resid**2) / (n - 3))
    se = sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    t_stat = coeffs[2] / se[2] if se[2] > 0 else 0.0
    if coeffs[2] > 0 and abs(t_stat) > t_thresh:
        exposure = "long_gamma"
    elif coeffs[2] < 0 and abs(t_stat) > t_thresh:
        exposure = "short_gamma"
    else:
        exposure = "neutral"
    return {"gamma": float(coeffs[2]), "gamma_t_stat": float(t_stat), "exposure": exposure,
            "beta": float(coeffs[1]), "alpha": float(coeffs[0]), "t_threshold": float(t_thresh)}


def detect_theta_exposure(returns: pd.Series) -> dict:
    """Detect theta-like profile: positive (steady income) vs negative (burst gains)."""
    r = returns.dropna().to_numpy().astype(np.float64)
    if len(r) < 50:
        return {"exposure": "neutral"}
    win_rate = float(np.mean(r > 0))
    ac1 = float(np.corrcoef(r[1:], r[:-1])[0, 1])
    if win_rate > 0.55 and abs(ac1) < 0.10:
        exposure = "positive_theta"
    elif win_rate < 0.45 and ac1 > 0:
        exposure = "negative_theta"
    else:
        exposure = "neutral"
    return {"exposure": exposure, "win_rate": win_rate, "autocorrelation_lag1": ac1}


_PAYOFF_MAP = {
    ("short_gamma", "positive_theta"): "covered_call",
    ("short_gamma", "neutral"): "short_straddle",
    ("short_gamma", "negative_theta"): "short_put",
    ("long_gamma", "negative_theta"): "long_straddle",
    ("long_gamma", "neutral"): "collar",
    ("long_gamma", "positive_theta"): "long_call",
    ("neutral", "positive_theta"): "income",
    ("neutral", "negative_theta"): "trend_following",
    ("neutral", "neutral"): "linear",
}


def compute_payoff_analysis(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Combine gamma + theta into an option-payoff label."""
    gamma = detect_gamma_exposure(returns, benchmark_returns)
    theta = detect_theta_exposure(returns)
    payoff = _PAYOFF_MAP.get((gamma["exposure"], theta["exposure"]), "unclassified")
    confidence = min(1.0, abs(gamma.get("gamma_t_stat", 0.0)) / 3.0)
    return {"payoff_type": payoff, "gamma_exposure": gamma["exposure"],
            "theta_exposure": theta["exposure"], "confidence": float(confidence)}


def _stars_p(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _stars_t(t: float, df: int = 1000) -> str:
    from scipy.stats import t as t_dist
    return _stars_p(2.0 * float(t_dist.sf(abs(t), df)))


_LABELS = {"short_vol": "Short", "long_vol": "Long", "neutral": "Neutral",
           "short_gamma": "Short", "long_gamma": "Long",
           "positive_theta": "Positive", "negative_theta": "Negative"}


def _updown_beta(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    a = pd.DataFrame({"r": returns, "b": benchmark_returns}).dropna()
    if len(a) < 30:
        return 0.0, 0.0
    up, down = a[a["b"] > 0], a[a["b"] < 0]

    def _beta(d: pd.DataFrame) -> float:
        if len(d) <= 2 or d["b"].var() <= 0:
            return 0.0
        return float(np.cov(d["r"], d["b"])[0, 1] / d["b"].var())

    return _beta(up), _beta(down)


def compute_greeks_summary(returns: pd.Series, benchmark_returns: pd.Series,
                           vix_data: pd.Series | None = None) -> dict:
    """One-call vol/gamma/theta/payoff profile with a human-readable summary line."""
    vol = detect_vol_exposure(returns, benchmark_returns, vix_data)
    gamma = detect_gamma_exposure(returns, benchmark_returns)
    theta = detect_theta_exposure(returns)
    payoff = compute_payoff_analysis(returns, benchmark_returns)

    vol_label = (f"{_LABELS.get(vol.get('exposure', 'neutral'), 'Neutral')} vol "
                 f"(rho={vol.get('correlation', 0):+.2f}{_stars_p(vol.get('p_value', 1))})")
    n_obs = len(returns.dropna())
    gamma_label = (f"{_LABELS.get(gamma.get('exposure', 'neutral'), 'Neutral')} gamma "
                   f"(coeff={gamma.get('gamma', 0):+.2f}, t={gamma.get('gamma_t_stat', 0):+.1f}"
                   f"{_stars_t(gamma.get('gamma_t_stat', 0), max(n_obs - 3, 1))})")
    ac1 = theta.get("autocorrelation_lag1", 0)
    theta_label = (f"{_LABELS.get(theta.get('exposure', 'neutral'), 'Neutral')} theta "
                   f"(WR={theta.get('win_rate', 0):.0%}, ac1={ac1:+.3f})")

    ub, db = _updown_beta(returns, benchmark_returns)
    trend = "positive" if ub > db + 0.2 else "negative" if db > ub + 0.2 else "neutral"
    trend_label = f"Trend convexity: {trend} (up_beta={ub:+.2f}, down_beta={db:+.2f})"
    return {"vol_exposure": vol, "gamma_exposure": gamma, "theta_exposure": theta,
            "payoff": payoff, "trend_convexity": trend,
            "overall_profile": f"{vol_label} | {gamma_label} | {theta_label} | {trend_label}"}


# --- crisis decomposition --------------------------------------------------

DEFAULT_CRISIS_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("dotcom_crash", "2000-03-01", "2002-10-31"),
    ("gfc", "2007-10-01", "2009-03-31"),
    ("eu_sovereign", "2011-07-01", "2011-12-31"),
    ("china_devaluation", "2015-08-01", "2016-02-29"),
    ("q4_2018", "2018-10-01", "2018-12-31"),
    ("covid_crash", "2020-02-15", "2020-04-30"),
    ("rates_2022", "2022-01-01", "2022-10-31"),
    ("regional_banks_2023", "2023-03-01", "2023-05-31"),
)


def _annual_sharpe(returns: pd.Series, ppy: int) -> float:
    r = returns.dropna()
    if len(r) < 5:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if np.isfinite(sd) and sd > 0 else float("nan")


def _max_dd(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def decompose_by_crisis(returns: pd.Series, benchmark: pd.Series | None = None,
                        windows: Iterable[tuple[str, str, str]] = DEFAULT_CRISIS_WINDOWS,
                        ppy: int = 252) -> dict:
    """Per-crisis-window Sharpe; crash-Sharpe ratio flags vol-premium vs convexity vs robust."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        except Exception:
            return {"error": "returns must have a datetime index", "verdict": "INSUFFICIENT_DATA"}

    full = _annual_sharpe(returns, ppy)
    if not np.isfinite(full):
        return {"error": "full-sample Sharpe undefined", "verdict": "INSUFFICIENT_DATA"}

    crises: list[dict[str, Any]] = []
    sharpes: list[float] = []
    for label, start, end in windows:
        try:
            mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        except Exception:
            continue
        sub = returns.loc[mask]
        if len(sub) < 5:
            crises.append({"label": label, "start": start, "end": end, "n_obs": int(len(sub)),
                           "sharpe": float("nan"), "outside_sample": True})
            continue
        sh = _annual_sharpe(sub, ppy)
        bench_ret = float("nan")
        if benchmark is not None:
            try:
                b = benchmark.copy()
                if not isinstance(b.index, pd.DatetimeIndex):
                    b.index = pd.to_datetime(b.index)
                bsub = b.loc[mask] if mask.any() else b.iloc[0:0]
                if len(bsub) >= 5:
                    bench_ret = float((1.0 + bsub).prod() - 1.0)
            except Exception:
                bench_ret = float("nan")
        crises.append({
            "label": label, "start": start, "end": end, "n_obs": int(len(sub)),
            "sharpe": round(sh, 4) if np.isfinite(sh) else None,
            "max_dd": round(_max_dd(sub), 4),
            "total_return": round(float((1.0 + sub).prod() - 1.0), 4),
            "benchmark_return": round(bench_ret, 4) if np.isfinite(bench_ret) else None,
            "outside_sample": False,
        })
        if np.isfinite(sh):
            sharpes.append(sh)

    if not sharpes:
        return {"full_sample_sharpe": round(full, 4), "crises": crises, "worst_crisis": None,
                "crash_sharpe_ratio": None, "verdict": "INSUFFICIENT_DATA"}

    mean_crisis = float(np.mean(sharpes))
    crash_ratio = mean_crisis / full if full != 0 else float("nan")
    worst = min((c for c in crises if c.get("sharpe") is not None),
                key=lambda c: c["sharpe"], default=None)
    if not np.isfinite(crash_ratio):
        verdict = "INSUFFICIENT_DATA"
    elif full <= 0:
        verdict = "NEGATIVE_EDGE"
    elif crash_ratio > 1.5:
        verdict = "CONVEXITY_CANDIDATE"
    elif crash_ratio >= 0.7:
        verdict = "ROBUST_TO_CRISES"
    elif crash_ratio >= 0.3:
        verdict = "MIXED"
    else:
        verdict = "VOL_RISK_PREMIUM_CANDIDATE"
    return {"full_sample_sharpe": round(full, 4), "mean_crisis_sharpe": round(mean_crisis, 4),
            "crash_sharpe_ratio": round(crash_ratio, 4) if np.isfinite(crash_ratio) else None,
            "worst_crisis": worst, "n_crises_evaluated": len(sharpes),
            "crises": crises, "verdict": verdict}
