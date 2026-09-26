"""Single-source-of-truth performance metrics.

Every part of the framework computes Sharpe/CAGR/drawdown HERE and nowhere else,
so two runs never report different numbers for the same returns. Conventions
(adopted from an earlier in-house library, which had the cleanest definitions):

  * Sharpe uses sample std (ddof=1) and annualises by sqrt(periods_per_year).
  * CAGR / max-drawdown are computed in log space for overflow safety.
  * Risk-free rate defaults to 0 (report excess separately when needed).

Pure numpy + stdlib — deterministic on every machine (no Numba, no threads, no RNG).
``statistics.NormalDist`` (Python 3.8+) is used for the Gaussian VaR helper so
this module stays free of optional C-extension dependencies (no scipy).
"""

from __future__ import annotations

import statistics
import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    return r[np.isfinite(r)]


def compute_sharpe(returns, periods_per_year: int = TRADING_DAYS, risk_free: float = 0.0) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def compute_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def compute_cagr(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    growth = 1.0 + r
    if np.any(growth <= 0):  # a -100% (or worse) bar wipes the curve
        warnings.warn(
            "compute_cagr: a return <= -100% wipes the equity curve; CAGR is undefined (NaN)",
            RuntimeWarning, stacklevel=2,
        )
        return float("nan")
    log_growth = np.log(growth).sum()
    years = r.size / periods_per_year
    if years <= 0:
        return 0.0
    return float(np.exp(log_growth / years) - 1.0)


def compute_equity_curve(returns) -> np.ndarray:
    r = _clean(returns)
    return np.cumprod(1.0 + r)


def compute_max_drawdown(returns) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    # Log space (log1p / cumsum / expm1): a long winning streak can't overflow
    # cumprod to +inf and silently turn every drawdown into NaN. Numerically
    # identical to the direct equity ratio for realistic curves.
    log_cum = np.cumsum(np.log1p(r))
    dd = np.expm1(log_cum - np.maximum.accumulate(log_cum))
    return float(dd.min())


def compute_sortino(returns, periods_per_year: int = TRADING_DAYS, target: float = 0.0) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - target / periods_per_year, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return 0.0
    return float((r.mean() - target / periods_per_year) / dd * np.sqrt(periods_per_year))


def compute_calmar(returns, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(compute_max_drawdown(returns))
    if mdd < 1e-12:
        return 0.0
    cagr = compute_cagr(returns, periods_per_year)
    return float(cagr / mdd) if np.isfinite(cagr) else 0.0


def compute_win_rate(returns) -> float:
    r = _clean(returns)
    traded = r[r != 0]
    return float((traded > 0).mean()) if traded.size else 0.0


def compute_expected_value(returns, periods_per_year: int = TRADING_DAYS) -> dict:
    """Arithmetic expected value of the per-period return.

    ``ev``        — mean return per bar (the per-period arithmetic expectation).
    ``ev_annual`` — annualised arithmetic mean = ev * periods_per_year.

    This complements CAGR (geometric mean): EV is the honest "expected return per
    year if periods were independent" — the standard trading expectancy generalised
    to a continuous return series.  NaN-safe; empty or all-NaN series → 0.0.
    """
    r = _clean(returns)
    if r.size == 0:
        return {"ev": 0.0, "ev_annual": 0.0}
    ev = float(r.mean())
    return {"ev": ev, "ev_annual": ev * periods_per_year}


def compute_expected_excess_return(
    returns,
    periods_per_year: int = TRADING_DAYS,
    risk_free_annual: float = 0.0,
) -> dict:
    """Arithmetic excess return over the risk-free rate — the alpha/excess-return metric.

    ``ev_excess``        — per-period arithmetic excess = ev − risk_free_annual/periods_per_year.
    ``ev_excess_annual`` — annualised arithmetic excess = ev_annual − risk_free_annual.

    This is the arithmetic analogue of Sharpe's numerator (before dividing by vol).
    A positive ``ev_excess_annual`` means the strategy earns more than cash on average,
    independent of how volatile that outperformance is.

    Relationship to other metrics
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``ev_annual``        — total arithmetic return (includes the risk-free component).
    * ``ev_excess_annual`` — alpha component (return above cash), i.e. what the strategy
                             adds beyond parking money in T-bills.
    * Sharpe              — ``ev_excess_annual / volatility_annual`` (risk-adjusted).

    ``risk_free_annual`` should be the annualised risk-free rate for the same period as
    the returns (e.g. annualised DGS3MO average).  Defaults to 0.0 so that at the
    default ``compute_core_metrics(returns)`` call ``ev_excess_annual == ev_annual``
    — all existing numbers remain byte-identical.

    NOTE: The pipeline currently passes ``risk_free_annual=0.0`` (default).  To wire
    a true risk-free series (e.g. period-average annualised DGS3MO from FRED), compute
    its mean annual rate for the backtest window and pass it here via
    ``compute_core_metrics(..., risk_free_annual=rf_ann)``.  That plumbing is left to
    the caller; this function is already correct for any non-zero rf.
    """
    r = _clean(returns)
    if r.size == 0:
        return {"ev_excess": 0.0, "ev_excess_annual": 0.0}
    ev_vals = compute_expected_value(r, periods_per_year)
    ev_annual = ev_vals["ev_annual"]
    ev_excess_annual = ev_annual - risk_free_annual
    ev_excess = ev_excess_annual / periods_per_year
    return {"ev_excess": float(ev_excess), "ev_excess_annual": float(ev_excess_annual)}


def _gaussian_var_inline(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Inline delta-normal (Gaussian) VaR: −(μ + z_alpha·σ) as a negative loss magnitude.

    Agrees with ``rigor.analysis.risk.gaussian_var`` — kept here to avoid importing
    optional-dependency analysis modules into the always-imported metrics core.
    Uses ``statistics.NormalDist`` (Python 3.8+ stdlib; no scipy required).

    Returns a negative number consistent with the sign convention of historical VaR
    in ``rigor.analysis.risk.value_at_risk`` (which returns ``np.percentile(r, alpha*100)``
    — a negative value for typical loss quantiles).
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z_alpha = statistics.NormalDist().inv_cdf(alpha)
    return float(mu + z_alpha * sigma)


def compute_skewness(returns) -> float:
    """Fisher-Pearson sample skewness (0 for a symmetric distribution)."""
    r = _clean(returns)
    if r.size < 3:
        return 0.0
    d = r - r.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(d**3) / m2**1.5)


def compute_kurtosis(returns) -> float:
    """Excess kurtosis (0 for a normal distribution; > 0 = fat tails)."""
    r = _clean(returns)
    if r.size < 4:
        return 0.0
    d = r - r.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(d**4) / m2**2 - 3.0)


def compute_core_metrics(
    returns,
    periods_per_year: int = TRADING_DAYS,
    risk_free_annual: float = 0.0,
) -> dict:
    """The canonical metrics dict used by the report card and selection.

    Parameters
    ----------
    returns:
        Array-like of per-period arithmetic returns.
    periods_per_year:
        Annualisation factor (252 for daily, 52 for weekly, 12 for monthly).
    risk_free_annual:
        Annualised risk-free rate used to compute ``ev_excess_annual``.
        Defaults to 0.0 (backward-compatible — all pre-existing keys are
        byte-identical at the default).  Pass the period-average annualised
        DGS3MO (or equivalent) to get a true excess-return figure.

    New keys added (existing keys unchanged at default rf=0)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``ev_excess_annual`` — annualised arithmetic excess return over risk-free
                           (= ev_annual when risk_free_annual=0.0).
    ``var_normal_95``    — delta-normal (Gaussian) VaR at 95% confidence
                           (= −(μ + z_{0.05}·σ)); negative = loss magnitude,
                           consistent with historical VaR sign convention.
    ``skew`` / ``kurtosis`` — sample skewness and *excess* kurtosis of the
                           per-period returns (0 / 0 for a normal distribution).
    """
    r = _clean(returns)
    equity = np.cumprod(1.0 + r) if r.size else np.array([1.0])
    ev_vals = compute_expected_value(r, periods_per_year)
    exc_vals = compute_expected_excess_return(r, periods_per_year, risk_free_annual)
    return {
        "n_obs": int(r.size),
        "sharpe": compute_sharpe(r, periods_per_year),
        "sortino": compute_sortino(r, periods_per_year),
        "calmar": compute_calmar(r, periods_per_year),
        "cagr": compute_cagr(r, periods_per_year),
        "volatility": compute_volatility(r, periods_per_year),
        "max_drawdown": compute_max_drawdown(r),
        "total_return": float(equity[-1] - 1.0) if r.size else 0.0,
        "win_rate": compute_win_rate(r),
        "ev": ev_vals["ev"],
        "ev_annual": ev_vals["ev_annual"],
        # --- new keys (added below existing for backward-compatibility) ---
        "ev_excess_annual": exc_vals["ev_excess_annual"],
        "var_normal_95": _gaussian_var_inline(r),
        "skew": compute_skewness(r),
        "kurtosis": compute_kurtosis(r),
    }


def periods_per_year_of(index: pd.DatetimeIndex, default: int = TRADING_DAYS) -> int:
    """Infer annualisation factor from a DatetimeIndex spacing (daily/weekly/monthly)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    median_gap = np.median(np.diff(index.values).astype("timedelta64[D]").astype(int))
    if median_gap <= 0:
        return default
    if median_gap <= 2:
        return TRADING_DAYS
    if median_gap <= 9:
        return 52
    if median_gap <= 45:
        return 12
    return 1
