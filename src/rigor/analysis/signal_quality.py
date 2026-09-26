"""Signal quality & the Fundamental Law of Active Management (#6).

Diagnostics on a *signal* (vs its forward returns), not on PnL — so they reveal
whether an edge is real and how fast it decays, independent of sizing:

  * ``rank_ic`` / ``rank_ic_rolling`` / ``icir`` — Spearman information coefficient.
  * ``xi_ic`` — Chatterjee's ξ: detects ANY (incl. non-monotonic) signal→return
    dependence that Spearman rank-IC misses.
  * ``signal_decay`` — fit IC(t)=IC₀·e^(−λt) → **half-life** ln2/λ + optimal horizon.
  * ``ou_halflife`` — Ornstein-Uhlenbeck mean-reversion speed of a level signal.
  * ``net_ic`` — IC after translating turnover×cost into IC space (flags signals
    that die after costs).
  * ``monotonicity`` — do signal deciles line up monotonically with returns?
  * ``variance_ratio`` — Lo-MacKinlay (trending vs mean-reverting).
  * **Fundamental Law** — ``fundamental_ir = IC·√BR·TC`` (Grinold-Kahn), with an
    autocorrelation-adjusted breadth estimate.
  * **Forecast-error metrics** — ``forecast_mae``, ``forecast_rmse``,
    ``forecast_mape``, ``forecast_error_report`` — for evaluating numeric return
    forecasts against realised returns (MAE / RMSE / MAPE).

scipy only (no heavy deps). Forecast-error functions are pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from scipy.stats import t as _t

__all__ = [
    "rank_ic", "xi_ic", "xi_ic_significance",
    "rank_ic_rolling", "icir", "bucket_returns", "signal_turnover",
    "monotonicity", "signal_decay", "ou_halflife", "net_ic", "variance_ratio",
    "breadth_from_ic_autocorr", "fundamental_ir", "icir_flam", "flam_summary",
    # Signal-decay / mean-reversion half-life (guarded, status-returning)
    "compute_signal_decay", "compute_ou_half_life",
    # Forecast-error metrics
    "forecast_mae", "forecast_rmse", "forecast_mape", "forecast_error_report",
]


def _safe_div(a, b, default=0.0):
    return a / b if b not in (0, 0.0) and np.isfinite(b) else default


def rank_ic(signals, forward_returns) -> float:
    """Single-period Spearman rank IC between a signal and its forward return."""
    s = np.asarray(signals, dtype="float64")
    f = np.asarray(forward_returns, dtype="float64")
    m = np.isfinite(s) & np.isfinite(f)
    if m.sum() < 5:
        return 0.0
    c, _ = spearmanr(s[m], f[m])
    return float(c) if np.isfinite(c) else 0.0


def xi_ic(signals, forward_returns) -> float:
    """Chatterjee's ξ_n(signal → forward return) — detects ANY dependence.

    Unlike Spearman rank-IC (which only sees *monotonic* association), ξ → 0
    under independence and → 1 when the forward return is a noiseless function
    of the signal, including non-monotonic ones (e.g. Y = X²). Uses the
    tie-corrected general estimator, so it is robust to repeated values. ξ is
    asymmetric and asymptotically in [0, 1]; a tiny-negative value is expected
    in small samples under independence.
    """
    s = np.asarray(signals, dtype="float64")
    f = np.asarray(forward_returns, dtype="float64")
    m = np.isfinite(s) & np.isfinite(f)
    if m.sum() < 5:
        return 0.0
    s, f = s[m], f[m]
    n = s.size
    order = np.argsort(s, kind="stable")          # sort pairs by X ascending
    y = f[order]
    r = rankdata(y, method="max")                 # r_i = #{ Y_j <= y_(i) }
    le = rankdata(-y, method="max")               # l_i = #{ Y_j >= y_(i) }
    denom = 2.0 * float(np.sum(le * (n - le)))
    if denom == 0.0:                              # constant Y (or degenerate)
        return 0.0
    num = float(n * np.sum(np.abs(np.diff(r))))
    xi = 1.0 - num / denom
    return float(xi) if np.isfinite(xi) else 0.0


def xi_ic_significance(signals, forward_returns) -> dict:
    """One-sided significance of Chatterjee's ξ (test ξ > 0).

    Under independence and no ties, ``sqrt(n)·ξ_n → N(0, 2/5)``, so
    ``p = norm.sf(sqrt(n)·ξ / sqrt(0.4))`` is the asymptotic one-sided p-value.
    Returns ``{"xi": float, "p_value": float, "n": int}``; small samples
    (n < 5) yield ξ 0.0 and p 1.0.
    """
    s = np.asarray(signals, dtype="float64")
    f = np.asarray(forward_returns, dtype="float64")
    m = np.isfinite(s) & np.isfinite(f)
    n = int(m.sum())
    if n < 5:
        return {"xi": 0.0, "p_value": 1.0, "n": n}
    xi = xi_ic(s, f)
    p = float(norm.sf(np.sqrt(n) * xi / np.sqrt(0.4)))
    return {"xi": xi, "p_value": p, "n": n}


def rank_ic_rolling(signals: pd.Series, forward_returns: pd.Series, window: int = 60) -> dict:
    """Rolling IC → mean, std, IC information ratio, and the IC series."""
    if len(signals) < window:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
                "ic_series": pd.Series(dtype="float64")}
    vals = [rank_ic(signals.iloc[i - window:i].to_numpy(),
                    forward_returns.iloc[i - window:i].to_numpy())
            for i in range(window, len(signals) + 1)]
    ser = pd.Series(vals, index=signals.index[window - 1:])
    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"ic_mean": mean, "ic_std": std, "ic_ir": _safe_div(mean, std), "ic_series": ser}


def icir(signals: pd.Series, forward_returns: pd.Series, window: int = 60) -> float:
    """IC information ratio = mean(rolling IC) / std(rolling IC)."""
    return rank_ic_rolling(signals, forward_returns, window)["ic_ir"]


def bucket_returns(signals, forward_returns, n_buckets: int = 5) -> pd.DataFrame:
    """Mean forward return per signal quantile bucket (low→high signal)."""
    s = np.asarray(signals, dtype="float64")
    f = np.asarray(forward_returns, dtype="float64")
    m = np.isfinite(s) & np.isfinite(f)
    s, f = s[m], f[m]
    if len(s) < n_buckets * 2:
        return pd.DataFrame()
    try:
        b = pd.qcut(s, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = [{"bucket": k + 1, "mean_return": float(f[b == k].mean()), "count": int((b == k).sum())}
            for k in range(int(b.max()) + 1) if (b == k).sum() > 0]
    return pd.DataFrame(rows)


def signal_turnover(signal_series: pd.Series) -> float:
    """Average absolute period-over-period change in the signal."""
    return float(signal_series.diff().dropna().abs().mean()) if len(signal_series) > 1 else 0.0


def monotonicity(returns_by_group, ppy: int = 252) -> dict:
    """Spearman rank of group-mean returns vs group order (low→high signal) +
    strict monotonicity, long-short spread, and per-group Sharpes."""
    groups = ([np.asarray(v, dtype="float64") for v in returns_by_group.values()]
              if isinstance(returns_by_group, dict)
              else [np.asarray(v, dtype="float64") for v in returns_by_group])
    n = len(groups)
    if n < 3:
        return {"is_monotonic": False, "reason": "need >= 3 groups"}
    means = [float(g.mean()) for g in groups]
    rho, p = spearmanr(list(range(1, n + 1)), means)
    diffs = np.diff(means)
    sharpes = [float(g.mean() / g.std(ddof=1) * np.sqrt(ppy)) if len(g) > 1 and g.std(ddof=1) > 0
               else 0.0 for g in groups]
    return {
        "is_monotonic": bool(abs(rho) > 0.7 and p < 0.05),
        "is_strictly_monotonic": bool((diffs > 0).all() or (diffs < 0).all()),
        "direction": "positive" if rho > 0 else "negative",
        "spearman_rho": float(rho), "p_value": float(p),
        "long_short_spread": float((means[-1] - means[0]) * ppy),
        "group_means": means, "group_sharpes": sharpes, "n_groups": n,
    }


def signal_decay(ic_by_lag) -> dict:
    """Fit IC(t)=IC₀·e^(−λt) (log-linear OLS) → half-life, decay rate, peak/horizon."""
    if isinstance(ic_by_lag, dict):
        lags = np.array(sorted(ic_by_lag), dtype="float64")
        ics = np.array([ic_by_lag[k] for k in lags], dtype="float64")
    else:
        ics = np.asarray(ic_by_lag, dtype="float64")
        lags = np.arange(1, len(ics) + 1, dtype="float64")
    m = np.isfinite(ics) & (np.abs(ics) > 1e-10)
    if m.sum() < 3:
        return {"half_life": float("nan"), "decay_rate": float("nan"), "peak_ic": 0.0}
    log_ic = np.log(np.abs(ics[m]))
    a = np.column_stack([np.ones(m.sum()), lags[m]])
    coeffs, *_ = np.linalg.lstsq(a, log_ic, rcond=None)
    rate = -coeffs[1]
    fitted = a @ coeffs
    ss_tot = float(np.sum((log_ic - log_ic.mean()) ** 2))
    peak = int(np.argmax(np.abs(ics)))
    return {
        "half_life": float(np.log(2) / rate) if rate > 0 else float("inf"),
        "decay_rate": float(rate), "peak_ic": float(ics[peak]),
        "optimal_horizon": int(lags[peak]),
        "r_squared": float(1.0 - np.sum((log_ic - fitted) ** 2) / ss_tot) if ss_tot > 0 else 0.0,
    }


def ou_halflife(series, ppy: int = 252) -> dict:
    """Ornstein-Uhlenbeck mean-reversion half-life via Δx = a + b·x_{t-1} (κ=−b)."""
    x = np.asarray(series.to_numpy() if isinstance(series, pd.Series) else series, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"half_life": float("nan"), "is_mean_reverting": False}
    dx, xl = np.diff(x), x[:-1]
    a = np.column_stack([np.ones(len(xl)), xl])
    coeffs, *_ = np.linalg.lstsq(a, dx, rcond=None)
    b = coeffs[1]
    kappa = -b
    resid = dx - a @ coeffs
    sigma = float(np.std(resid, ddof=2))
    se_b = sigma * np.sqrt(np.linalg.inv(a.T @ a)[1, 1])
    t_stat = b / se_b if se_b > 0 else 0.0
    p = 2.0 * _t.sf(abs(t_stat), df=len(xl) - 2)
    return {
        "half_life": float(np.log(2) / kappa) if kappa > 0 else float("inf"),
        "kappa": float(kappa), "t_stat": float(t_stat), "p_value": float(p),
        "is_mean_reverting": bool(b < 0 and p < 0.05),
    }


def net_ic(ic_series, turnover: float, cost_bps_per_trade: float) -> dict:
    """Gross vs net IC after translating turnover × cost into IC space; flags a
    signal whose real gross IC is eaten to negative net by costs."""
    ic = np.asarray(ic_series.to_numpy() if isinstance(ic_series, pd.Series) else ic_series,
                    dtype="float64")
    ic = ic[np.isfinite(ic)]
    if not len(ic):
        return {"ic_gross": 0.0, "ic_net": 0.0, "ic_net_negative_flag": False}
    gross, vol = float(ic.mean()), float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    drag = (turnover * cost_bps_per_trade / 10_000.0) / vol if vol > 1e-12 else 0.0
    nett = gross - drag
    return {"ic_gross": gross, "ic_net": nett, "ic_vol": vol, "cost_drag_ic": drag,
            "ic_gross_ir": _safe_div(gross, vol), "ic_net_ir": _safe_div(nett, vol),
            "ic_net_negative_flag": bool(gross > 0.02 and nett < 0.0)}


def variance_ratio(returns, q: int = 5) -> dict:
    """Lo-MacKinlay variance ratio at horizon ``q`` with the heteroskedasticity-
    robust z. VR>1 trending, <1 mean-reverting; |z|>1.96 ⇒ reject random walk."""
    r = np.asarray(returns.to_numpy() if isinstance(returns, pd.Series) else returns,
                   dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < q * 4:
        return {"variance_ratio": 1.0, "z_stat": 0.0, "verdict": "insufficient"}
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")  # q-period sums
    varq = np.sum((rq - q * mu) ** 2) / (q * (n - q + 1) * (1 - q / n))
    vr = float(varq / var1) if var1 > 0 else 1.0
    # heteroskedasticity-robust variance of VR (Lo-MacKinlay 1988)
    theta = 0.0
    for k in range(1, q):
        dk = ((r[k:] - mu) ** 2) @ ((r[:-k] - mu) ** 2)
        theta += float((2.0 * (q - k) / q) ** 2 * dk / (np.sum((r - mu) ** 2)) ** 2)
    z = float((vr - 1.0) / np.sqrt(theta)) if theta > 0 else 0.0
    verdict = "trending" if vr > 1 and abs(z) > 1.96 else (
        "mean_reverting" if vr < 1 and abs(z) > 1.96 else "random_walk")
    return {"variance_ratio": vr, "z_stat": z, "p_value": float(2 * norm.sf(abs(z))),
            "verdict": verdict}


# --- Fundamental Law of Active Management (Grinold-Kahn) -------------------

def breadth_from_ic_autocorr(ic_series, n_bets_per_period: float | None = None) -> float:
    """Breadth BR = N·(1−ρ)/(1+ρ) with ρ = lag-1 autocorr of IC (serial-correlation
    adjusted: autocorrelated ICs aren't independent bets)."""
    ic = np.asarray(ic_series, dtype="float64")
    ic = ic[np.isfinite(ic)]
    n = len(ic)
    if n < 4:
        return 0.0
    bets = float(n if n_bets_per_period is None else n_bets_per_period)
    rho = float(np.corrcoef(ic[:-1], ic[1:])[0, 1])
    if not np.isfinite(rho):
        return bets
    rho = float(np.clip(rho, -0.99, 0.99))
    return bets * (1.0 - rho) / (1.0 + rho)


def icir_flam(ic: float, breadth: float) -> float:
    """Annualised IC information ratio under the Fundamental Law: IC·√BR."""
    return float(ic * np.sqrt(max(breadth, 0.0)))


def fundamental_ir(ic: float, breadth: float, tc: float = 1.0) -> float:
    """IR = IC · √BR · TC (Grinold-Kahn). ``tc`` = transfer coefficient (0-1)."""
    return float(ic * np.sqrt(max(breadth, 0.0)) * tc)


def flam_summary(ic_series, *, tc: float = 1.0) -> dict:
    """IC / breadth / IR decomposition from a rolling-IC series."""
    ic = np.asarray(ic_series.to_numpy() if isinstance(ic_series, pd.Series) else ic_series,
                    dtype="float64")
    ic = ic[np.isfinite(ic)]
    ic_mean = float(ic.mean()) if len(ic) else 0.0
    br = breadth_from_ic_autocorr(ic)
    return {"ic": ic_mean, "breadth": br, "tc": tc,
            "icir": icir_flam(ic_mean, br), "ir": fundamental_ir(ic_mean, br, tc)}


# ---------------------------------------------------------------------------
# Forecast-error metrics
# ---------------------------------------------------------------------------
# These functions measure how close *numeric* return forecasts are to realised
# returns.  They are distinct from trade-level MAE (maximum adverse excursion)
# and from IC-based signal diagnostics above.
#
# When to use which:
#   MAE  — "typical absolute error"; same units as the target; robust to
#           outliers (uses absolute value, not square).  Good default summary.
#   RMSE — penalises large errors more than MAE (uses squared residuals).
#           Useful when large misses are disproportionately costly.
#   MAPE — "percentage error"; unit-free, easy to communicate, but unstable
#           when actual values are near zero (division blows up).  Use only
#           when actuals are safely bounded away from zero.
# ---------------------------------------------------------------------------


def _align_pairs(
    predictions, actuals
) -> tuple[np.ndarray, np.ndarray]:
    """Return two clean float64 arrays aligned on index (if Series) with NaN
    pairs dropped.  Raises ValueError if lengths differ and at least one input
    is not a Series (no index to align on)."""
    if isinstance(predictions, pd.Series) and isinstance(actuals, pd.Series):
        combined = pd.concat(
            {"p": predictions, "a": actuals}, axis=1
        ).dropna()
        p = combined["p"].to_numpy(dtype="float64")
        a = combined["a"].to_numpy(dtype="float64")
        return p, a

    p = np.asarray(predictions, dtype="float64")
    a = np.asarray(actuals, dtype="float64")
    if p.shape != a.shape:
        raise ValueError(
            f"predictions and actuals must have the same length; "
            f"got {len(p)} vs {len(a)}.  Pass pd.Series to align on index."
        )
    mask = np.isfinite(p) & np.isfinite(a)
    return p[mask], a[mask]


def forecast_mae(predictions, actuals) -> float:
    """Mean Absolute Error = mean(|actual − prediction|).

    Parameters
    ----------
    predictions, actuals:
        Aligned numeric arrays or pd.Series of return forecasts and
        realised returns.  If both are Series they are inner-joined on index
        before NaN pairs are dropped.

    Returns
    -------
    float
        MAE in the same units as *actuals*.  Returns ``float('nan')`` when
        no valid pairs remain.

    Notes
    -----
    MAE is the "typical" absolute error and is robust to outliers compared
    with RMSE.  It does not distinguish the direction of the error.
    """
    p, a = _align_pairs(predictions, actuals)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)))


def forecast_rmse(predictions, actuals) -> float:
    """Root Mean Squared Error = sqrt(mean((actual − prediction)^2)).

    Parameters
    ----------
    predictions, actuals:
        Aligned numeric arrays or pd.Series.  Series are inner-joined on
        index; NaN pairs are dropped.

    Returns
    -------
    float
        RMSE in the same units as *actuals*.  Returns ``float('nan')`` when
        no valid pairs remain.

    Notes
    -----
    RMSE penalises large errors more heavily than MAE because it squares the
    residuals before averaging.  Use when large misses are disproportionately
    costly (e.g. tail-risk-sensitive strategies).
    """
    p, a = _align_pairs(predictions, actuals)
    if len(p) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def forecast_mape(
    predictions,
    actuals,
    zero_threshold: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error = mean(|(actual − prediction) / actual|).

    Parameters
    ----------
    predictions, actuals:
        Aligned numeric arrays or pd.Series.  Series are inner-joined on
        index; NaN pairs are dropped.
    zero_threshold:
        Pairs where ``|actual| < zero_threshold`` are **silently skipped**
        (excluded from the mean) to prevent division-by-zero inflation.
        Default 1e-8 is well below any realistic daily-return magnitude.

    Returns
    -------
    float
        MAPE as a non-negative fraction (0.10 = 10 %).  Returns
        ``float('nan')`` when no valid pairs remain after the near-zero
        guard.

    Notes
    -----
    MAPE is unit-free and easy to communicate but is undefined (or very
    noisy) when actual returns are near zero — which is common for daily
    return series.  Prefer MAE / RMSE for daily returns; reserve MAPE for
    aggregated (e.g. monthly) returns or level forecasts.

    Guard detail
    ------------
    Rather than clipping (which would distort the percentage), near-zero
    actual observations are **excluded** from the average.  The caller
    should check ``n`` in :func:`forecast_error_report` to confirm the
    exclusion rate is acceptable.
    """
    p, a = _align_pairs(predictions, actuals)
    if len(p) == 0:
        return float("nan")
    valid = np.abs(a) >= zero_threshold
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs((a[valid] - p[valid]) / a[valid])))


def forecast_error_report(
    predictions,
    actuals,
    zero_threshold: float = 1e-8,
) -> dict:
    """Bundle MAE, RMSE, MAPE and observation counts into a single dict.

    Parameters
    ----------
    predictions, actuals:
        Aligned numeric arrays or pd.Series.
    zero_threshold:
        Passed through to :func:`forecast_mape`; pairs where
        ``|actual| < zero_threshold`` are excluded from MAPE only.

    Returns
    -------
    dict with keys:
        ``mae``        — Mean Absolute Error (float).
        ``rmse``       — Root Mean Squared Error (float).
        ``mape``       — Mean Absolute Percentage Error (float | nan).
        ``n``          — Number of valid (non-NaN) pairs used for MAE/RMSE.
        ``n_mape``     — Number of pairs used for MAPE (after near-zero
                         guard; always ``<= n``).
    """
    p, a = _align_pairs(predictions, actuals)
    n = len(p)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "mape": float("nan"), "n": 0, "n_mape": 0}

    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    valid_mape = np.abs(a) >= zero_threshold
    n_mape = int(valid_mape.sum())
    mape = (
        float(np.mean(np.abs((a[valid_mape] - p[valid_mape]) / a[valid_mape])))
        if n_mape > 0
        else float("nan")
    )

    return {"mae": mae, "rmse": rmse, "mape": mape, "n": n, "n_mape": n_mape}


# ---------------------------------------------------------------------------
# Signal decay (exponential IC-decay fit) & Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------
# These are the *guarded, status-returning* siblings of ``signal_decay`` /
# ``ou_halflife`` above. Where the originals return a thin dict on bad input,
# these always return a fully-populated dict carrying a ``status`` string and
# NaNs on a degenerate fit — never raising — so they're safe to call inside a
# batch diagnostics loop. Both are deterministic and pure numpy + scipy (a
# hard dependency of this module); no optional library is required. If
# statsmodels is installed it is *not* needed here — the closed-form OLS below
# is exact for a single regressor.

# Minimum usable observations for each fit. Below these the slope estimate is
# too unstable to report and we return NaN + a status rather than a number.
_MIN_DECAY_OBS = 4       # exponential decay needs a few (lag, IC) points
_MIN_OU_OBS = 30         # AR(1) half-life needs a reasonable sample


def compute_signal_decay(
    ic_by_lag: dict | list | np.ndarray | pd.Series,
) -> dict:
    """Fit an exponential decay to IC-by-horizon and report the half-life.

    Models ``IC(t) = IC0 * exp(-lambda * t)`` and estimates ``lambda`` by a
    log-linear OLS of ``log|IC(t)|`` on the lag ``t``. The decay half-life is
    ``ln(2) / lambda`` — the horizon over which the (absolute) information
    coefficient halves. A short half-life means the edge is fleeting; a long
    one means it persists.

    Parameters
    ----------
    ic_by_lag:
        Either a mapping ``{lag: ic}`` (lags need not be sorted) or a 1-D
        sequence of ICs whose positions are treated as lags ``1, 2, 3, ...``.

    Returns
    -------
    dict with keys:
        ``half_life``      — ln(2)/lambda in lag units; ``inf`` if the series
                             does not decay (lambda <= 0); ``nan`` on a guard
                             trip.
        ``decay_rate``     — fitted ``lambda`` (per-lag); ``nan`` on a guard.
        ``r_squared``      — goodness-of-fit of the log-linear regression.
        ``peak_ic``        — signed IC at the strongest |IC| lag.
        ``optimal_horizon``— the lag (in input units) of that peak.
        ``n_obs``          — number of usable (finite, non-zero) IC points.
        ``status``         — ``"ok"`` or a human-readable reason for NaNs.

    Notes
    -----
    Guarded: fewer than ``_MIN_DECAY_OBS`` usable points, or an ill-conditioned
    design matrix, returns NaNs with an explanatory ``status`` rather than
    raising. Zero / non-finite ICs are dropped before the log is taken.
    """
    if isinstance(ic_by_lag, dict):
        lags = np.array(sorted(ic_by_lag), dtype="float64")
        ics = np.array([ic_by_lag[k] for k in lags], dtype="float64")
    else:
        seq = ic_by_lag.to_numpy() if isinstance(ic_by_lag, pd.Series) else ic_by_lag
        ics = np.asarray(seq, dtype="float64")
        lags = np.arange(1, len(ics) + 1, dtype="float64")

    nan = float("nan")
    base = {
        "half_life": nan, "decay_rate": nan, "r_squared": nan,
        "peak_ic": 0.0, "optimal_horizon": 0, "n_obs": 0,
    }

    if ics.size == 0:
        return {**base, "status": "empty input"}

    mask = np.isfinite(ics) & (np.abs(ics) > 1e-10)
    n_obs = int(mask.sum())
    if n_obs < _MIN_DECAY_OBS:
        return {
            **base, "n_obs": n_obs,
            "status": f"insufficient usable IC points ({n_obs} < {_MIN_DECAY_OBS})",
        }

    lags_c, ics_c = lags[mask], ics[mask]
    if np.allclose(lags_c, lags_c[0]):
        return {
            **base, "n_obs": n_obs,
            "status": "degenerate lags (no horizon spread)",
        }

    log_ic = np.log(np.abs(ics_c))
    design = np.column_stack([np.ones(n_obs), lags_c])
    try:
        coeffs, *_ = np.linalg.lstsq(design, log_ic, rcond=None)
    except np.linalg.LinAlgError:
        return {**base, "n_obs": n_obs, "status": "OLS failed (singular design)"}

    rate = float(-coeffs[1])
    fitted = design @ coeffs
    ss_tot = float(np.sum((log_ic - log_ic.mean()) ** 2))
    r_squared = float(1.0 - np.sum((log_ic - fitted) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    peak = int(np.argmax(np.abs(ics)))

    return {
        "half_life": float(np.log(2) / rate) if rate > 0 else float("inf"),
        "decay_rate": rate,
        "r_squared": r_squared,
        "peak_ic": float(ics[peak]),
        "optimal_horizon": int(lags[peak]),
        "n_obs": n_obs,
        "status": "ok" if rate > 0 else "no decay (rate <= 0)",
    }


def compute_ou_half_life(
    series: pd.Series | np.ndarray | list,
    periods_per_year: int = 252,
) -> dict:
    """Ornstein-Uhlenbeck mean-reversion half-life of a level series via OLS.

    Fits the discrete AR(1) mean-reversion model
    ``delta_x_t = a + b * x_{t-1} + eps`` (so ``kappa = -b`` is the reversion
    speed) and reports the half-life ``ln(2) / kappa`` — the time for a
    deviation from the long-run mean ``mu = a / kappa`` to decay by half.
    A strongly mean-reverting series gives a short half-life; a random walk
    gives ``kappa -> 0`` and an effectively infinite half-life.

    Parameters
    ----------
    series:
        The level series whose mean-reversion is measured (e.g. a spread, a
        ratio, or a stationary signal level — *not* a return series).
    periods_per_year:
        Calendar scale used only to annualise the half-life.

    Returns
    -------
    dict with keys:
        ``half_life``           — ln(2)/kappa in bar units; ``inf`` for a
                                  non-reverting (kappa <= 0) series; ``nan`` on
                                  a guard trip.
        ``half_life_annualized``— half-life / periods_per_year.
        ``kappa``               — reversion speed ``-b``.
        ``mu``                  — implied long-run mean ``a / kappa``.
        ``sigma``               — residual standard deviation of the fit.
        ``t_stat`` / ``p_value``— significance of the slope ``b`` (two-sided).
        ``is_mean_reverting``   — ``b < 0`` and ``p_value < 0.05``.
        ``n_obs``               — number of finite observations used.
        ``status``              — ``"ok"`` or a human-readable reason for NaNs.

    Notes
    -----
    Guarded: fewer than ``_MIN_OU_OBS`` finite points, a constant series, or a
    singular design returns NaNs with an explanatory ``status`` rather than
    raising. Deterministic and pure numpy + scipy.
    """
    seq = series.to_numpy() if isinstance(series, pd.Series) else series
    x = np.asarray(seq, dtype="float64")
    x = x[np.isfinite(x)]

    nan = float("nan")
    base = {
        "half_life": nan, "half_life_annualized": nan, "kappa": nan,
        "mu": nan, "sigma": nan, "t_stat": nan, "p_value": nan,
        "is_mean_reverting": False, "n_obs": int(x.size),
    }

    if x.size < _MIN_OU_OBS:
        return {
            **base,
            "status": f"insufficient observations ({x.size} < {_MIN_OU_OBS})",
        }
    if np.allclose(x, x[0]):
        return {**base, "status": "constant series (no variation)"}

    dx, x_lag = np.diff(x), x[:-1]
    design = np.column_stack([np.ones(x_lag.size), x_lag])
    try:
        coeffs, *_ = np.linalg.lstsq(design, dx, rcond=None)
    except np.linalg.LinAlgError:
        return {**base, "status": "OLS failed (singular design)"}

    a, b = float(coeffs[0]), float(coeffs[1])
    kappa = -b
    resid = dx - design @ coeffs
    dof = x_lag.size - 2
    sigma = float(np.std(resid, ddof=2)) if dof > 0 else float("nan")

    try:
        xtx_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return {**base, "status": "OLS failed (singular normal matrix)"}
    se_b = sigma * np.sqrt(xtx_inv[1, 1]) if np.isfinite(sigma) else 0.0
    t_stat = b / se_b if se_b > 0 else 0.0
    p_value = float(2.0 * _t.sf(abs(t_stat), df=dof)) if dof > 0 else float("nan")

    half_life = float(np.log(2) / kappa) if kappa > 0 else float("inf")
    mu = a / kappa if kappa > 0 else float("nan")

    return {
        "half_life": half_life,
        "half_life_annualized": (
            half_life / periods_per_year if np.isfinite(half_life) else float("inf")
        ),
        "kappa": float(kappa),
        "mu": float(mu) if np.isfinite(mu) else float("nan"),
        "sigma": sigma,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "is_mean_reverting": bool(b < 0 and np.isfinite(p_value) and p_value < 0.05),
        "n_obs": int(x.size),
        "status": "ok",
    }
