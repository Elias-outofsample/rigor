"""Return-distribution & stationarity/autocorrelation test battery.

A forensic battery for the *statistical character* of a return (or residual)
series — normality, stationarity, serial dependence and conditional
heteroskedasticity — answering "is this a stable, exploitable process or a
random walk / drifting artefact?".

  * ``jarque_bera`` — normality of the unconditional distribution.
  * ``runs_test`` — Wald-Wolfowitz randomness of the sign sequence (median split).
  * ``variance_ratio`` — Lo-MacKinlay (1988) overlapping-block VR (trending vs
    mean-reverting). Pure numpy/scipy.
  * ``durbin_watson`` — lag-1 serial correlation of residuals.
  * ``arch_lm_test`` / ``white_hetero_test`` — Engle ARCH-LM and a White-style
    heteroskedasticity test, both via closed-form numpy OLS (no statsmodels).
  * ``adf_test`` / ``kpss_test`` — Augmented Dickey-Fuller and KPSS stationarity
    tests; these need the optional ``statsmodels`` and **degrade gracefully** to
    ``NaN`` + a ``status`` string when it is absent.
  * ``ljung_box_test`` — Ljung-Box portmanteau autocorrelation test (statsmodels
    when available, else a pure chi-square fallback on the sample ACF).
  * ``analyze_distribution`` — the full bundle; the pure path (JB / runs / VR /
    DW / ARCH-LM / White) always runs, the statsmodels gates report a fallback.

The JB / runs / variance-ratio / Durbin-Watson / ARCH-LM / White paths are pure
scipy/numpy and always work. ADF, KPSS and (optionally) Ljung-Box come from
``statsmodels`` via the :mod:`rigor.analysis._deps` soft-import — never hard-imported
at module top — so the core install is unbreakable.

References
----------
Lo & MacKinlay (1988), "Stock Market Prices Do Not Follow Random Walks".
Engle (1982), ARCH-LM. Jarque & Bera (1980). Kwiatkowski et al. (1992), KPSS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import jarque_bera as _scipy_jb

from ._deps import optional_import

__all__ = [
    "analyze_distribution",
    "jarque_bera",
    "runs_test",
    "variance_ratio",
    "durbin_watson",
    "arch_lm_test",
    "white_hetero_test",
    "adf_test",
    "kpss_test",
    "ljung_box_test",
]

# Minimum observations below which the full battery is not meaningful.
_MIN_OBS = 10


def _clean(returns) -> np.ndarray:
    """Coerce input to a finite float64 1-D array (drops NaN/inf)."""
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype="float64")
    else:
        arr = np.asarray(returns, dtype="float64")
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# Pure-path tests (numpy / scipy only — always available)
# ---------------------------------------------------------------------------


def jarque_bera(returns) -> dict:
    """Jarque-Bera test of normality (skewness + excess kurtosis).

    ``is_normal`` is ``True`` when ``p_value > 0.05`` (fail to reject normality).
    Returns NaNs + a ``status`` on a degenerate (constant / too-short) series.
    """
    r = _clean(returns)
    n = int(r.size)
    nan = float("nan")
    base = {"statistic": nan, "p_value": nan, "is_normal": False, "n_obs": n}
    if n < _MIN_OBS:
        return {**base, "status": f"insufficient data ({n} < {_MIN_OBS})"}
    if np.allclose(r, r[0]):
        return {**base, "status": "constant series (no variation)"}
    stat, p = _scipy_jb(r)
    return {"statistic": float(stat), "p_value": float(p),
            "is_normal": bool(p > 0.05), "n_obs": n, "status": "ok"}


def runs_test(returns) -> dict:
    """Wald-Wolfowitz runs test for randomness of the sign sequence.

    Splits the series at its median and counts runs of consecutive
    above/below-median observations. ``is_random`` is ``True`` when
    ``p_value > 0.05``. A degenerate split (all on one side) reports a neutral
    random result.
    """
    r = _clean(returns)
    n = int(r.size)
    nan = float("nan")
    base = {"n_runs": 0, "z_stat": nan, "p_value": nan, "is_random": True, "n_obs": n}
    if n < _MIN_OBS:
        return {**base, "z_stat": 0.0, "p_value": 1.0,
                "status": f"insufficient data ({n} < {_MIN_OBS})"}

    signs = r > np.median(r)
    n_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n_pos = int(np.sum(signs))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {**base, "n_runs": n_runs, "z_stat": 0.0, "p_value": 1.0,
                "status": "degenerate split (all on one side of median)"}

    expected = 1.0 + 2.0 * n_pos * n_neg / n
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n**2 * (n - 1))
    z = (n_runs - expected) / np.sqrt(max(var, 1e-12))
    p = 2.0 * norm.sf(abs(z))
    return {"n_runs": n_runs, "z_stat": float(z), "p_value": float(p),
            "is_random": bool(p > 0.05), "n_obs": n, "status": "ok"}


def variance_ratio(returns, q: int = 2) -> dict:
    """Lo-MacKinlay (1988) variance-ratio test with overlapping blocks.

    Uses the efficient overlapping q-sum estimator (Eq. 9) which has optimal
    asymptotic power vs. non-overlapping sampling. ``vr > 1`` ⇒ trending (positive
    serial correlation), ``vr < 1`` ⇒ mean-reverting; ``|z| > 1.96`` rejects the
    random-walk null. Pure numpy/scipy.
    """
    r = _clean(returns)
    n = int(r.size)
    if q < 2:
        q = 2
    if n < q * 2:
        return {"vr": 1.0, "z_stat": 0.0, "p_value": 1.0, "q": q,
                "interpretation": "random_walk", "n_obs": n,
                "status": f"insufficient data ({n} < {q * 2})"}

    cumsum = np.concatenate([[0.0], np.cumsum(r)])
    r_q = cumsum[q:] - cumsum[:-q]  # overlapping q-period sums

    mu = float(np.mean(r))
    var_1 = float(np.var(r, ddof=1))
    if var_1 <= 0:
        return {"vr": 1.0, "z_stat": 0.0, "p_value": 1.0, "q": q,
                "interpretation": "random_walk", "n_obs": n,
                "status": "zero variance"}

    # Lo-MacKinlay Eq. 9: the normaliser folds in the /q scaling.
    m = q * (n - q + 1) * (1.0 - q / n)
    var_per_period_q = float(np.sum((r_q - q * mu) ** 2) / m) if m > 0 else 0.0
    vr = var_per_period_q / var_1

    # Asymptotic variance under homoskedasticity (Lo-MacKinlay Eq. 12).
    phi = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n)
    z = (vr - 1.0) / np.sqrt(phi) if phi > 0 else 0.0
    p = 2.0 * norm.sf(abs(z))

    if vr > 1.2:
        interp = "trending"
    elif vr < 0.8:
        interp = "mean_reverting"
    else:
        interp = "random_walk"
    return {"vr": float(vr), "z_stat": float(z), "p_value": float(p), "q": q,
            "interpretation": interp, "n_obs": n, "status": "ok"}


def durbin_watson(residuals) -> dict:
    """Durbin-Watson statistic for lag-1 serial correlation of residuals.

    The statistic lives in ``[0, 4]``: ≈2 means no autocorrelation, <2 positive,
    >2 negative. ``has_autocorrelation`` flags |dw − 2| > 0.5 as a rough screen.
    """
    r = _clean(residuals)
    n = int(r.size)
    if n < 3:
        return {"statistic": 2.0, "has_autocorrelation": False, "n_obs": n,
                "status": f"insufficient data ({n} < 3)"}
    denom = float(np.sum(r**2))
    if denom <= 0:
        return {"statistic": 2.0, "has_autocorrelation": False, "n_obs": n,
                "status": "zero variance"}
    dw = float(np.sum(np.diff(r) ** 2) / denom)
    return {"statistic": dw, "has_autocorrelation": bool(abs(dw - 2.0) > 0.5),
            "n_obs": n, "status": "ok"}


def arch_lm_test(returns, lags: int = 5) -> dict:
    """Engle's ARCH-LM test for conditional heteroskedasticity (vol clustering).

    Regresses squared deviations on their own ``lags`` and tests ``n·R²`` against
    a chi-square. ``has_arch_effects`` is ``True`` when ``p_value < 0.05``. Pure
    numpy/scipy closed-form OLS (no statsmodels).
    """
    r = _clean(returns)
    n = int(r.size)
    if n < lags + _MIN_OBS:
        return {"lm_stat": 0.0, "p_value": 1.0, "has_arch_effects": False,
                "lags": lags, "n_obs": n,
                "status": f"insufficient data ({n} < {lags + _MIN_OBS})"}

    r_sq = (r - np.mean(r)) ** 2
    y = r_sq[lags:]
    cols = [r_sq[lags - i - 1:n - i - 1] for i in range(lags)]
    design = np.column_stack([np.ones(len(y)), *cols])
    try:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"lm_stat": 0.0, "p_value": 1.0, "has_arch_effects": False,
                "lags": lags, "n_obs": n, "status": "OLS failed (singular design)"}
    fitted = design @ coeffs
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = len(y) * r2
    p = float(chi2.sf(lm, lags))
    return {"lm_stat": float(lm), "p_value": p, "has_arch_effects": bool(p < 0.05),
            "lags": lags, "n_obs": n, "status": "ok"}


def white_hetero_test(returns, exog=None) -> dict:
    """White-style test for (time-dependent) heteroskedasticity.

    With no ``exog`` it regresses squared returns on a quadratic time trend
    (``1, t, t²``) — a screen for vol drifting with time. Otherwise it uses the
    supplied regressors. ``has_hetero`` flags ``p_value < 0.05``. Pure numpy/scipy.
    """
    r = _clean(returns)
    n = int(r.size)
    if n < 20:
        return {"lm_stat": 0.0, "p_value": 1.0, "has_hetero": False, "n_obs": n,
                "status": f"insufficient data ({n} < 20)"}

    r_sq = r**2
    if exog is None:
        t = np.arange(n, dtype="float64")
        design = np.column_stack([np.ones(n), t, t**2])
        df = 2
    else:
        ex = np.asarray(exog, dtype="float64")
        design = np.column_stack([np.ones(n), ex])
        df = design.shape[1] - 1

    try:
        coeffs, *_ = np.linalg.lstsq(design, r_sq, rcond=None)
    except np.linalg.LinAlgError:
        return {"lm_stat": 0.0, "p_value": 1.0, "has_hetero": False, "n_obs": n,
                "status": "OLS failed (singular design)"}
    fitted = design @ coeffs
    ss_res = float(np.sum((r_sq - fitted) ** 2))
    ss_tot = float(np.sum((r_sq - np.mean(r_sq)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = n * r2
    p = float(chi2.sf(lm, df))
    return {"lm_stat": float(lm), "p_value": p, "has_hetero": bool(p < 0.05),
            "n_obs": n, "status": "ok"}


# ---------------------------------------------------------------------------
# statsmodels-backed tests (soft-imported — degrade to NaN + status)
# ---------------------------------------------------------------------------

_NO_SM = "statsmodels not installed (pip install -e '.[analysis]')"


def adf_test(returns) -> dict:
    """Augmented Dickey-Fuller test for a unit root (non-stationarity).

    ``is_stationary`` is ``True`` when ``p_value < 0.05`` (reject the unit-root
    null). Needs the optional ``statsmodels``; without it (or on a degenerate
    series) returns NaNs + a ``status`` rather than raising.
    """
    r = _clean(returns)
    n = int(r.size)
    nan = float("nan")
    base = {"statistic": nan, "p_value": nan, "is_stationary": False, "n_obs": n}
    if n < _MIN_OBS:
        return {**base, "status": f"insufficient data ({n} < {_MIN_OBS})"}
    if np.allclose(r, r[0]):
        return {**base, "status": "constant series (no variation)"}

    stattools = optional_import("statsmodels.tsa.stattools")
    if stattools is None:
        return {**base, "status": _NO_SM}
    try:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
        res = stattools.adfuller(r, maxlag=maxlag)
        return {"statistic": float(res[0]), "p_value": float(res[1]),
                "is_stationary": bool(res[1] < 0.05), "n_obs": n, "status": "ok"}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**base, "status": f"ADF failed: {exc}"}


def kpss_test(returns) -> dict:
    """KPSS test of (level-)stationarity around a constant.

    Complements ADF with the *opposite* null: ``is_stationary`` is ``True`` when
    ``p_value > 0.05`` (fail to reject stationarity). Needs the optional
    ``statsmodels``; degrades to NaN + ``status`` otherwise.
    """
    r = _clean(returns)
    n = int(r.size)
    nan = float("nan")
    base = {"statistic": nan, "p_value": nan, "is_stationary": False, "n_obs": n}
    if n < _MIN_OBS:
        return {**base, "status": f"insufficient data ({n} < {_MIN_OBS})"}
    if np.allclose(r, r[0]):
        return {**base, "status": "constant series (no variation)"}

    stattools = optional_import("statsmodels.tsa.stattools")
    if stattools is None:
        return {**base, "status": _NO_SM}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, _, _ = stattools.kpss(r, regression="c", nlags="auto")
        return {"statistic": float(stat), "p_value": float(p),
                "is_stationary": bool(p > 0.05), "n_obs": n, "status": "ok"}
    except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        return {**base, "status": f"KPSS failed: {exc}"}


def _ljung_box_pure(r: np.ndarray, lags: int) -> dict:
    """Pure chi-square Ljung-Box fallback from the sample ACF (no statsmodels)."""
    n = int(r.size)
    x = r - np.mean(r)
    denom = float(np.sum(x**2))
    if denom <= 0:
        return {"p_value": 1.0, "has_autocorrelation": False, "lags": lags,
                "n_obs": n, "status": "zero variance"}
    stat = 0.0
    for k in range(1, lags + 1):
        rho_k = float(np.sum(x[k:] * x[:-k]) / denom)
        stat += rho_k**2 / (n - k)
    stat *= n * (n + 2)
    p = float(chi2.sf(stat, lags))
    return {"statistic": float(stat), "p_value": p,
            "has_autocorrelation": bool(p < 0.05), "lags": lags, "n_obs": n,
            "status": "ok (pure fallback)"}


def ljung_box_test(returns, lags: int | None = None) -> dict:
    """Ljung-Box portmanteau test for autocorrelation up to ``lags``.

    ``has_autocorrelation`` is ``True`` when ``p_value < 0.05``. Uses
    ``statsmodels`` when present, else a numerically identical pure chi-square
    fallback computed from the sample ACF — so this test always works.
    """
    r = _clean(returns)
    n = int(r.size)
    if n < _MIN_OBS:
        return {"p_value": float("nan"), "has_autocorrelation": False,
                "lags": 0, "n_obs": n,
                "status": f"insufficient data ({n} < {_MIN_OBS})"}
    lb_lags = lags if lags is not None else min(10, max(1, n // 10))
    lb_lags = max(1, min(lb_lags, n - 1))

    diag = optional_import("statsmodels.stats.diagnostic")
    if diag is None:
        return _ljung_box_pure(r, lb_lags)
    try:
        out = diag.acorr_ljungbox(r, lags=lb_lags, return_df=True)
        p = float(out["lb_pvalue"].iloc[-1])
        stat = float(out["lb_stat"].iloc[-1])
        return {"statistic": stat, "p_value": p,
                "has_autocorrelation": bool(p < 0.05), "lags": lb_lags,
                "n_obs": n, "status": "ok"}
    except (ValueError, KeyError, IndexError):
        return _ljung_box_pure(r, lb_lags)


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------


def analyze_distribution(returns) -> dict:
    """Run the full distribution / stationarity / autocorrelation battery.

    The pure path (Jarque-Bera, runs, variance-ratio, Durbin-Watson, ARCH-LM,
    White) always runs. ADF / KPSS need the optional ``statsmodels`` and report a
    ``status`` fallback when it is absent; Ljung-Box always produces a result
    (statsmodels or a pure chi-square fallback). Returns ``{"error": ...}`` only
    when there is too little data to compute any test.
    """
    r = _clean(returns)
    n = int(r.size)
    if n < _MIN_OBS:
        return {"error": "insufficient data", "n_obs": n}

    ser = pd.Series(r)
    return {
        "jarque_bera": jarque_bera(r),
        "adf": adf_test(r),
        "kpss": kpss_test(r),
        "ljung_box": ljung_box_test(r),
        "variance_ratio": variance_ratio(r, q=2),
        "runs_test": runs_test(r),
        "durbin_watson": durbin_watson(r),
        "arch_lm": arch_lm_test(r, lags=5),
        "white_hetero": white_hetero_test(r),
        "stats": {
            "mean": float(np.mean(r)), "std": float(np.std(r, ddof=1)),
            "skew": float(ser.skew()), "kurtosis": float(ser.kurtosis()),
            "min": float(np.min(r)), "max": float(np.max(r)), "n_obs": n,
        },
        "statsmodels_available": optional_import("statsmodels") is not None,
    }
