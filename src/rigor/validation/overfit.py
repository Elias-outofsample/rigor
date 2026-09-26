"""Anti-overfit statistics (Bailey & Lopez de Prado; Harvey-Liu-Zhu; Lo).

CONVENTION (important): ``probabilistic_sharpe_ratio``, ``deflated_sharpe_ratio``,
``sharpe_se``, ``haircut_sharpe`` and ``min_track_record_length`` all expect the
**per-period** Sharpe (NOT annualised) and ``n_obs`` in periods. Passing an
annualised Sharpe saturates the z-scores (PSR/DSR -> 1.0). The orchestrator
converts annualised -> per-period (SR / sqrt(ppy)) before calling; PSR and DSR
also accept an optional ``annual_periods`` to self-convert (default 1 = no-op).

``skew`` is Fisher skewness; ``kurt`` is EXCESS kurtosis (Fisher; normal = 0).
Pure numpy + scipy -> deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..metrics import compute_kurtosis, compute_skewness

__all__ = [
    "sharpe_se",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "haircut_sharpe",
    "min_track_record_length",
    "newey_west_sharpe",
    "deflated_sharpe_romano_wolf",
    "pezier_white_adjusted_sharpe",
]

_EULER = 0.5772156649015329


def sharpe_se(sharpe: float, n_obs: int, skew: float = 0.0, kurt: float = 0.0) -> float:
    """Standard error of the (per-period) Sharpe ratio, Lo (2002) non-normal form."""
    if n_obs < 2:
        return float("inf")
    se_sq = (1.0 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4.0) * sharpe**2) / (n_obs - 1)
    return float(np.sqrt(max(se_sq, 0.0)))


def probabilistic_sharpe_ratio(
    sharpe: float, benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 0.0,
    annual_periods: int = 1,
) -> float:
    """PSR = P(true SR > benchmark). Bailey & Lopez de Prado (2012).

    ``sharpe``/``benchmark`` are per-period. Pass ``annual_periods`` (e.g. 252) to
    have an *annualised* Sharpe down-scaled for you (SR/sqrt(ppy)) instead of
    silently saturating the CDF to 1.0; the default of 1 is a no-op.
    """
    if annual_periods > 1:
        scale = np.sqrt(float(annual_periods))
        sharpe, benchmark = sharpe / scale, benchmark / scale
    se = sharpe_se(sharpe, n_obs, skew, kurt)
    if se == 0 or not np.isfinite(se):
        return 0.5
    return float(norm.cdf((sharpe - benchmark) / se))


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max Sharpe] under the null over N independent trials (BLdP 2014, eq.5)."""
    if n_trials <= 1 or n_obs <= 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    e_max = (1.0 - _EULER) * z1 + _EULER * z2
    return float(e_max * np.sqrt(1.0 / (n_obs - 1)))


def deflated_sharpe_ratio(
    sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 0.0,
    annual_periods: int = 1,
) -> float:
    """DSR = P(skill | N trials) = Phi[(SR - E[max SR]) / SE]. BLdP (2014).

    ``sharpe`` is per-period. Pass ``annual_periods`` (e.g. 252) to have an
    *annualised* Sharpe down-scaled for you (SR/sqrt(ppy)) instead of silently
    saturating the CDF to 1.0; the default of 1 is a no-op.
    """
    if annual_periods > 1:
        sharpe = sharpe / np.sqrt(float(annual_periods))
    e_max = expected_max_sharpe(n_trials, n_obs)
    se = sharpe_se(sharpe, n_obs, skew, kurt)
    if se == 0 or not np.isfinite(se):
        return 0.5
    return float(norm.cdf((sharpe - e_max) / se))


def haircut_sharpe(
    sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 0.0,
    method: str = "bhy",
) -> dict:
    """Harvey-Liu-Zhu (2016) multiple-testing haircut on the Sharpe ratio.

    ``method="bhy"`` applies the Benjamini-Yekutieli (2001) correction — the harmonic
    ``Σ 1/k`` factor, conservative and valid under arbitrary dependence across trials;
    ``"sidak"`` applies the Šidák FWER correction. The ``"bhy"`` label is kept for
    backward compatibility, but the formula it selects is Benjamini-Yekutieli.
    """
    se = sharpe_se(sharpe, n_obs, skew, kurt)
    if se == 0 or n_trials <= 0:
        return {"haircut_sharpe": sharpe, "haircut_pct": 0.0, "p_adjusted": 1.0}
    t_stat = sharpe / se
    p_value = 2.0 * norm.sf(abs(t_stat))
    if method == "sidak":
        p_adj = 1.0 - (1.0 - p_value) ** n_trials
    elif method == "bhy":
        harmonic = sum(1.0 / k for k in range(1, n_trials + 1))
        p_adj = min(p_value * n_trials * harmonic, 1.0)
    else:  # bonferroni
        p_adj = min(p_value * n_trials, 1.0)
    if p_adj >= 1.0:
        adj = 0.0
    elif p_adj <= 0.0:
        adj = sharpe
    else:
        adj = min(norm.isf(p_adj / 2.0) * se, sharpe)
    hc_pct = 1.0 - (adj / sharpe) if sharpe != 0 else 0.0
    return {
        "haircut_sharpe": float(adj),
        "haircut_pct": float(max(hc_pct, 0.0)),
        "p_adjusted": float(p_adj),
    }


def min_track_record_length(
    sharpe: float, target: float = 0.0, skew: float = 0.0, kurt: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Min number of periods to distinguish SR from target at 1-alpha (BLdP 2014)."""
    if sharpe <= target:
        return float("inf")
    z = norm.ppf(1.0 - alpha)
    d = sharpe - target
    n_star = (1.0 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4.0) * sharpe**2) * (z / d) ** 2
    return float(max(n_star, 1.0))


def newey_west_sharpe(returns: np.ndarray, ppy: int, max_lags: int | None = None) -> dict:
    """Newey-West HAC-adjusted (annualised) Sharpe — corrects for autocorrelation."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return {"raw_sharpe": 0.0, "nw_sharpe": 0.0, "ratio": 1.0, "lags_used": 0}
    mu, std = float(r.mean()), float(r.std(ddof=1))
    raw = mu / std * np.sqrt(ppy) if std > 1e-12 else 0.0
    lags = max_lags if max_lags is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(1, min(lags, n // 4))
    d = r - mu
    nw_var = float(np.mean(d**2))
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        nw_var += 2.0 * w * float(np.mean(d[k:] * d[: n - k]))
    nw_var = max(nw_var, 1e-18)
    nw = mu / np.sqrt(nw_var) * np.sqrt(ppy)
    return {
        "raw_sharpe": float(raw),
        "nw_sharpe": float(nw),
        "ratio": float(nw / raw) if abs(raw) > 1e-9 else 1.0,
        "lags_used": int(lags),
    }


def deflated_sharpe_romano_wolf(
    sharpe: float, n_obs: int, returns_matrix: np.ndarray, *,
    skew: float = 0.0, kurt: float = 0.0, n_bootstrap: int = 1000,
    block_length: int = 10, seed: int = 42, max_trials_for_bootstrap: int = 500,
) -> dict:
    """DSR with an *empirical* E[max SR] from a stationary block bootstrap.

    The closed-form ``deflated_sharpe_ratio`` assumes independent trials. When
    neighbouring grid params produce correlated returns, it overstates the
    multiple-testing penalty on isolated peaks and understates it on plateaus.
    This variant resamples the (n_obs, n_trials) returns matrix with a stationary
    block bootstrap (Politis-Romano), preserving cross-trial correlation and
    serial dependence, and recenters each column on its mean (Romano-Wolf 2005).

    ``sharpe`` is the **per-period** Sharpe of the deployed combo (same convention
    as the rest of this module). Returns the empirical and closed-form DSR/E[max].
    """
    rm = np.asarray(returns_matrix, dtype=np.float64)
    if rm.ndim != 2 or rm.shape[0] < 5 or rm.shape[1] < 2:
        return {"error": "returns_matrix must be 2-D with >=5 obs and >=2 trials",
                "dsr_rw": float("nan")}
    n, k = rm.shape
    se = sharpe_se(sharpe, n_obs, skew, kurt)
    classic_dsr = deflated_sharpe_ratio(sharpe, k, n_obs, skew, kurt)
    classic_e_max = expected_max_sharpe(k, n_obs)

    rng_sub = np.random.default_rng(seed + 1)
    sampled_k = k
    if max_trials_for_bootstrap and k > max_trials_for_bootstrap:
        cols = rng_sub.choice(k, size=max_trials_for_bootstrap, replace=False)
        rm_boot, sampled_k = rm[:, cols], max_trials_for_bootstrap
    else:
        rm_boot = rm

    rng = np.random.default_rng(seed)
    p_geo = 1.0 / max(block_length, 1)
    centered = rm_boot - rm_boot.mean(axis=0, keepdims=True)
    boot_max = []
    for _ in range(int(n_bootstrap)):
        new_block = np.asarray(rng.random(n) < p_geo)
        new_block[0] = True
        starts = rng.integers(0, n, size=n)
        anchor = np.maximum.accumulate(np.where(new_block, starts, 0) + 1) - 1
        last_new = np.maximum.accumulate(np.where(new_block, np.arange(n), 0))
        idx = (anchor + (np.arange(n) - last_new)) % n
        sample = centered[idx]
        stds = sample.std(axis=0, ddof=1)
        stds = np.where(stds > 0, stds, np.inf)
        sr = sample.mean(axis=0) / stds   # per-period; ann. scale cancels in argmax
        boot_max.append(float(np.where(np.isfinite(sr), sr, -np.inf).max()))

    e_max_emp = float(np.mean(boot_max)) if boot_max else 0.0
    dsr_rw = (0.5 if se == 0 or not np.isfinite(se)
              else float(norm.cdf((sharpe - e_max_emp) / se)))
    return {
        "dsr_rw": round(dsr_rw, 6), "dsr_classic": round(classic_dsr, 6),
        "e_max_empirical": round(e_max_emp, 6), "e_max_classic": round(classic_e_max, 6),
        "n_trials": int(k), "n_trials_sampled_for_bootstrap": int(sampled_k),
        "n_obs": int(n_obs), "n_bootstrap": int(n_bootstrap),
    }


def pezier_white_adjusted_sharpe(returns, periods_per_year: int = 252) -> float:
    """Pezier & White (2006) skewness/kurtosis-adjusted Sharpe ratio (annualised).

    The plain Sharpe assumes Gaussian returns, so it rewards a strategy that buys
    its smooth track record with rare catastrophic losses (negative skew, fat
    tails) exactly as much as a genuinely symmetric one. The adjusted Sharpe
    discounts the Sharpe by the investor's preference for positive skew and
    aversion to kurtosis, making it a useful anti-overfit sniff test: a high plain
    Sharpe that collapses once skew/kurtosis are penalised is a red flag.

    Formula (P&W, applied to the **per-period** Sharpe ``SR``)::

        ASR = SR * (1 + (S / 6) * SR - (K_excess / 24) * SR**2)

    where ``S`` is the Fisher-Pearson skewness and ``K_excess`` is the *excess*
    kurtosis (normal = 0) of the per-period returns. We reuse
    :func:`rigor.metrics.compute_skewness` / :func:`rigor.metrics.compute_kurtosis`
    for convention consistency; ``compute_kurtosis`` already returns *excess*
    kurtosis, so it slots directly into the ``K_excess`` term (i.e. the textbook
    ``(K - 3)`` is exactly our ``K_excess``).

    Interpretation relative to the plain Sharpe (same returns, same annualisation):

    * **Negative skew and/or fat tails** (``K_excess > 0``) -> ``ASR < SR``: the
      smooth-then-crash payoff is penalised.
    * **Positive skew** -> ``ASR > SR``: convex, lottery-like payoffs are rewarded.
    * **~Normal** (``S ≈ 0``, ``K_excess ≈ 0``) -> ``ASR ≈ SR``.

    Annualisation follows the house convention (``rigor.metrics.compute_sharpe``):
    the per-period Sharpe uses the sample std (``ddof=1``) and is scaled by
    ``sqrt(periods_per_year)``. We apply the P&W correction to the *per-period*
    Sharpe and then annualise, so ``ASR`` and the plain annualised Sharpe live on
    the same scale and are directly comparable.

    Parameters
    ----------
    returns:
        Array-like of per-period arithmetic returns (NaNs are dropped).
    periods_per_year:
        Annualisation factor (252 daily, 52 weekly, 12 monthly).

    Returns
    -------
    float
        The annualised adjusted Sharpe. Guarded edge cases return ``0.0``
        (house style, matching ``compute_sharpe``): fewer than 3 finite
        observations, zero volatility, or a non-finite result.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = r.std(ddof=1)
    # A constant series has a tiny float-rounding std (~1e-19, not exactly 0); a
    # 1e-12 floor (matching newey_west_sharpe / compute_calmar) treats it as zero
    # vol so the SR**3 term can't explode.
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    sr_pp = float(r.mean() / sd)  # per-period Sharpe (annual scale applied below)

    skewness = compute_skewness(r)
    kurt_excess = compute_kurtosis(r)  # already excess (normal = 0) -> the (K-3) term

    adj = sr_pp * (
        1.0
        + (skewness / 6.0) * sr_pp
        - (kurt_excess / 24.0) * sr_pp**2
    )
    asr = adj * np.sqrt(float(periods_per_year))
    return float(asr) if np.isfinite(asr) else 0.0
