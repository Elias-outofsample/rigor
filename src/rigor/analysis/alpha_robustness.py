"""Alpha-vs-risk-premium robustness (#2, companion to ``factor``).

Distinguishes idiosyncratic alpha (skill) from compensation for factor exposure
(risk premium), via the asset-pricing diagnostics:

  * ``test_alpha_frequency_invariance`` — does alpha survive at D/W/M frequencies?
  * ``compute_regime_conditional_alpha`` — is alpha present in every regime?
  * ``detect_missing_factors`` — PCA on residuals: is an unmodelled factor left?
  * ``stress_test_alpha`` — alpha in crisis vs calm windows.
  * ``alpha_vs_risk_premium_verdict`` — GENUINE_ALPHA / RISK_PREMIUM / MOMENTUM_FACTOR
    / MIXED / NEGATIVE_ALPHA, using progressive-alpha absorption + Harvey-Liu-Zhu t.

Builds on ``rigor.analysis.factor``; scipy only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .factor import compute_factor_regression

__all__ = [
    "CRISIS_PERIODS", "test_alpha_frequency_invariance", "compute_regime_conditional_alpha",
    "detect_missing_factors", "stress_test_alpha", "alpha_vs_risk_premium_verdict",
]

# Default crisis windows (label → (start, end)).
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "dotcom": ("2000-03-01", "2002-10-31"),
    "gfc": ("2007-10-01", "2009-03-31"),
    "euro_debt": ("2011-07-01", "2011-12-31"),
    "china_2015": ("2015-08-01", "2016-02-29"),
    "q4_2018": ("2018-10-01", "2018-12-31"),
    "covid": ("2020-02-15", "2020-04-30"),
    "rates_2022": ("2022-01-01", "2022-10-31"),
    "svb_2023": ("2023-03-01", "2023-05-31"),
}


def _harvey_t_threshold(n_trials: int) -> float:
    """Harvey-Liu-Zhu (2016) Sidak-adjusted two-sided t threshold (baseline 3.0)."""
    if n_trials <= 1:
        return 3.0
    alpha = 1.0 - (1.0 - 0.05) ** (1.0 / n_trials)
    return float(max(3.0, norm.ppf(1.0 - alpha / 2.0)))


def test_alpha_frequency_invariance(returns: pd.Series, factors: pd.DataFrame,
                                    frequencies: tuple[str, ...] = ("D", "W", "M"),
                                    use_hac: bool = True) -> dict:
    """Run the factor regression at each frequency; genuine alpha survives all of them."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        return {"error": "needs DatetimeIndex"}
    by_freq: dict[str, dict] = {}
    for freq in frequencies:
        try:
            r = ((1.0 + returns).resample(freq).prod() - 1.0).dropna()
            if len(r) < 30:
                continue
            res = compute_factor_regression(r, factors, use_hac=use_hac, resample_factors=True)
            if "error" in res:
                continue
            by_freq[freq] = {"alpha": res["alpha"], "t_stat": res["alpha_t_stat"],
                             "p_value": res["alpha_p_value"], "n_obs": res["n_obs"],
                             "r_squared": res["r_squared"]}
        except Exception:
            continue
    if not by_freq:
        return {"by_frequency": {}, "is_invariant": False, "error": "no valid runs"}
    alphas = np.array([v["alpha"] for v in by_freq.values()])
    t_stats = np.array([v["t_stat"] for v in by_freq.values()])
    same_sign = bool(np.all(alphas > 0) or np.all(alphas < 0))
    all_sig = bool(np.all(np.abs(t_stats) > 2.0))
    disp = (float(np.std(alphas, ddof=1) / abs(np.mean(alphas)))
            if abs(np.mean(alphas)) > 1e-9 else float("inf"))
    return {"by_frequency": by_freq, "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0,
            "relative_dispersion": disp, "all_same_sign": same_sign,
            "all_significant_t2": all_sig,
            "is_invariant": same_sign and all_sig and disp < 0.5}


def compute_regime_conditional_alpha(returns: pd.Series, factors: pd.DataFrame,
                                     regime_labels: pd.Series, use_hac: bool = True) -> dict:
    """Factor regression within each regime; robust alpha is significant in all of them."""
    by_regime: dict[str, dict] = {}
    aligned = pd.DataFrame({"r": returns}).join(
        regime_labels.rename("regime"), how="inner").dropna()
    for label in aligned["regime"].unique():
        r_sub = aligned.loc[aligned["regime"] == label, "r"]
        if len(r_sub) < 60:
            continue
        try:
            res = compute_factor_regression(r_sub, factors, use_hac=use_hac)
            if "error" in res:
                continue
            by_regime[str(label)] = {"alpha": res["alpha"], "t_stat": res["alpha_t_stat"],
                                     "p_value": res["alpha_p_value"], "n_obs": res["n_obs"],
                                     "betas": res["betas"]}
        except Exception:
            continue
    if not by_regime:
        return {"by_regime": {}, "is_robust_across_regimes": False}
    alphas = np.array([v["alpha"] for v in by_regime.values()])
    same_sign = bool(np.all(alphas > 0) or np.all(alphas < 0))
    n_sig = sum(1 for v in by_regime.values() if abs(v["t_stat"]) > 2.0)
    return {"by_regime": by_regime, "alpha_mean": float(np.mean(alphas)),
            "alpha_dispersion": float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0,
            "n_regimes": len(by_regime), "n_significant": n_sig, "all_same_sign": same_sign,
            "is_robust_across_regimes": same_sign and n_sig >= len(by_regime) - 1}


def detect_missing_factors(returns: pd.Series, factors: pd.DataFrame,
                           n_components: int = 5, use_hac: bool = True) -> dict:
    """PCA on regression residuals — flags an unmodelled factor behind the 'alpha'."""
    res = compute_factor_regression(returns, factors, use_hac=use_hac)
    if "error" in res:
        return {"error": res["error"]}
    aligned = pd.DataFrame({"r": returns}).join(factors, how="inner").dropna()
    rf = aligned["RF"].to_numpy() if "RF" in aligned.columns else 0.0
    factor_cols = [c for c in factors.columns if c != "RF"]
    y = aligned["r"].to_numpy() - (rf if isinstance(rf, np.ndarray) else 0.0)
    X = np.column_stack([np.ones(len(y))] + [aligned[c].to_numpy() for c in factor_cols])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coeffs

    feat = np.column_stack([resid, resid**2, np.abs(resid)])
    feat = feat - feat.mean(axis=0)
    feat = feat / (feat.std(axis=0, ddof=1) + 1e-12)
    eigvals, eigvecs = np.linalg.eigh(np.cov(feat, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    explained = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
    pc1 = feat @ eigvecs[:, 0]
    pc1_se = pc1.std(ddof=1) / np.sqrt(len(pc1))
    pc1_t = float(pc1.mean() / pc1_se) if pc1_se > 0 else 0.0
    return {"residual_std": float(resid.std(ddof=1)),
            "explained_variance": [float(v) for v in explained[:n_components]],
            "pc1_t_stat": pc1_t, "pc1_significant": abs(pc1_t) > 3.0,
            "missing_factor_likely": abs(pc1_t) > 3.0 and explained[0] > 0.5}


def stress_test_alpha(returns: pd.Series, factors: pd.DataFrame,
                      crisis_periods: dict[str, tuple[str, str]] | None = None,
                      use_hac: bool = True) -> dict:
    """Alpha during crisis windows vs outside; fair-weather alpha is a tail-risk premium."""
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS
    if not isinstance(returns.index, pd.DatetimeIndex):
        return {"error": "needs DatetimeIndex"}
    mask = pd.Series(False, index=returns.index)
    for _, (start, end) in crisis_periods.items():
        mask |= (returns.index >= start) & (returns.index <= end)
    out: dict = {}
    for key, sub in (("in_crisis", returns[mask]), ("out_crisis", returns[~mask])):
        if len(sub) >= 30:
            res = compute_factor_regression(sub, factors, use_hac=use_hac)
            out[key] = {"alpha": res.get("alpha"), "t_stat": res.get("alpha_t_stat"),
                        "n_obs": res.get("n_obs")}
    if "in_crisis" in out and "out_crisis" in out:
        a_in, a_out = out["in_crisis"].get("alpha"), out["out_crisis"].get("alpha")
        out["survives_crisis"] = bool(a_in is not None and a_out is not None
                                      and np.sign(a_in) == np.sign(a_out))
    else:
        out["survives_crisis"] = False
    return out


def alpha_vs_risk_premium_verdict(progressive_alpha: dict[str, dict],
                                  factor_exposures: dict[str, float] | None = None,
                                  n_trials: int = 1) -> dict:
    """Final verdict from progressive-alpha absorption + Harvey-Liu-Zhu t threshold.

    ``progressive_alpha`` is the output of ``factor.compute_progressive_alpha`` — each
    model dict must carry ``alpha`` and ``alpha_t_stat`` (or ``t``). Returns one of
    GENUINE_ALPHA / RISK_PREMIUM / MOMENTUM_FACTOR / MIXED / NEGATIVE_ALPHA / NO_EDGE.
    """
    t_threshold = _harvey_t_threshold(n_trials)

    def _t(d: dict) -> float:
        return abs(d.get("alpha_t_stat", d.get("t", 0.0)))

    capm = progressive_alpha.get("CAPM") or progressive_alpha.get("Mkt") or {}
    ff5 = progressive_alpha.get("FF5") or {}
    ff6 = progressive_alpha.get("FF6") or progressive_alpha.get("Carhart") or ff5
    a_capm = capm.get("alpha", capm.get("alpha_cond"))
    a_ff6 = ff6.get("alpha", ff6.get("alpha_cond"))
    t_capm, t_ff6 = _t(capm), _t(ff6)
    if a_capm is None or a_ff6 is None:
        return {"verdict": "INSUFFICIENT_DATA", "t_threshold": t_threshold}

    absorption = 1.0 - abs(a_ff6) / abs(a_capm) if abs(a_capm) > 1e-9 else 0.0
    beta_mom = (factor_exposures or {}).get("MOM", (factor_exposures or {}).get("UMD", 0.0))
    alpha_positive = a_ff6 > 0

    if not alpha_positive and t_ff6 >= t_threshold:
        verdict, rationale = "NEGATIVE_ALPHA", (
            f"FF6 alpha statistically negative (alpha={a_ff6:.2%}, t={t_ff6:.2f}) — no edge.")
    elif not alpha_positive:
        verdict, rationale = "NO_EDGE", (
            f"FF6 alpha non-positive ({a_ff6:.2%}), t={t_ff6:.2f} below {t_threshold:.2f}.")
    elif t_capm >= t_threshold and t_ff6 >= t_threshold and absorption < 0.50:
        verdict, rationale = "GENUINE_ALPHA", (
            f"Alpha survives CAPM->FF6 (absorption={absorption:.0%}); "
            f"t_FF6={t_ff6:.2f} >= {t_threshold:.2f}.")
    elif beta_mom > 0.30 and absorption > 0.50:
        verdict, rationale = "MOMENTUM_FACTOR", (
            f"beta_MOM={beta_mom:.2f} absorbs most alpha (absorption={absorption:.0%}).")
    elif t_ff6 < 2.0 and absorption > 0.70:
        verdict, rationale = "RISK_PREMIUM", (
            f"FF6 alpha t={t_ff6:.2f} insignificant; {absorption:.0%} factor-absorbed.")
    elif t_ff6 < t_threshold:
        verdict, rationale = "MIXED", (
            f"FF6 alpha t={t_ff6:.2f} between 2.0 and {t_threshold:.2f} — borderline.")
    else:
        verdict, rationale = "GENUINE_ALPHA", (
            f"FF6 alpha t={t_ff6:.2f} >= {t_threshold:.2f}; absorption={absorption:.0%}.")
    return {"verdict": verdict, "rationale": rationale, "alpha_capm": float(a_capm),
            "alpha_ff6": float(a_ff6), "t_capm": float(t_capm), "t_ff6": float(t_ff6),
            "absorption_pct": float(absorption), "beta_mom": float(beta_mom),
            "t_threshold_harvey": float(t_threshold), "alpha_positive": bool(alpha_positive)}
