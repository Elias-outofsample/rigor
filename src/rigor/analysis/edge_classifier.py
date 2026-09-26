"""Edge-source classifier (#3).

Maps a diagnostic vector (signal stats + factor regression + exposure + crisis
decomposition) to a likely edge source — the "where does the alpha come from?"
question aggregate stats alone can't answer. Sources are not mutually exclusive,
so each is scored independently then ranked.

Inputs come from the rest of ``rigor.analysis`` (signal_quality, structural,
exposure, attribution/factor). Pure numpy — no optional deps.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["classify_edge_source", "UNIVERSE_THRESHOLDS"]


# Per-universe threshold presets — daily equity sits in a different regime than
# weekly commodities or hourly crypto. ``universe=`` swaps the active set.
UNIVERSE_THRESHOLDS: dict[str, dict[str, float]] = {
    "equity_us": {
        "ic_hl_short": 5.0, "ou_hl_short": 20.0, "ou_p_signif": 0.05,
        "autocorr_strong": -0.25, "autocorr_med": -0.15, "autocorr_mild": -0.05,
        "mom_autocorr_strong": 0.15, "mom_autocorr_mild": 0.05,
        "vol_corr_strong": -0.15, "skew_neg": -0.5, "skew_pos": 0.5,
        "kurt_fat": 3.0, "crash_short_vol": 0.3, "crash_convexity": 1.5,
        "ff6_t_genuine": 3.0, "ff6_t_marginal": 2.0, "min_obs": 60,
    },
    "equity_global": {
        "ic_hl_short": 5.0, "ou_hl_short": 20.0, "ou_p_signif": 0.05,
        "autocorr_strong": -0.25, "autocorr_med": -0.15, "autocorr_mild": -0.05,
        "mom_autocorr_strong": 0.15, "mom_autocorr_mild": 0.05,
        "vol_corr_strong": -0.15, "skew_neg": -0.5, "skew_pos": 0.5,
        "kurt_fat": 3.0, "crash_short_vol": 0.3, "crash_convexity": 1.5,
        "ff6_t_genuine": 3.0, "ff6_t_marginal": 2.0, "min_obs": 60,
    },
    "commodities": {
        "ic_hl_short": 8.0, "ou_hl_short": 35.0, "ou_p_signif": 0.10,
        "autocorr_strong": -0.20, "autocorr_med": -0.12, "autocorr_mild": -0.04,
        "mom_autocorr_strong": 0.10, "mom_autocorr_mild": 0.04,
        "vol_corr_strong": -0.10, "skew_neg": -0.3, "skew_pos": 0.3,
        "kurt_fat": 5.0, "crash_short_vol": 0.4, "crash_convexity": 1.4,
        "ff6_t_genuine": 2.5, "ff6_t_marginal": 1.8, "min_obs": 80,
    },
    "fx": {
        "ic_hl_short": 3.0, "ou_hl_short": 15.0, "ou_p_signif": 0.05,
        "autocorr_strong": -0.20, "autocorr_med": -0.10, "autocorr_mild": -0.03,
        "mom_autocorr_strong": 0.10, "mom_autocorr_mild": 0.03,
        "vol_corr_strong": -0.20, "skew_neg": -0.3, "skew_pos": 0.3,
        "kurt_fat": 4.0, "crash_short_vol": 0.4, "crash_convexity": 1.6,
        "ff6_t_genuine": 2.5, "ff6_t_marginal": 1.8, "min_obs": 100,
    },
    "crypto": {
        "ic_hl_short": 6.0, "ou_hl_short": 30.0, "ou_p_signif": 0.10,
        "autocorr_strong": -0.30, "autocorr_med": -0.20, "autocorr_mild": -0.08,
        "mom_autocorr_strong": 0.20, "mom_autocorr_mild": 0.08,
        "vol_corr_strong": -0.20, "skew_neg": -0.7, "skew_pos": 0.7,
        "kurt_fat": 8.0, "crash_short_vol": 0.2, "crash_convexity": 2.0,
        "ff6_t_genuine": 2.5, "ff6_t_marginal": 1.8, "min_obs": 120,
    },
}


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _score_mean_reversion(diag: dict, t: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    ic_hl = _safe(diag.get("ic_half_life"), np.inf)
    ou_hl = _safe(diag.get("ou_half_life"), np.inf)
    ou_p = _safe(diag.get("ou_pvalue"), 1.0)
    ac = _safe(diag.get("autocorr_lag1"), 0.0)
    if ic_hl < t["ic_hl_short"]:
        score += 0.3
        notes.append(f"IC half-life {ic_hl:.1f} bars (short → fast decay)")
    if 0 < ou_hl < t["ou_hl_short"] and ou_p < t["ou_p_signif"]:
        score += 0.3
        notes.append(f"OU mean-reversion significant (half-life {ou_hl:.1f}, p={ou_p:.3f})")
    if ac < t["autocorr_strong"]:
        score += 0.6
        notes.append(f"Strong negative return autocorrelation (rho1={ac:.3f})")
    elif ac < t["autocorr_med"]:
        score += 0.45
        notes.append(f"Negative return autocorrelation (rho1={ac:.3f})")
    elif ac < t["autocorr_mild"]:
        score += 0.2
        notes.append(f"Mild negative return autocorrelation (rho1={ac:.3f})")
    return min(score, 1.0), notes


def _score_momentum_premium(diag: dict, t: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    umd_abs = _safe(diag.get("umd_absorption"), 0.0)
    umd_t = _safe(diag.get("umd_t_stat"), 0.0)
    umd_beta = _safe(diag.get("umd_beta"), 0.0)
    ac = _safe(diag.get("autocorr_lag1"), 0.0)
    if umd_abs > 0.5 and abs(umd_t) > 1.96:
        score += 0.6
        notes.append(f"UMD absorbs {umd_abs*100:.0f}% of FF5 alpha "
                     f"(beta={umd_beta:+.2f}, t={umd_t:+.2f})")
    elif umd_abs > 0.3 and abs(umd_t) > 1.96:
        score += 0.35
        notes.append(f"Partial UMD absorption ({umd_abs*100:.0f}%)")
    if umd_beta > 0.30 and abs(umd_t) > 1.96:
        score += 0.2
        notes.append(f"Significant positive momentum loading (beta={umd_beta:.2f})")
    if ac > t["mom_autocorr_strong"]:
        score += 0.5
        notes.append(f"Positive return autocorrelation (rho1={ac:+.3f}) — TS-momentum")
    elif ac > t["mom_autocorr_mild"]:
        score += 0.25
        notes.append(f"Mild positive autocorrelation (rho1={ac:+.3f})")
    return min(score, 1.0), notes


def _score_short_vol_premium(diag: dict, t: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    vol_corr = _safe(diag.get("vol_correlation"), 0.0)
    skew = _safe(diag.get("skewness"), 0.0)
    kurt = _safe(diag.get("kurtosis"), 0.0)
    crash = _safe(diag.get("crash_sharpe_ratio"), 1.0)
    if vol_corr < t["vol_corr_strong"]:
        score += 0.3
        notes.append(f"Negative correlation to volatility (rho={vol_corr:.2f})")
    if skew < t["skew_neg"]:
        score += 0.25
        notes.append(f"Negative skewness ({skew:.2f}) — left-tail risk")
    if kurt > t["kurt_fat"]:
        score += 0.15
        notes.append(f"Excess kurtosis ({kurt:.2f}) — fat tails")
    if crash < t["crash_short_vol"]:
        score += 0.3
        notes.append(f"Sharpe collapses in crash regime ({crash*100:.0f}% of full-sample)")
    return score, notes


def _score_long_convexity(diag: dict, t: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    skew = _safe(diag.get("skewness"), 0.0)
    gamma_t = _safe(diag.get("gamma_t_stat"), 0.0)
    crash = _safe(diag.get("crash_sharpe_ratio"), 1.0)
    if skew > t["skew_pos"]:
        score += 0.3
        notes.append(f"Positive skewness ({skew:.2f}) — right-tail bias")
    if gamma_t > 1.96:
        score += 0.4
        notes.append(f"Significant long-gamma payoff (t={gamma_t:.2f})")
    if crash > t["crash_convexity"]:
        score += 0.3
        notes.append(f"Sharpe IMPROVES in crash regime ({crash*100:.0f}% of full-sample)")
    return score, notes


def _score_carry(diag: dict, t: dict) -> tuple[float, list[str]]:
    turnover = _safe(diag.get("turnover"), 1.0)
    drift_t = _safe(diag.get("drift_t_stat"), 0.0)
    if turnover < 0.05 and drift_t > 2.0:
        return 0.5, [f"Low turnover ({turnover*100:.1f}%/bar) with significant drift "
                     f"(t={drift_t:.2f}) — carry signature"]
    return 0.0, []


def _score_microstructure(diag: dict, t: dict) -> tuple[float, list[str]]:
    ic_hl = _safe(diag.get("ic_half_life"), np.inf)
    turnover = _safe(diag.get("turnover"), 0.0)
    if ic_hl < 1.5 and turnover > 0.5:
        return 0.6, [f"Sub-bar IC decay (half-life {ic_hl:.2f}) + high turnover "
                     f"({turnover*100:.0f}%) — microstructure or execution edge"]
    return 0.0, []


def _score_residual_alpha(diag: dict, t: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    alpha_t = _safe(diag.get("ff6_alpha_t_stat"), 0.0)
    alpha_p = _safe(diag.get("ff6_alpha_p_value"), 1.0)
    umd_abs = _safe(diag.get("umd_absorption"), 0.0)
    if alpha_t > t["ff6_t_genuine"] and alpha_p < 0.05 and umd_abs < 0.3:
        score += 0.7
        notes.append(f"FF6 alpha t={alpha_t:.2f} (Harvey {t['ff6_t_genuine']:.1f}), "
                     f"<{umd_abs*100:.0f}% UMD-absorbed — genuine residual alpha")
    elif alpha_t > t["ff6_t_marginal"] and alpha_p < 0.05 and umd_abs < 0.5:
        score += 0.4
        notes.append(f"FF6 alpha t={alpha_t:.2f} survives partial momentum absorption")
    return score, notes


def _is_degenerate(diag: dict, t: dict) -> tuple[bool, str]:
    ac = diag.get("autocorr_lag1")
    if ac is None:
        return True, "autocorr_lag1 missing"
    try:
        acf = float(ac)
    except (TypeError, ValueError):
        return True, "autocorr_lag1 not numeric"
    if not np.isfinite(acf):
        return True, "autocorr_lag1 non-finite (constant or empty series)"
    if abs(acf) > 0.95:
        return True, f"autocorr_lag1 saturated ({acf:+.2f}) — too auto-correlated to classify"
    n_obs = diag.get("n_valid_obs")
    if n_obs is not None:
        try:
            min_obs = int(t.get("min_obs", 60))
            if int(n_obs) < min_obs:
                return True, f"only {n_obs} valid observations (universe min: {min_obs})"
        except (TypeError, ValueError):
            pass
    return False, ""


_SCORERS = {
    "MEAN_REVERSION": _score_mean_reversion,
    "MOMENTUM_PREMIUM": _score_momentum_premium,
    "SHORT_VOL_PREMIUM": _score_short_vol_premium,
    "LONG_CONVEXITY": _score_long_convexity,
    "CARRY": _score_carry,
    "MICROSTRUCTURE": _score_microstructure,
    "RESIDUAL_ALPHA": _score_residual_alpha,
}


def classify_edge_source(diag: dict, universe: str = "equity_us") -> dict:
    """Classify the likely edge source from a diagnostic vector.

    ``diag`` may carry any subset of: ic_half_life, ou_half_life, ou_pvalue,
    autocorr_lag1, umd_absorption, umd_t_stat, umd_beta, vol_correlation,
    skewness, kurtosis, crash_sharpe_ratio, gamma_t_stat, turnover,
    drift_t_stat, ff6_alpha_t_stat, ff6_alpha_p_value, n_valid_obs.

    ``universe`` selects a ``UNIVERSE_THRESHOLDS`` preset (falls back to
    ``equity_us``). Returns primary/secondary labels, the score table,
    per-source notes, a narrative, and a confidence band.
    """
    t = UNIVERSE_THRESHOLDS.get(universe, UNIVERSE_THRESHOLDS["equity_us"])
    degenerate, reason = _is_degenerate(diag, t)
    if degenerate:
        return {
            "primary": "DEGENERATE_INPUT", "secondary": [], "scores": {},
            "narrative": (f"Cannot classify edge source: {reason}. The returns series "
                          "is too short, constant, or otherwise degenerate for the "
                          "diagnostic tests to be meaningful."),
            "notes_by_source": {}, "confidence": "n/a", "universe": universe,
        }

    scores, notes_by_source = {}, {}
    for src, fn in _SCORERS.items():
        s, notes = fn(diag, t)
        scores[src] = round(s, 4)
        notes_by_source[src] = notes

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    primary = ranked[0][0] if ranked and ranked[0][1] >= 0.45 else "UNCLASSIFIED"
    primary_score = ranked[0][1] if ranked else 0.0
    secondary = [k for k, v in ranked[1:] if v >= 0.45]
    confidence = ("high" if primary_score >= 0.7
                  else "medium" if primary_score >= 0.5 else "low")

    if primary == "UNCLASSIFIED":
        narrative = ("Insufficient diagnostic signal to identify a dominant edge "
                     "source. Run factor regression, OU mean-reversion, vol exposure, "
                     "and crash decomposition before deployment.")
    else:
        primary_notes = "; ".join(notes_by_source.get(primary, [])) or "no notes"
        sec = f" Secondary candidates: {', '.join(secondary)}." if secondary else ""
        narrative = (f"Primary edge source: {primary} (score {primary_score:.2f}, "
                     f"confidence {confidence}). Evidence: {primary_notes}.{sec}")

    return {
        "primary": primary, "secondary": secondary, "scores": scores,
        "narrative": narrative, "notes_by_source": notes_by_source,
        "confidence": confidence, "universe": universe,
    }
