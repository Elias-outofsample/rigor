"""Deep-diagnostics module — report-ready surface over ``rigor.analysis``.

Aggregates tail risk, risk battery, crisis decomposition, capacity, options-exposure
(Risk-DNA) and factor attribution into a single ``strategy_diagnostics`` call.
All blocks degrade gracefully on missing inputs or failures; no block ever raises.

Offline and deterministic — no network calls made here. Factor data must be supplied
by the caller (``load_french_factors`` is network-dependent).
"""
from __future__ import annotations

import html as _html
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..validation import robustness
from . import cost, exposure, factor, risk, structural, tail_risk
from . import regime as _regime

_log = logging.getLogger(__name__)

__all__ = [
    "strategy_diagnostics",
    "diagnostics_to_html",
    "diagnostics_summary",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_returns(returns: Any) -> pd.Series:
    """Return a clean pd.Series with a DatetimeIndex (best-effort)."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns, dtype=np.float64))
    returns = returns.dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        except Exception:
            pass
    return returns


def _safe(fn, *args, **kwargs):
    """Call *fn* and return its result; return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.debug("diagnostics block failed: %s", exc)
        return None


def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, dict):
        return not val or val.get("error") is not None
    if isinstance(val, (list, tuple)):
        return len(val) == 0
    return False


# ---------------------------------------------------------------------------
# New block helpers — pure functions, each _safe()-wrapped at call site
# ---------------------------------------------------------------------------

def _compute_regime_block(rets: pd.Series) -> dict | None:
    """Vol-regime performance using detect_market_vol_regime + compute_regime_performance."""
    if len(rets) < 126:
        return None
    regime_series = _regime.detect_market_vol_regime(rets)
    perf = _regime.compute_regime_performance(rets, regime_series)
    if not perf:
        return None
    # Compact: per-regime sharpe + pct_time only
    compact: dict[str, Any] = {}
    regime_labels = {0: "low_vol", 1: "mid_vol", 2: "high_vol"}
    for key, stats in perf.items():
        label = regime_labels.get(int(key), str(key))
        compact[label] = {
            "sharpe": stats.get("sharpe"),
            "pct_time": stats.get("pct_time"),
        }
    return compact


def _compute_monte_carlo_block(rets: pd.Series) -> dict | None:
    """Bootstrap Sharpe CI via robustness.bootstrap_sharpe_ci (seeded, deterministic)."""
    if len(rets) < 30:
        return None
    result = robustness.bootstrap_sharpe_ci(rets, n_bootstrap=2000, ci=0.95)
    if not result:
        return None
    sharpe = result.get("sharpe")
    ci_low = result.get("ci_lower")
    ci_high = result.get("ci_upper")
    p_pos = result.get("p_positive")
    return {
        "sharpe": sharpe,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_sharpe_gt_0": p_pos,
        "block_size": result.get("block_size"),
    }


def _compute_persistence_block(rets: pd.Series) -> dict | None:
    """Return-persistence via edge_decay slope + lag-1..5 autocorrelations."""
    if len(rets) < 60:
        return None
    decay = structural.edge_decay(rets)
    has_decay = bool(decay.get("has_decay", False))
    slope_annual = decay.get("slope_annual", 0.0)

    # Lag-1..5 autocorrelation
    arr = rets.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    acf: dict[str, float | None] = {}
    if len(arr) >= 10:
        mu = arr.mean()
        demeaned = arr - mu
        var = float(np.dot(demeaned, demeaned))
        for lag in range(1, 6):
            if len(arr) > lag and var > 0:
                cov = float(np.dot(demeaned[lag:], demeaned[:-lag]))
                acf[f"acf_lag{lag}"] = round(cov / var, 4)
            else:
                acf[f"acf_lag{lag}"] = None

    # Flag: decaying edge OR lag-1 ACF > 2/sqrt(n) (unusual persistence/anti-persistence)
    n_obs = len(arr)
    acf_thresh = 2.0 / (n_obs**0.5) if n_obs > 0 else 0.1
    lag1 = acf.get("acf_lag1")
    acf_flag = lag1 is not None and abs(lag1) > acf_thresh
    decay_flag = has_decay or acf_flag

    result: dict[str, Any] = {
        "has_decay": has_decay,
        "slope_annual": slope_annual,
        "decay_flag": decay_flag,
    }
    result.update(acf)
    return result


def _compute_concentration_block(rets: pd.Series) -> dict | None:
    """Single-month return concentration: largest month's share of total cum return."""
    if len(rets) < 20:
        return None
    if not isinstance(rets.index, pd.DatetimeIndex):
        return None

    # Compound to monthly returns
    monthly: pd.Series = (1 + rets).resample("ME").prod() - 1
    monthly = monthly.dropna()
    n_months = len(monthly)
    if n_months < 2:
        return None

    # Total cumulative return (can be negative — handle gracefully)
    cum_return = float((1 + monthly).prod() - 1)
    if abs(cum_return) < 1e-9:
        # Near-zero total: use absolute contribution share vs sum of absolute returns
        abs_sum = float(monthly.abs().sum())
        if abs_sum < 1e-12:
            return None
        shares = (monthly.abs() / abs_sum).fillna(0.0)
    else:
        # Sign-adjusted: each month's contribution to total (fraction of cum return)
        # Use absolute share of |total| for a meaningful concentration measure
        abs_monthly = monthly.abs()
        abs_sum = float(abs_monthly.sum())
        if abs_sum < 1e-12:
            return None
        shares = (abs_monthly / abs_sum).fillna(0.0)

    max_share = float(shares.max())
    max_month_idx = shares.idxmax()
    max_month = str(max_month_idx)[:7] if max_month_idx is not None else None

    if max_share > 0.30:
        level = "FAIL"
    elif max_share > 0.20:
        level = "WARN"
    else:
        level = "OK"

    return {
        "max_month_share": round(max_share, 4),
        "max_month": max_month,
        "n_months": int(n_months),
        "level": level,
    }


def _compute_decay_split_block(rets: pd.Series) -> dict | None:
    """Performance decay: first vs second half Sharpe and 4-sub-period slope."""
    if len(rets) < 60:
        return None

    from .. import metrics as _m

    ppy = (_m.periods_per_year_of(rets.index)
           if isinstance(rets.index, pd.DatetimeIndex) else _m.TRADING_DAYS)

    arr = rets.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    mid = n // 2

    sh_first = float(_m.compute_sharpe(arr[:mid], ppy))
    sh_second = float(_m.compute_sharpe(arr[mid:], ppy))

    # Decay fraction: (first - second) / |first|; undefined when first ≈ 0
    decay = float((sh_first - sh_second) / abs(sh_first)) if abs(sh_first) > 1e-6 else 0.0

    # 4-subperiod Sharpe slope (linear regression over sub-period index)
    n_subs = 4
    chunks = np.array_split(arr, n_subs)
    sub_sharpes = np.array([float(_m.compute_sharpe(c, ppy)) for c in chunks if len(c) >= 2])
    if len(sub_sharpes) >= 2:
        x = np.arange(len(sub_sharpes), dtype=np.float64)
        A = np.column_stack([np.ones(len(x)), x])
        slope_sub = float(np.linalg.lstsq(A, sub_sharpes, rcond=None)[0][1])
    else:
        slope_sub = 0.0

    if decay > 0.50:
        level = "FAIL"
    elif decay > 0.20:
        level = "WARN"
    else:
        level = "OK"

    return {
        "sharpe_first_half": round(sh_first, 4),
        "sharpe_second_half": round(sh_second, 4),
        "decay": round(decay, 4),
        "sub_period_slope": round(slope_sub, 4),
        "sub_period_sharpes": [round(float(s), 4) for s in sub_sharpes],
        "level": level,
    }


def _compute_corr_breakdown_block(rets: pd.Series, benchmark: pd.Series) -> dict | None:
    """Rolling-correlation breakdown vs benchmark; flags regime-shift count."""
    if len(rets) < 126 or benchmark is None:
        return None

    window = 252
    aligned = pd.DataFrame({"r": rets, "b": benchmark}).dropna()
    if len(aligned) < window + 10:
        return None

    rolling_corr = aligned["r"].rolling(window).corr(aligned["b"]).dropna()
    if len(rolling_corr) < 2:
        return None

    mean_rho = float(rolling_corr.mean())
    delta = rolling_corr.diff().abs().dropna()
    n_regime_shifts = int((delta > 0.30).sum())
    max_abs_delta = float(delta.max()) if len(delta) > 0 else 0.0
    breakdown_flag = n_regime_shifts >= 3

    return {
        "mean_rho": round(mean_rho, 4),
        "n_regime_shifts": n_regime_shifts,
        "max_abs_delta": round(max_abs_delta, 4),
        "breakdown_flag": breakdown_flag,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strategy_diagnostics(
    returns: pd.Series | np.ndarray,
    *,
    benchmark: pd.Series | None = None,
    factors: pd.DataFrame | None = None,
) -> dict:
    """Run the full deep-diagnostics battery on a strategy's return series.

    Parameters
    ----------
    returns:
        Daily (or periodic) return series. pd.Series with DatetimeIndex preferred;
        arrays are accepted and coerced.
    benchmark:
        Optional benchmark return series aligned to the same calendar.  Required for
        ``risk_dna`` (options-exposure profile) and ``corr_breakdown``.
        Passed through to ``decompose_by_crisis`` as well.
    factors:
        Optional factor panel (columns Mkt-RF/SMB/HML/RF etc.). When supplied and
        non-empty, a factor regression is added.  Do NOT pass ``load_french_factors``
        output here — that needs a network call; supply your own pre-loaded frame.

    Returns
    -------
    dict with keys:
        ``tail``, ``risk``, ``crisis``, ``capacity`` — always attempted.
        ``risk_dna`` — only when *benchmark* is provided.
        ``factor`` — only when *factors* is a non-empty DataFrame.
        ``regime`` — vol-regime performance breakdown (always attempted).
        ``monte_carlo`` — bootstrap Sharpe CI (always attempted).
        ``persistence`` — return autocorrelation + edge-decay (always attempted).
        ``concentration`` — single-month return concentration (always attempted).
        ``decay_split`` — first-half vs second-half performance decay (always attempted).
        ``corr_breakdown`` — rolling-correlation regime shifts (only when *benchmark* provided).
        ``available`` — list of block names that are non-empty.
    """
    rets = _coerce_returns(returns)

    # --- tail -----------------------------------------------------------------
    tail_block: dict | None = None
    if not _is_empty(rets):
        raw = _safe(tail_risk.compute_extreme_risk_measures, rets)
        gpd = _safe(tail_risk.fit_gpd_tail, rets)
        kupiec = _safe(tail_risk.kupiec_pof_test, rets, var_quantile=0.05)
        if raw is not None:
            tail_block = {**raw, "fit_gpd": gpd, "kupiec_var95": kupiec}

    # --- risk battery ---------------------------------------------------------
    risk_block: dict | None = _safe(risk.risk_summary, rets)

    # --- crisis decomposition -------------------------------------------------
    crisis_block: dict | None = _safe(
        exposure.decompose_by_crisis, rets, benchmark
    )

    # --- capacity -------------------------------------------------------------
    cap_raw: dict | None = _safe(cost.estimate_aum_capacity, rets)
    capacity_block: dict | None = None
    if cap_raw is not None:
        # Keep headline numbers; include a 5-point downsampled sharpe_vs_aum curve.
        curve = cap_raw.get("sharpe_vs_aum", [])
        n = len(curve)
        if n > 5:
            step = max(1, n // 5)
            curve = curve[::step][:5]
        capacity_block = {
            "capacity_low": cap_raw.get("capacity_low"),
            "capacity_central": cap_raw.get("capacity_central"),
            "capacity_high": cap_raw.get("capacity_high"),
            "base_sharpe": cap_raw.get("base_sharpe"),
            "sharpe_vs_aum": curve,
        }

    # --- risk DNA (options-exposure profile) — benchmark required -------------
    risk_dna_block: dict | None = None
    if benchmark is not None and not _is_empty(rets):
        risk_dna_block = _safe(exposure.compute_greeks_summary, rets, benchmark)

    # --- factor regression — factors required ---------------------------------
    factor_block: dict | None = None
    if (
        factors is not None
        and isinstance(factors, pd.DataFrame)
        and not factors.empty
    ):
        factor_block = _safe(
            factor.compute_factor_regression, rets, factors, use_hac=True
        )
        if isinstance(factor_block, dict) and factor_block.get("error"):
            factor_block = None

    # --- regime block (vol-regime performance) --------------------------------
    regime_block: dict | None = _safe(_compute_regime_block, rets)

    # --- monte_carlo (bootstrap Sharpe CI) ------------------------------------
    monte_carlo_block: dict | None = _safe(_compute_monte_carlo_block, rets)

    # --- persistence (autocorrelation + edge decay) ---------------------------
    persistence_block: dict | None = _safe(_compute_persistence_block, rets)

    # --- concentration (single-month share) -----------------------------------
    concentration_block: dict | None = _safe(_compute_concentration_block, rets)

    # --- decay_split (first vs second half) -----------------------------------
    decay_split_block: dict | None = _safe(_compute_decay_split_block, rets)

    # --- corr_breakdown — benchmark required ----------------------------------
    corr_breakdown_block: dict | None = None
    if benchmark is not None and not _is_empty(rets):
        corr_breakdown_block = _safe(_compute_corr_breakdown_block, rets, benchmark)

    # --- assemble result -------------------------------------------------------
    available = [
        name
        for name, block in [
            ("tail", tail_block),
            ("risk", risk_block),
            ("crisis", crisis_block),
            ("capacity", capacity_block),
            ("risk_dna", risk_dna_block),
            ("factor", factor_block),
            ("regime", regime_block),
            ("monte_carlo", monte_carlo_block),
            ("persistence", persistence_block),
            ("concentration", concentration_block),
            ("decay_split", decay_split_block),
            ("corr_breakdown", corr_breakdown_block),
        ]
        if not _is_empty(block)
    ]

    return {
        "tail": tail_block,
        "risk": risk_block,
        "crisis": crisis_block,
        "capacity": capacity_block,
        "risk_dna": risk_dna_block,
        "factor": factor_block,
        "regime": regime_block,
        "monte_carlo": monte_carlo_block,
        "persistence": persistence_block,
        "concentration": concentration_block,
        "decay_split": decay_split_block,
        "corr_breakdown": corr_breakdown_block,
        "available": available,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _fmt(val: Any) -> str:
    """Format a scalar value for display in HTML."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if not np.isfinite(val):
            return "—"
        if abs(val) < 0.001:
            return f"{val:.2e}"
        return f"{val:.4f}"
    if isinstance(val, int):
        return f"{val:,}"
    return _html.escape(str(val))


def _kv_table(rows: list[tuple[str, Any]], title: str) -> str:
    """Render a key/value HTML table with a title row."""
    lines = [f"<h3>{_html.escape(title)}</h3>", '<table class="kpis">']
    for k, v in rows:
        lines.append(
            f"  <tr><td>{_html.escape(str(k))}</td><td>{_fmt(v)}</td></tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def _tail_html(block: dict) -> str:
    gpd = block.get("fit_gpd") or block.get("gpd") or {}
    kupiec = block.get("kupiec_var95") or {}
    rows: list[tuple[str, Any]] = [
        ("Hill tail index α", block.get("hill_alpha")),
        ("Fat-tailed (α < 4)", block.get("fat_tailed")),
        ("GPD shape ξ", gpd.get("xi")),
        ("GPD scale σ", gpd.get("sigma")),
        ("GPD VaR-99 (EVT)", gpd.get("var_99")),
        ("GPD CVaR-99 (EVT)", gpd.get("cvar_99")),
        ("Kupiec n_obs", kupiec.get("n_obs")),
        ("Kupiec breaches", kupiec.get("n_breaches")),
        ("Kupiec expected", kupiec.get("expected_breaches")),
        ("Kupiec p-value", kupiec.get("p_value")),
        ("Kupiec verdict", kupiec.get("verdict")),
    ]
    return _kv_table([(k, v) for k, v in rows if v is not None], "Tail Risk (EVT)")


def _risk_html(block: dict) -> str:
    label_map = {
        "var_95": "VaR-95",
        "cvar_95": "CVaR-95",
        "cornish_fisher_var_95": "Cornish-Fisher VaR-95",
        "omega": "Omega",
        "tail_ratio": "Tail Ratio",
        "rachev": "Rachev Ratio",
        "gain_to_pain": "Gain-to-Pain",
        "ulcer_index": "Ulcer Index",
        "upi": "Ulcer Performance Index",
        "cdar_95": "CDaR-95",
        "pain_index": "Pain Index",
        "sterling": "Sterling Ratio",
        "burke": "Burke Ratio",
    }
    rows = [(label_map.get(k, k), v) for k, v in block.items() if k in label_map]
    return _kv_table(rows, "Risk Battery")


def _crisis_html(block: dict) -> str:
    rows: list[tuple[str, Any]] = [
        ("Full-sample Sharpe", block.get("full_sample_sharpe")),
        ("Mean crisis Sharpe", block.get("mean_crisis_sharpe")),
        ("Crash Sharpe ratio", block.get("crash_sharpe_ratio")),
        ("Crisis verdict", block.get("verdict")),
        ("Crises evaluated", block.get("n_crises_evaluated")),
    ]
    worst = block.get("worst_crisis")
    if worst:
        rows.append(("Worst crisis", worst.get("label")))
        rows.append(("Worst crisis Sharpe", worst.get("sharpe")))
        rows.append(("Worst crisis MDD", worst.get("max_dd")))
    return _kv_table([(k, v) for k, v in rows if v is not None], "Crisis Decomposition")


def _capacity_html(block: dict) -> str:
    def _aum(v: Any) -> str:
        if v is None or not np.isfinite(float(v)):
            return "—"
        v = float(v)
        if v >= 1e9:
            return f"${v / 1e9:.1f}B"
        if v >= 1e6:
            return f"${v / 1e6:.1f}M"
        if v >= 1e3:
            return f"${v / 1e3:.0f}K"
        return f"${v:.0f}"

    rows: list[tuple[str, str]] = [
        ("Base Sharpe", _fmt(block.get("base_sharpe"))),
        ("Capacity low (10 bps impact)", _aum(block.get("capacity_low"))),
        ("Capacity central (20 bps)", _aum(block.get("capacity_central"))),
        ("Capacity high (50 bps)", _aum(block.get("capacity_high"))),
    ]
    return _kv_table(rows, "AUM Capacity")


def _risk_dna_html(block: dict) -> str:
    payoff = block.get("payoff") or {}
    vol_e = (block.get("vol_exposure") or {})
    gamma_e = (block.get("gamma_exposure") or {})
    theta_e = (block.get("theta_exposure") or {})
    rows: list[tuple[str, Any]] = [
        ("Overall profile", block.get("overall_profile")),
        ("Payoff type", payoff.get("payoff_type")),
        ("Vol exposure", vol_e.get("exposure")),
        ("Vol correlation (ρ)", vol_e.get("correlation")),
        ("Gamma exposure", gamma_e.get("exposure")),
        ("Gamma coefficient", gamma_e.get("gamma")),
        ("Gamma t-stat", gamma_e.get("gamma_t_stat")),
        ("Theta exposure", theta_e.get("exposure")),
        ("Win rate", theta_e.get("win_rate")),
        ("Trend convexity", block.get("trend_convexity")),
    ]
    return _kv_table([(k, v) for k, v in rows if v is not None], "Risk-DNA / Options Exposure")


def _factor_html(block: dict) -> str:
    rows: list[tuple[str, Any]] = [
        ("Alpha (annualised)", block.get("alpha")),
        ("Alpha t-stat", block.get("alpha_t_stat")),
        ("Alpha p-value", block.get("alpha_p_value")),
        ("Alpha net", block.get("alpha_net")),
        ("R²", block.get("r_squared")),
        ("Adj R²", block.get("adj_r_squared")),
        ("SE type", block.get("se_type")),
        ("N obs", block.get("n_obs")),
    ]
    betas = block.get("betas") or {}
    for fname, bval in betas.items():
        rows.append((f"β {fname}", bval))
    return _kv_table([(k, v) for k, v in rows if v is not None], "Factor Regression")


def _regime_html(block: dict) -> str:
    regime_labels = {"low_vol": "Low Vol", "mid_vol": "Mid Vol", "high_vol": "High Vol"}
    rows: list[tuple[str, Any]] = []
    for key, stats in block.items():
        label = regime_labels.get(str(key), str(key))
        if isinstance(stats, dict):
            rows.append((f"{label} — Sharpe", stats.get("sharpe")))
            pct = stats.get("pct_time")
            rows.append((f"{label} — % time", f"{pct * 100:.1f}%" if pct is not None else None))
    return _kv_table([(k, v) for k, v in rows if v is not None], "Vol-Regime Performance")


def _monte_carlo_html(block: dict) -> str:
    rows: list[tuple[str, Any]] = [
        ("Sharpe (point estimate)", block.get("sharpe")),
        ("CI 95% lower", block.get("ci_low")),
        ("CI 95% upper", block.get("ci_high")),
        ("P(Sharpe > 0)", block.get("p_sharpe_gt_0")),
        ("Bootstrap block size", block.get("block_size")),
    ]
    title = "Monte Carlo / Bootstrap Sharpe CI"
    return _kv_table([(k, v) for k, v in rows if v is not None], title)


def _persistence_html(block: dict) -> str:
    rows: list[tuple[str, Any]] = [
        ("Edge decay flag", block.get("decay_flag")),
        ("Has rolling-Sharpe decay", block.get("has_decay")),
        ("Slope (annualised)", block.get("slope_annual")),
    ]
    for lag in range(1, 6):
        key = f"acf_lag{lag}"
        if key in block:
            rows.append((f"ACF lag-{lag}", block.get(key)))
    return _kv_table([(k, v) for k, v in rows if v is not None], "Return Persistence / Edge Decay")


def _concentration_html(block: dict) -> str:
    level = block.get("level", "")
    flag_str = f"[{level}]" if level else ""
    rows: list[tuple[str, Any]] = [
        ("Flag", flag_str),
        ("Largest month share", block.get("max_month_share")),
        ("Largest month", block.get("max_month")),
        ("Number of months", block.get("n_months")),
    ]
    return _kv_table([(k, v) for k, v in rows if v is not None], "Single-Month Concentration")


def _decay_split_html(block: dict) -> str:
    level = block.get("level", "")
    flag_str = f"[{level}]" if level else ""
    rows: list[tuple[str, Any]] = [
        ("Flag", flag_str),
        ("First-half Sharpe", block.get("sharpe_first_half")),
        ("Second-half Sharpe", block.get("sharpe_second_half")),
        ("Decay fraction", block.get("decay")),
        ("Sub-period slope", block.get("sub_period_slope")),
        ("Sub-period Sharpes", str(block.get("sub_period_sharpes", []))),
    ]
    title = "Performance Decay (Half-Period Split)"
    return _kv_table([(k, v) for k, v in rows if v is not None], title)


def _corr_breakdown_html(block: dict) -> str:
    flag = block.get("breakdown_flag")
    flag_str = "[WARN: breakdown detected]" if flag else "[OK]"
    rows: list[tuple[str, Any]] = [
        ("Flag", flag_str),
        ("Mean rolling ρ (252d)", block.get("mean_rho")),
        ("Regime shifts (|Δρ| > 0.3)", block.get("n_regime_shifts")),
        ("Max |Δρ|", block.get("max_abs_delta")),
    ]
    return _kv_table([(k, v) for k, v in rows if v is not None], "Rolling-Correlation Breakdown")


def diagnostics_to_html(diag: dict) -> str:
    """Render the diagnostics dict as a self-contained HTML fragment.

    Returns an empty string when no blocks are populated.
    Only blocks listed in ``diag['available']`` are rendered.
    """
    available: list[str] = diag.get("available", [])
    if not available:
        return ""

    _renderers = {
        "tail": _tail_html,
        "risk": _risk_html,
        "crisis": _crisis_html,
        "capacity": _capacity_html,
        "risk_dna": _risk_dna_html,
        "factor": _factor_html,
        "regime": _regime_html,
        "monte_carlo": _monte_carlo_html,
        "persistence": _persistence_html,
        "concentration": _concentration_html,
        "decay_split": _decay_split_html,
        "corr_breakdown": _corr_breakdown_html,
    }

    parts: list[str] = ['<div class="section">']
    for name in available:
        block = diag.get(name)
        renderer = _renderers.get(name)
        if block and renderer:
            try:
                parts.append(renderer(block))
            except Exception as exc:  # noqa: BLE001
                _log.debug("HTML render failed for %s: %s", name, exc)
    parts.append("</div>")

    # If only the outer div was added (nothing rendered), return empty.
    if len(parts) == 2:
        return ""
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

def _json_safe(val: Any) -> float | str | bool | None:
    """Coerce a value to a JSON-serialisable scalar."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        v = float(val)
        return v if np.isfinite(v) else None
    if isinstance(val, str):
        return val
    return str(val)


def diagnostics_summary(diag: dict) -> dict:
    """Flat JSON-serialisable dict of headline numbers across all populated blocks.

    Suitable for embedding in ``summary.json`` alongside the strategy's core metrics.
    """
    out: dict[str, float | str | bool | None] = {}
    available: list[str] = diag.get("available", [])
    out["available_blocks"] = ", ".join(available) if available else None

    # tail
    tail = diag.get("tail") or {}
    gpd_data = tail.get("fit_gpd") or tail.get("gpd") or {}
    kupiec_data = tail.get("kupiec_var95") or {}
    out["var_99_gpd"] = _json_safe(gpd_data.get("var_99"))
    out["cvar_99_gpd"] = _json_safe(gpd_data.get("cvar_99"))
    out["gpd_xi"] = _json_safe(gpd_data.get("xi"))
    out["gpd_sigma"] = _json_safe(gpd_data.get("sigma"))
    out["hill_alpha"] = _json_safe(tail.get("hill_alpha"))
    out["fat_tailed"] = _json_safe(tail.get("fat_tailed"))
    out["kupiec_verdict"] = _json_safe(kupiec_data.get("verdict"))
    out["kupiec_p_value"] = _json_safe(kupiec_data.get("p_value"))

    # risk battery
    risk_block = diag.get("risk") or {}
    out["var_95"] = _json_safe(risk_block.get("var_95"))
    out["cvar_95"] = _json_safe(risk_block.get("cvar_95"))
    out["omega"] = _json_safe(risk_block.get("omega"))
    out["tail_ratio"] = _json_safe(risk_block.get("tail_ratio"))
    out["gain_to_pain"] = _json_safe(risk_block.get("gain_to_pain"))
    out["ulcer_index"] = _json_safe(risk_block.get("ulcer_index"))
    out["upi"] = _json_safe(risk_block.get("upi"))
    out["sterling"] = _json_safe(risk_block.get("sterling"))
    out["burke"] = _json_safe(risk_block.get("burke"))

    # crisis
    crisis = diag.get("crisis") or {}
    out["crash_sharpe_ratio"] = _json_safe(crisis.get("crash_sharpe_ratio"))
    out["crisis_verdict"] = _json_safe(crisis.get("verdict"))
    out["mean_crisis_sharpe"] = _json_safe(crisis.get("mean_crisis_sharpe"))
    worst = crisis.get("worst_crisis") or {}
    out["worst_crisis_label"] = _json_safe(worst.get("label"))
    out["worst_crisis_sharpe"] = _json_safe(worst.get("sharpe"))

    # capacity
    cap = diag.get("capacity") or {}
    out["capacity_low"] = _json_safe(cap.get("capacity_low"))
    out["capacity_central"] = _json_safe(cap.get("capacity_central"))
    out["capacity_high"] = _json_safe(cap.get("capacity_high"))
    out["capacity_base_sharpe"] = _json_safe(cap.get("base_sharpe"))

    # risk_dna
    rdna = diag.get("risk_dna") or {}
    payoff = rdna.get("payoff") or {}
    vol_e = rdna.get("vol_exposure") or {}
    gamma_e = rdna.get("gamma_exposure") or {}
    theta_e = rdna.get("theta_exposure") or {}
    out["risk_dna_payoff_type"] = _json_safe(payoff.get("payoff_type"))
    out["risk_dna_vol_exposure"] = _json_safe(vol_e.get("exposure"))
    out["risk_dna_gamma_exposure"] = _json_safe(gamma_e.get("exposure"))
    out["risk_dna_theta_exposure"] = _json_safe(theta_e.get("exposure"))
    out["risk_dna_profile"] = _json_safe(rdna.get("overall_profile"))

    # factor
    fac = diag.get("factor") or {}
    out["factor_alpha"] = _json_safe(fac.get("alpha"))
    out["factor_alpha_t"] = _json_safe(fac.get("alpha_t_stat"))
    out["factor_alpha_p"] = _json_safe(fac.get("alpha_p_value"))
    out["factor_r_squared"] = _json_safe(fac.get("r_squared"))

    # regime — best and worst Sharpe across vol regimes
    reg = diag.get("regime") or {}
    regime_sharpes: list[float] = []
    regime_labels_seen: list[str] = []
    for rk, rv in reg.items():
        if isinstance(rv, dict) and rv.get("sharpe") is not None:
            sh = rv["sharpe"]
            if np.isfinite(float(sh)):
                regime_sharpes.append(float(sh))
                regime_labels_seen.append(str(rk))
    if regime_sharpes:
        best_idx = int(np.argmax(regime_sharpes))
        worst_idx = int(np.argmin(regime_sharpes))
        out["regime_best"] = _json_safe(regime_labels_seen[best_idx])
        out["regime_best_sharpe"] = _json_safe(regime_sharpes[best_idx])
        out["regime_worst"] = _json_safe(regime_labels_seen[worst_idx])
        out["regime_worst_sharpe"] = _json_safe(regime_sharpes[worst_idx])
    else:
        out["regime_best"] = None
        out["regime_best_sharpe"] = None
        out["regime_worst"] = None
        out["regime_worst_sharpe"] = None

    # monte_carlo
    mc = diag.get("monte_carlo") or {}
    out["mc_sharpe"] = _json_safe(mc.get("sharpe"))
    out["mc_sharpe_ci_low"] = _json_safe(mc.get("ci_low"))
    out["mc_sharpe_ci_high"] = _json_safe(mc.get("ci_high"))
    out["mc_p_sharpe_gt_0"] = _json_safe(mc.get("p_sharpe_gt_0"))

    # persistence
    pers = diag.get("persistence") or {}
    out["edge_decay_flag"] = _json_safe(pers.get("decay_flag"))
    out["edge_decay_slope_annual"] = _json_safe(pers.get("slope_annual"))
    out["acf_lag1"] = _json_safe(pers.get("acf_lag1"))

    # concentration
    conc = diag.get("concentration") or {}
    out["max_month_share"] = _json_safe(conc.get("max_month_share"))
    out["concentration_level"] = _json_safe(conc.get("level"))
    out["concentration_max_month"] = _json_safe(conc.get("max_month"))

    # decay_split
    ds = diag.get("decay_split") or {}
    out["perf_decay"] = _json_safe(ds.get("decay"))
    out["perf_decay_level"] = _json_safe(ds.get("level"))
    out["perf_decay_sharpe_first"] = _json_safe(ds.get("sharpe_first_half"))
    out["perf_decay_sharpe_second"] = _json_safe(ds.get("sharpe_second_half"))

    # corr_breakdown
    cb = diag.get("corr_breakdown") or {}
    out["corr_breakdown_mean_rho"] = _json_safe(cb.get("mean_rho"))
    out["corr_breakdown_shifts"] = _json_safe(cb.get("n_regime_shifts"))
    out["corr_breakdown_flag"] = _json_safe(cb.get("breakdown_flag"))

    # Verify JSON-serialisability (guard against future bugs silently)
    try:
        json.dumps(out)
    except (TypeError, ValueError) as exc:
        _log.warning("diagnostics_summary: serialisation guard failed — %s", exc)

    return out
