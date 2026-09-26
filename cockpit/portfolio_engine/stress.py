"""Stress testing — replay the portfolio through historical crisis windows.

Each scenario is a named date range; the portfolio (or benchmark) return series is
sliced to it and the crisis-period Sharpe / CAGR / MaxDD / CVaR / end-equity are
reported. Ported from the prior research library's stress scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rigor import metrics as _m

from . import risk as _risk

__all__ = [
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "run_scenario",
    "run_all_scenarios",
    "stress_vs_calm_correlation",
]


@dataclass(frozen=True)
class StressScenario:
    name: str
    start: str
    end: str


HISTORICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario("GFC 2008-09", "2007-10-01", "2009-03-31"),
    StressScenario("Dotcom", "2000-03-01", "2002-10-31"),
    StressScenario("COVID 2020", "2020-02-01", "2020-04-30"),
    StressScenario("Volmageddon 2018", "2018-01-15", "2018-02-28"),
    StressScenario("Bonds 2022", "2022-01-01", "2022-12-31"),
    StressScenario("SVB 2023", "2023-03-01", "2023-06-30"),
)


def run_scenario(returns: pd.Series, scenario: StressScenario) -> dict:
    """Slice ``returns`` to the scenario window and report crisis-period metrics."""
    s = returns.dropna()
    window = s.loc[(s.index >= pd.Timestamp(scenario.start))
                   & (s.index <= pd.Timestamp(scenario.end))]
    if len(window) < 2:
        return {"name": scenario.name, "n_obs": len(window), "covered": False}
    arr = window.to_numpy(dtype="float64")
    ppy = _m.periods_per_year_of(window.index)
    return {
        "name": scenario.name, "covered": True, "n_obs": len(window),
        "sharpe": float(_m.compute_sharpe(arr, ppy)), "cagr": float(_m.compute_cagr(arr, ppy)),
        "max_dd": float(_m.compute_max_drawdown(arr)),
        "total_return": float((1.0 + window).prod() - 1.0),
        "cvar_95": _risk.conditional_var(window, 0.05),
    }


def run_all_scenarios(returns: pd.Series) -> list[dict]:
    """Run every historical scenario the return series covers."""
    return [r for sc in HISTORICAL_SCENARIOS if (r := run_scenario(returns, sc))["covered"]]


# ---------------------------------------------------------------------------
# False-diversification detector
# ---------------------------------------------------------------------------

def _avg_offdiagonal(corr: np.ndarray) -> float:
    """Mean of all off-diagonal entries of a square correlation matrix."""
    n = corr.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    off = corr[iu]
    finite = off[np.isfinite(off)]
    return float(finite.mean()) if len(finite) else 0.0


def stress_vs_calm_correlation(
    returns_matrix: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    stress_quantile: float = 0.10,
) -> dict:
    """Split the timeline into stress vs calm days and compute cross-leg correlations.

    Stress days are the worst ``stress_quantile`` fraction of days, ranked by the
    ``benchmark`` series when provided, or by the equal-weight average of all legs
    otherwise (the portfolio's own down-days).

    Parameters
    ----------
    returns_matrix:
        Per-leg daily returns, one column per leg.  NaNs are dropped row-wise.
    benchmark:
        External market proxy used to rank days.  Optional.
    stress_quantile:
        Fraction of worst days classified as stress (default 0.10 = bottom decile).

    Returns
    -------
    dict with keys:

    ``assets``
        List of column names.
    ``calm_corr``, ``stress_corr``, ``delta_corr``
        2-D lists (square, side = n_legs) of calm/stress/stress-minus-calm correlations.
    ``avg_offdiag_calm``, ``avg_offdiag_stress``
        Mean off-diagonal correlation in each regime.
    ``corr_increase``
        ``avg_offdiag_stress - avg_offdiag_calm``.
    ``false_diversification_flag``
        True when ``corr_increase > 0.2`` *and* there are >= 30 observations in each
        regime.  Signals correlation that evaporates during market stress.
    ``n_stress_days``, ``n_calm_days``
        Number of observations in each regime.
    ``note``
        Present only when the function falls back to a degenerate result (e.g. too
        few observations or a single leg).
    """
    assets = list(returns_matrix.columns)
    n_legs = len(assets)

    _empty_matrices: list[list] = []
    _zero_result = {
        "assets": assets,
        "calm_corr": _empty_matrices,
        "stress_corr": _empty_matrices,
        "delta_corr": _empty_matrices,
        "avg_offdiag_calm": 0.0,
        "avg_offdiag_stress": 0.0,
        "corr_increase": 0.0,
        "false_diversification_flag": False,
        "n_stress_days": 0,
        "n_calm_days": 0,
    }

    # --- single-leg guard ---------------------------------------------------
    if n_legs < 2:
        return {**_zero_result, "note": "single leg — correlation undefined"}

    # --- drop rows with any NaN across legs ---------------------------------
    rm = returns_matrix[assets].dropna(how="any")

    if len(rm) < 2:
        return {**_zero_result, "note": "insufficient observations after dropping NaNs"}

    # --- build the ranking proxy --------------------------------------------
    if benchmark is not None:
        proxy = benchmark.reindex(rm.index).dropna()
        rm = rm.reindex(proxy.index).dropna(how="any")
        proxy = proxy.reindex(rm.index)
    else:
        proxy = rm.mean(axis=1)  # equal-weight portfolio daily return

    # --- stress / calm split ------------------------------------------------
    threshold = float(proxy.quantile(stress_quantile))
    stress_mask = proxy <= threshold
    calm_mask = ~stress_mask

    n_stress = int(stress_mask.sum())
    n_calm = int(calm_mask.sum())

    # Robustness guard: < 30 observations in either regime
    if n_stress < 30 or n_calm < 30:
        return {
            **_zero_result,
            "n_stress_days": n_stress,
            "n_calm_days": n_calm,
            "note": (
                f"too few observations in stress ({n_stress}) or calm ({n_calm}) regime "
                f"— need >= 30 in each"
            ),
        }

    stress_df = rm.loc[stress_mask]
    calm_df = rm.loc[calm_mask]

    # --- correlation matrices -----------------------------------------------
    def _safe_corr(df: pd.DataFrame) -> np.ndarray:
        """Pearson correlation matrix; fill non-finite entries with 0.0."""
        c = df.corr(method="pearson").to_numpy(dtype=float)
        c = np.where(np.isfinite(c), c, 0.0)
        np.fill_diagonal(c, 1.0)
        return c

    calm_c = _safe_corr(calm_df)
    stress_c = _safe_corr(stress_df)
    delta_c = stress_c - calm_c

    avg_calm = _avg_offdiagonal(calm_c)
    avg_stress = _avg_offdiagonal(stress_c)
    increase = avg_stress - avg_calm

    flag = bool(increase > 0.2)

    return {
        "assets": assets,
        "calm_corr": calm_c.tolist(),
        "stress_corr": stress_c.tolist(),
        "delta_corr": delta_c.tolist(),
        "avg_offdiag_calm": avg_calm,
        "avg_offdiag_stress": avg_stress,
        "corr_increase": increase,
        "false_diversification_flag": flag,
        "n_stress_days": n_stress,
        "n_calm_days": n_calm,
    }
