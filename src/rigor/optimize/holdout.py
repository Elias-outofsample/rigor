"""Holdout validation — a never-optimised final slice.

Split the winner's return series into three contiguous windows by date:
in-sample (0-60%), out-of-sample (60-80%), and a **holdout** (final 20%) that
the search never sees. If the holdout Sharpe collapses relative to the OOS
Sharpe, the winner is overfit or regime-fragile. Ported from the prior research
library's stage-4.5 holdout (degradation rule: holdout < 0, or < 30% of OOS).
"""
from __future__ import annotations

import pandas as pd

from .. import metrics as _m

__all__ = ["holdout_validate"]


def holdout_validate(returns: pd.Series, ppy: int | None = None,
                     *, oos_frac: float = 0.60, holdout_frac: float = 0.80) -> dict:
    """IS/OOS/holdout Sharpe split on the winner's returns + a ``degraded`` flag."""
    r = returns.dropna()
    n = len(r)
    if n <= 50:
        return {"note": "series too short for holdout (need > 50 bars)"}
    ppy = ppy or _m.periods_per_year_of(r.index)
    i_oos, i_hold = int(n * oos_frac), int(n * holdout_frac)

    def _sh(block: pd.Series) -> float:
        return _m.compute_sharpe(block.to_numpy(dtype="float64"), ppy) if len(block) >= 2 else 0.0

    is_sh = _sh(r.iloc[:i_oos])
    oos_sh = _sh(r.iloc[i_oos:i_hold])
    hold_sh = _sh(r.iloc[i_hold:])
    degraded = bool(hold_sh < 0.0 or (oos_sh > 0 and hold_sh < 0.30 * oos_sh))
    return {
        "is_sharpe": round(float(is_sh), 4), "oos_sharpe": round(float(oos_sh), 4),
        "holdout_sharpe": round(float(hold_sh), 4),
        "is_n": i_oos, "oos_n": i_hold - i_oos, "holdout_n": n - i_hold,
        "degraded": degraded,
    }
