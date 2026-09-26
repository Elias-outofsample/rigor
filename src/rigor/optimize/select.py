"""Config selection: how to pick the "best" config from a searched grid.

Two modes:

  * ``sharpe`` (default) — the highest in-sample Sharpe. Simple and transparent;
    the anti-overfit machinery (deflated Sharpe, data-snooping tests, CPCV-PBO)
    then tells you how much to trust that winner.
  * ``robust`` — a two-stage lexicographic ladder ported from the prior research
    library. **Stage A** keeps only configs that clear every robustness floor
    (Sharpe, PSR, no wipe-out) and rejects the whole grid if it is overfit-prone
    (CPCV-PBO too high). **Stage B** then maximises ``CAGR x log(1 + active bars)``
    among the robustness-equivalent survivors — encoding "at equal robustness,
    prefer the bigger compounding with more activity (more activity = more
    reliable statistics)". If nothing clears the floors it returns ``REJECTED``
    — the honest answer that the grid has no robust edge, rather than shipping
    the prettiest overfit config.

Both reuse ``rigor.metrics`` and ``rigor.validation`` primitives — no parallel
scoring system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import metrics as _m
from ..validation.overfit import probabilistic_sharpe_ratio

__all__ = ["RobustFloors", "config_metrics", "select"]


@dataclass(frozen=True)
class RobustFloors:
    """Acceptance floors for the ``robust`` selection mode (sourced defaults)."""
    sharpe: float = 0.3       # per-config annualised Sharpe floor
    psr: float = 0.6          # P(true SR > 0) floor
    cagr: float = -0.05       # reject configs losing > 5%/yr
    max_dd: float = -0.999    # reject near-total wipe-outs
    pbo: float = 0.5          # grid-level: reject if CPCV-PBO >= this


def config_metrics(returns: pd.Series) -> dict:
    """Per-config metrics used by selection, from a returns series."""
    r = returns.dropna()
    arr = r.to_numpy(dtype="float64")
    n = len(arr)
    if n < 2:
        return {"sharpe": float("nan"), "cagr": float("nan"), "max_dd": float("nan"),
                "psr": 0.0, "active_bars": 0}
    ppy = _m.periods_per_year_of(r.index)
    sharpe_ann = _m.compute_sharpe(arr, ppy)
    from scipy.stats import kurtosis, skew
    sk = float(skew(arr, bias=False)) if n > 2 else 0.0
    ku = float(kurtosis(arr, fisher=True, bias=False)) if n > 3 else 0.0
    psr = probabilistic_sharpe_ratio(sharpe_ann / np.sqrt(ppy), 0.0, n, sk, ku)
    return {
        "sharpe": sharpe_ann,
        "cagr": _m.compute_cagr(arr, ppy),
        "max_dd": _m.compute_max_drawdown(arr),
        "psr": psr,
        "active_bars": int(np.count_nonzero(arr)),
    }


def select(rows: list[dict], *, mode: str = "sharpe", floors: RobustFloors | None = None,
           pbo: float | None = None) -> dict:
    """Pick the winning config.

    ``rows`` is the ranked list of ``{"params", "sharpe", "returns", ...}`` dicts
    (already sorted by Sharpe, best first). Returns
    ``{"index", "params", "priority", ...}``; ``index`` indexes into ``rows`` and
    is ``None`` when the grid is rejected.
    """
    if mode == "sharpe":
        return {"index": 0, "params": rows[0]["params"], "priority": "TOP_SHARPE"}
    if mode != "robust":
        raise ValueError(f"mode must be 'sharpe' or 'robust', got {mode!r}")

    floors = floors or RobustFloors()
    if pbo is not None and float(pbo) >= floors.pbo:
        return {"index": None, "params": {}, "priority": "REJECTED",
                "reason": f"grid CPCV-PBO {float(pbo):.2f} >= {floors.pbo} (overfit-prone)"}

    survivors = []
    for i, row in enumerate(rows):
        met = row.get("metrics") or config_metrics(row["returns"])
        if not np.isfinite(met["sharpe"]):
            continue
        if (met["sharpe"] >= floors.sharpe and met["psr"] >= floors.psr
                and met["cagr"] > floors.cagr and met["max_dd"] > floors.max_dd):
            survivors.append((i, met))
    if not survivors:
        return {"index": None, "params": {}, "priority": "REJECTED",
                "reason": (f"no config cleared the floors (Sharpe>={floors.sharpe}, "
                           f"PSR>={floors.psr})")}

    # Stage B — maximise CAGR x log(1 + active bars), tie-break Sharpe then CAGR.
    def key(item):
        _, met = item
        perf = max(met["cagr"], 0.0) * np.log1p(met["active_bars"])
        return (perf, met["sharpe"], met["cagr"])

    best_i, best_met = max(survivors, key=key)
    return {
        "index": best_i, "params": rows[best_i]["params"], "priority": "ROBUST_PERF",
        "n_robust_candidates": len(survivors), "selected_metrics": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in best_met.items()},
    }
