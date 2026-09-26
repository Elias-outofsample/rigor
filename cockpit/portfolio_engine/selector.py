"""Selection: filter candidates by robustness constraints, then decorrelate.

``PortfolioSelector.select()`` first drops candidates that fail the user's floors
(Sharpe / CAGR / DSR / OOS-Sharpe / PBO / walk-forward / verdict / n_obs), then
greedily removes the weaker of any pair whose absolute correlation exceeds
``corr_max``. Hedges (negative standalone Sharpe by design) bypass the PnL floors
when ``allow_hedges`` is on, but still face the robustness floors. Rejections are
recorded with reasons so the UI can explain what got dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import StrategyCandidate

__all__ = ["Constraints", "PortfolioSelector", "pairwise_corr"]


@dataclass
class Constraints:
    min_sharpe: float = 0.0
    min_cagr: float = 0.0
    min_dsr: float = 0.0
    min_oos_sharpe: float = 0.0
    min_wf_eff: float = 0.0
    max_pbo: float = 1.0
    require_verdict: str = "ANY"   # ANY | PROMOTE | CONDITIONAL
    min_n_obs: int = 30
    corr_max: float = 1.0
    allow_hedges: bool = True


def pairwise_corr(candidates: list[StrategyCandidate]) -> tuple[list[str], np.ndarray]:
    """Pairwise Pearson correlation aligned on the date intersection (min 30 obs)."""
    names = [c.name for c in candidates]
    if not candidates:
        return names, np.zeros((0, 0))
    frame = pd.concat({c.name: c.returns for c in candidates}, axis=1)
    # .copy() so the array is writable — under pandas Copy-on-Write (default on
    # newer pandas), to_numpy() can return a read-only view, and fill_diagonal
    # writes in place.
    corr = frame.corr(min_periods=30).reindex(index=names, columns=names)
    mat = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(mat, 1.0)
    return names, mat


class PortfolioSelector:
    def __init__(self, candidates: list[StrategyCandidate], constraints: Constraints):
        self.candidates = candidates
        self.constraints = constraints
        self.rejections: list[dict] = []

    def filter_by_robustness(self) -> list[StrategyCandidate]:
        c = self.constraints
        kept = []
        for cand in self.candidates:
            reasons = []
            bypass_pnl = c.allow_hedges and cand.is_hedge
            # ``not (x >= floor)`` rather than ``x < floor`` so a non-finite metric
            # (e.g. a degenerate zero-variance strategy with NaN Sharpe) FAILS the
            # floor instead of silently passing it.
            if not bypass_pnl:
                if not (cand.sharpe >= c.min_sharpe):
                    reasons.append(f"sharpe {cand.sharpe:.3f} < {c.min_sharpe}")
                if not (cand.cagr >= c.min_cagr):
                    reasons.append(f"cagr {cand.cagr:.3f} < {c.min_cagr}")
                if not (cand.dsr >= c.min_dsr):
                    reasons.append(f"dsr {cand.dsr:.3f} < {c.min_dsr}")
                if not (cand.oos_sharpe >= c.min_oos_sharpe):
                    reasons.append(f"oos_sharpe {cand.oos_sharpe:.3f} < {c.min_oos_sharpe}")
            pbo = cand.pbo
            if pbo == pbo and pbo > c.max_pbo:  # only enforce when PBO known (not NaN)
                reasons.append(f"pbo {pbo:.3f} > {c.max_pbo}")
            if not (cand.wf_efficiency >= c.min_wf_eff):
                reasons.append(f"wf_eff {cand.wf_efficiency:.3f} < {c.min_wf_eff}")
            if c.require_verdict != "ANY" and cand.verdict != c.require_verdict:
                reasons.append(f"verdict {cand.verdict!r} != {c.require_verdict!r}")
            if len(cand.returns) < c.min_n_obs:
                reasons.append(f"n_obs {len(cand.returns)} < {c.min_n_obs}")
            if reasons:
                self.rejections.append({"name": cand.name, "reasons": reasons})
            else:
                kept.append(cand)
        return kept

    def filter_by_diversification(
        self, candidates: list[StrategyCandidate],
    ) -> list[StrategyCandidate]:
        corr_max = self.constraints.corr_max
        if corr_max >= 1.0 or len(candidates) < 2:
            return candidates
        pool = list(candidates)
        _, mat = pairwise_corr(pool)
        abs_mat = np.abs(mat)
        np.fill_diagonal(abs_mat, 0.0)
        abs_mat = np.where(np.isnan(abs_mat), 0.0, abs_mat)
        active = np.ones(len(pool), dtype=bool)
        while True:
            i, j = divmod(int(np.argmax(abs_mat)), abs_mat.shape[1])
            worst = float(abs_mat[i, j])
            if worst <= corr_max:
                break
            drop = i if pool[i].oos_sharpe < pool[j].oos_sharpe else j
            kept = j if drop == i else i
            self.rejections.append({"name": pool[drop].name,
                                    "reasons": [f"corr |{worst:.2f}| > {corr_max} "
                                                f"with {pool[kept].name}"]})
            active[drop] = False
            abs_mat[drop, :] = 0.0
            abs_mat[:, drop] = 0.0
        return [c for c, a in zip(pool, active, strict=True) if a]

    def select(self) -> list[StrategyCandidate]:
        return self.filter_by_diversification(self.filter_by_robustness())
