"""Strategy discovery — turn the Rigor strategy book into portfolio candidates.

A ``StrategyCandidate`` is one strategy's loaded state: its daily returns plus
whatever validation metadata is on disk. ``scan_strategies`` walks the Rigor
``strategies/`` tree and builds one per strategy, reading the Rigor artifact contract:

  strategies/<category>/<slug>/artifacts/<slug>_returns.csv   (date,returns)
                                       <slug>_summary.json   (metrics + verdict)
                                       <slug>_optimization.json  (optional: CPCV)

All performance metrics derive from the returns via ``rigor.metrics`` — the same
definitions the rest of the framework uses, so a Sharpe here equals a Sharpe in
the catalog. The ``cpcv``/``wf`` dicts are best-effort: full PBO/OOS-Sharpe come
from a strategy's optimisation report when it has one, otherwise the verdict and
the realised metrics stand in.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rigor import metrics as _m

__all__ = ["StrategyCandidate", "scan_strategies", "read_returns_csv"]


def read_returns_csv(path: Path) -> pd.Series | None:
    """Parse an Rigor ``<slug>_returns.csv`` (``date,returns``) into a Series.
    Returns ``None`` if unparseable or under 30 usable observations."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or len(df.columns) < 2:
        return None
    idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=idx).dropna()
    s = s[np.isfinite(s)]
    return s if len(s) >= 30 else None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


@dataclass
class StrategyCandidate:
    """One strategy available to a portfolio: its returns + validation metadata."""
    name: str
    directory: Path
    returns: pd.Series
    slug: str = ""
    category: str = ""
    cpcv: dict = field(default_factory=dict)
    wf: dict = field(default_factory=dict)
    verdict_grade: str = "UNKNOWN"
    sector_exposure: dict = field(default_factory=lambda: {"Unknown": 1.0})

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return f if math.isfinite(f) else default

    @property
    def ppy(self) -> int:
        return _m.periods_per_year_of(self.returns.index)

    @property
    def sharpe(self) -> float:
        return float(_m.compute_sharpe(self.returns.to_numpy(dtype="float64"), self.ppy))

    @property
    def cagr(self) -> float:
        return float(_m.compute_cagr(self.returns.to_numpy(dtype="float64"), self.ppy))

    @property
    def max_drawdown(self) -> float:
        return float(_m.compute_max_drawdown(self.returns.to_numpy(dtype="float64")))

    @property
    def verdict(self) -> str:
        """Promotion verdict (PROMOTE / CONDITIONAL / REJECT / UNKNOWN)."""
        return str(self.cpcv.get("verdict") or self.verdict_grade or "UNKNOWN")

    @property
    def pbo(self) -> float:
        """CPCV probability of overfit; NaN when the strategy has no grid/CPCV."""
        return self._safe_float(self.cpcv.get("pbo"), float("nan"))

    @property
    def oos_sharpe(self) -> float:
        """Out-of-sample Sharpe: CPCV mean-OOS if available, else walk-forward
        OOS, else the full-period Sharpe."""
        for src, key in ((self.cpcv, "mean_oos_sharpe"), (self.wf, "oos_sharpe_mean")):
            v = self._safe_float(src.get(key), float("nan"))
            if math.isfinite(v):
                return v
        return self.sharpe

    @property
    def dsr(self) -> float:
        """Deflated Sharpe Ratio — P(true SR > 0), skew/kurtosis-aware (Bailey-LdP)."""
        from rigor.validation.overfit import deflated_sharpe_ratio
        from scipy.stats import kurtosis, skew
        arr = self.returns.to_numpy(dtype="float64")
        n = len(arr)
        if n < 4:
            return 0.0
        sk = float(skew(arr, bias=False))
        ku = float(kurtosis(arr, fisher=True, bias=False))
        sharpe_pp = self.sharpe / math.sqrt(self.ppy)
        return float(deflated_sharpe_ratio(sharpe_pp, 1, n, sk, ku))

    @property
    def wf_efficiency(self) -> float:
        """Walk-forward OOS/IS Sharpe efficiency (0 when no walk-forward data)."""
        eff = self._safe_float(self.wf.get("wf_efficiency"), float("nan"))
        if math.isfinite(eff):
            return eff
        is_sr = self._safe_float(self.wf.get("is_sharpe_mean"), float("nan"))
        oos_sr = self._safe_float(self.wf.get("oos_sharpe_mean"), float("nan"))
        if math.isfinite(is_sr) and is_sr != 0 and math.isfinite(oos_sr):
            return oos_sr / is_sr
        return 0.0

    @property
    def internal_leverage(self) -> float:
        """Gross exposure implied by the strategy's sizing. Rigor strategies are
        1.0 (no leverage metadata); kept for the portfolio leverage cap."""
        return 1.0

    @property
    def is_hedge(self) -> bool:
        """Structured as a hedge (low/neg standalone Sharpe by design) rather than
        a PnL driver — detected by name + a low standalone Sharpe."""
        up = self.name.upper()
        if "HEDGE" in up:
            return True
        return ("SHORT" in up) and self.sharpe < 0.3


def scan_strategies(
    strategy_root: str | Path, *, include_baseline: bool = False, min_obs: int = 30,
) -> list[StrategyCandidate]:
    """Discover every Rigor strategy that can feed a portfolio.

    Walks ``strategy_root`` for ``<category>/<slug>/artifacts/<slug>_returns.csv``.
    Baselines (``strategies/Baseline/...``) are excluded unless ``include_baseline``.
    Deduped by strategy directory; sorted by slug for determinism.
    """
    root = Path(strategy_root)
    out: dict[Path, StrategyCandidate] = {}
    for returns_csv in sorted(root.rglob("artifacts/*_returns.csv")):
        strat_dir = returns_csv.parent.parent
        if strat_dir in out:
            continue
        rel = strat_dir.relative_to(root).parts
        is_baseline = rel and rel[0] == "Baseline"
        if is_baseline and not include_baseline:
            continue
        slug = returns_csv.name[: -len("_returns.csv")]
        rets = read_returns_csv(returns_csv)
        if rets is None or len(rets) < min_obs:
            continue
        artifacts = returns_csv.parent
        summary = _load_json(artifacts / f"{slug}_summary.json")
        opt = _load_json(artifacts / f"{slug}_optimization.json")
        category = rel[1] if is_baseline and len(rel) > 1 else (rel[0] if rel else "")

        cpcv = dict(opt.get("cpcv") or {})
        cpcv.setdefault("verdict", (summary.get("verdict") or {}).get("verdict", "UNKNOWN"))
        wf = dict(opt.get("walk_forward_opt") or {})
        out[strat_dir] = StrategyCandidate(
            name=summary.get("name") or slug,
            directory=strat_dir,
            returns=rets,
            slug=slug,
            category=category,
            cpcv=cpcv,
            wf=wf,
            verdict_grade=(summary.get("verdict") or {}).get("grade", "UNKNOWN"),
        )
    return [out[k] for k in sorted(out, key=lambda d: out[d].slug)]
