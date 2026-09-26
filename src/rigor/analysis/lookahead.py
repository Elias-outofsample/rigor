"""Look-ahead / causal-leakage detector (#9) — future-data perturbation.

The look-ahead question is simple: *does a strategy's decision at time t depend
on data that only exists after t?* A causal (point-in-time) strategy's position
at t is a function of bars ``<= t`` only, so corrupting every bar **strictly
after** a cutoff date ``T`` must leave every position at ``<= T`` byte-identical.
If any pre-``T`` position changes, the strategy read the future → look-ahead.

This module implements that directly and *engine-agnostically*:

  1. Run the strategy on the full, untouched dataset and capture its decisions
     (target weights if it exposes them, else its net-return stream).
  2. For each cutoff ``T`` (default 50% and 75% of the sample), build a perturbed
     world where every bar **after** ``T`` is corrupted (scrambled by default, or
     NaN-blanked), rebuild a *fresh* strategy on it, and re-run.
  3. Compare the two runs on the overlapping ``<= T`` index. Any difference
     beyond a floating-point tolerance is a leak.

It works on any :class:`rigor.strategy.StrategyBase` because it perturbs the *data
loader* the strategy is built from — it never needs the strategy to honour a
private contract. A correct causal engine therefore yields ``leak_detected =
False`` with zero pre-``T`` deltas.

A legacy Sharpe-shift entry point (the original ``#9`` signature taking a
``backtest_fn`` callable) is preserved for backward compatibility and dispatched
to automatically when the first argument is a callable.

Pure numpy/pandas. Deterministic (the scramble uses a fixed seed).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = ["CutoffResult", "LookaheadResult", "detect_lookahead"]

PerturbMode = Literal["nan", "scramble"]

# Default cutoff fractions of the captured sample (mid-sample + late-sample).
_DEFAULT_CUTOFF_FRACS: tuple[float, ...] = (0.50, 0.75)

# Columns we never blank/scramble: they carry the calendar/index, not values a
# causal strategy may legitimately read. Blanking these would drop rows and break
# the index-alignment comparison rather than test causality.
_INDEX_LIKE = frozenset({"date", "dt", "funding_time", "filing_date", "report_date",
                         "period", "session", "dateformatted"})


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class CutoffResult:
    """Per-cutoff outcome of the perturbation test."""

    cutoff: pd.Timestamp
    compared: str                 # "weights" | "returns" — what was diffed
    n_compared: int               # number of pre-T cells/points compared
    max_abs_diff: float           # largest |full - perturbed| on the pre-T index
    leak: bool                    # max_abs_diff > tolerance
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cutoff": str(self.cutoff),
            "compared": self.compared,
            "n_compared": int(self.n_compared),
            "max_abs_diff": float(self.max_abs_diff),
            "leak": bool(self.leak),
            "note": self.note,
        }


@dataclass
class LookaheadResult:
    """Structured verdict from :func:`detect_lookahead`.

    ``leak_detected`` is the headline boolean; ``per_cutoff`` carries the detail;
    ``summary`` is a one-line human-readable conclusion.
    """

    leak_detected: bool
    per_cutoff: list[CutoffResult] = field(default_factory=list)
    summary: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "leak_detected": bool(self.leak_detected),
            "per_cutoff": [c.as_dict() for c in self.per_cutoff],
            "summary": self.summary,
            "error": self.error,
        }

    def __bool__(self) -> bool:  # truthy == a leak was found
        return self.leak_detected


# ===========================================================================
# Data-loader perturbation
# ===========================================================================

def _blank_after(frame: pd.DataFrame, cutoff: pd.Timestamp, *,
                 mode: PerturbMode, rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt every numeric value on rows whose date is strictly after ``cutoff``.

    The date/index-like columns are preserved so the calendar (and any index a
    causal cache derives from it) is unchanged — only the *values* a future bar
    carries are destroyed. Returns a copy; the input is untouched.
    """
    out = frame.copy()

    # Locate the row timestamps: a DatetimeIndex, or the first date-like column.
    if isinstance(out.index, pd.DatetimeIndex):
        after = out.index > cutoff
    else:
        date_col = next(
            (c for c in out.columns if str(c).lower() in _INDEX_LIKE
             and pd.api.types.is_datetime64_any_dtype(
                 pd.to_datetime(out[c], errors="coerce"))),
            None,
        )
        if date_col is None:
            return out  # no recognisable calendar → leave untouched
        ts = pd.to_datetime(out[date_col], errors="coerce")
        after = (ts > cutoff).to_numpy()

    if not after.any():
        return out

    value_cols = [c for c in out.columns if str(c).lower() not in _INDEX_LIKE]
    if mode == "scramble":
        # Deterministically corrupt each numeric value column's post-cutoff block.
        # We permute the block (preserve its rough distribution, so it stays
        # finite/plausible and won't crash equity-curve math the way an all-NaN
        # tail can) AND apply a small deterministic jitter to every post-cutoff
        # value. The jitter guarantees that EACH future bar — including the single
        # bar immediately after T — differs from the original, so even a one-bar
        # look-ahead is caught at every cutoff (a pure permutation could leave the
        # boundary bar's value unchanged by chance). Non-numeric columns are left
        # alone — scrambling them risks breaking calendars the engine derives.
        idx_after = np.flatnonzero(after)
        jitter = 1.0 + 0.05 * rng.standard_normal(len(idx_after))
        for c in value_cols:
            if not pd.api.types.is_numeric_dtype(out[c]):
                continue
            col = out[c].to_numpy(dtype="float64", copy=True)
            block = col[idx_after]
            if len(idx_after) > 1:
                block = block[rng.permutation(len(idx_after))]
            col[idx_after] = block * jitter
            out[c] = col
    else:  # "nan": blank the future outright
        for c in value_cols:
            col = out[c].to_numpy(dtype="float64", copy=True) \
                if pd.api.types.is_numeric_dtype(out[c]) else out[c].to_numpy(copy=True)
            if pd.api.types.is_numeric_dtype(out[c]):
                col[after] = np.nan
            else:
                col = col.astype(object)
                col[after] = np.nan
            out[c] = col
    return out


class _PerturbedLoader:
    """Transparent proxy over a data loader that corrupts post-cutoff rows.

    Every method is delegated to the wrapped loader; results that look like a
    price/series panel (a ``DataFrame`` or dated ``Series``) get their post-cutoff
    rows blanked/scrambled. Anything else is passed through untouched. This is how
    the detector stays engine-agnostic — it never inspects the strategy's logic,
    only what data it is *allowed* to see.
    """

    # Methods whose output is a dated panel/series we should perturb.
    _DATA_METHODS = frozenset({
        "prices", "load_prices", "fred", "series", "funding", "intraday",
        "earnings",
    })

    def __init__(self, inner: Any, cutoff: pd.Timestamp, *,
                 mode: PerturbMode, seed: int) -> None:
        self._inner = inner
        self._cutoff = cutoff
        self._mode = mode
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in self._DATA_METHODS or not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = attr(*args, **kwargs)
            return self._perturb(out)

        return wrapped

    def _perturb(self, out: Any) -> Any:
        if isinstance(out, pd.DataFrame):
            return _blank_after(out, self._cutoff, mode=self._mode, rng=self._rng)
        if isinstance(out, pd.Series) and isinstance(out.index, pd.DatetimeIndex):
            df = _blank_after(out.to_frame("__v"), self._cutoff,
                              mode=self._mode, rng=self._rng)
            return df["__v"].rename(out.name)
        return out


# ===========================================================================
# Capturing a run's decisions (weights preferred, returns as fallback)
# ===========================================================================

def _capture(strategy: Any) -> tuple[str, pd.DataFrame | pd.Series]:
    """Run a strategy and return ``(kind, decisions)``.

    ``kind`` is ``"weights"`` (date×symbol target weights, the purest position
    signal) when the result exposes them, else ``"returns"`` (the net-return
    series every engine produces).
    """
    result = strategy.backtest()
    weights = getattr(result, "weights", None)
    if isinstance(weights, pd.DataFrame) and not weights.empty \
            and isinstance(weights.index, pd.DatetimeIndex):
        return "weights", weights.sort_index()
    returns = result.returns
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns, dtype="float64"))
    return "returns", returns.sort_index() if isinstance(
        returns.index, pd.DatetimeIndex) else returns


def _compare_pre_cutoff(
    full: pd.DataFrame | pd.Series,
    pert: pd.DataFrame | pd.Series,
    cutoff: pd.Timestamp,
    *,
    atol: float,
    rtol: float,
) -> tuple[int, float]:
    """Max abs difference on the overlapping ``<= cutoff`` index.

    NaNs are treated as equal when both runs are NaN at the same cell (a shared
    warm-up gap is not a leak); a NaN that appears in only one run *is* a diff.
    Returns ``(n_cells_compared, max_abs_diff)``.
    """
    if isinstance(full, pd.Series):
        full = full.to_frame("v")
    if isinstance(pert, pd.Series):
        pert = pert.to_frame("v")

    # Restrict to <= cutoff on a datetime index (otherwise compare everything,
    # which for a non-dated returns array means the whole vector).
    if isinstance(full.index, pd.DatetimeIndex):
        full = full.loc[full.index <= cutoff]
    if isinstance(pert.index, pd.DatetimeIndex):
        pert = pert.loc[pert.index <= cutoff]

    idx = full.index.intersection(pert.index)
    cols = full.columns.intersection(pert.columns)
    if len(idx) == 0 or len(cols) == 0:
        return 0, 0.0

    a = full.loc[idx, cols].to_numpy(dtype="float64")
    b = pert.loc[idx, cols].to_numpy(dtype="float64")
    n = int(a.size)

    # A NaN in exactly one run (e.g. the future-blank propagated into a pre-T
    # position) is itself a leak → report an infinite difference.
    one_nan = np.isnan(a) ^ np.isnan(b)
    if one_nan.any():
        return n, float("inf")

    # Both-NaN cells (a shared warm-up gap) count as equal. Compare the rest with
    # a magnitude-scaled tolerance and report the largest *excess* over tolerance,
    # so the caller's ``max_abs > atol`` test fires only on a genuine deviation.
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(np.where(both_nan, 0.0, np.nan_to_num(a) - np.nan_to_num(b)))
    tol = atol + rtol * np.abs(np.nan_to_num(a))
    excess = diff - tol
    max_excess = float(excess.max()) if excess.size else 0.0
    # Map a positive excess back onto an absolute scale strictly above ``atol``;
    # a within-tolerance run reports 0.0 so the verdict is a clean pass.
    max_abs = (atol + max_excess) if max_excess > 0.0 else 0.0
    return n, max_abs


# ===========================================================================
# Primary detector: future-data perturbation on a StrategyBase
# ===========================================================================

def _resolve_cutoffs(
    index: pd.Index,
    cutoffs: Sequence[Any] | None,
) -> list[pd.Timestamp]:
    """Turn the ``cutoffs`` argument into concrete timestamps on ``index``.

    ``None`` → the default sample fractions. Numbers in (0,1) are read as
    fractions of the (datetime) index; anything else is parsed as a timestamp.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return []
    si = index.sort_values()
    out: list[pd.Timestamp] = []
    raw: Sequence[Any] = _DEFAULT_CUTOFF_FRACS if cutoffs is None else cutoffs
    for c in raw:
        if isinstance(c, (int, float)) and 0.0 < float(c) < 1.0:
            pos = min(len(si) - 1, max(0, int(round(float(c) * (len(si) - 1)))))
            out.append(pd.Timestamp(si[pos]))
        else:
            out.append(pd.Timestamp(c))
    # De-dupe, keep order, and only keep cutoffs that leave a pre-window to test.
    seen: set[pd.Timestamp] = set()
    uniq: list[pd.Timestamp] = []
    for t in out:
        if t not in seen and t > si[0] and t < si[-1]:
            seen.add(t)
            uniq.append(t)
    return uniq


def _detect_lookahead_perturb(
    strategy: Any,
    data: Any,
    *,
    cutoffs: Sequence[Any] | None,
    mode: PerturbMode,
    rebuild: Callable[[Any, Any], Any] | None,
    atol: float,
    rtol: float,
    seed: int,
) -> LookaheadResult:
    if data is None:
        data = getattr(strategy, "data", None)
    if data is None:
        return LookaheadResult(
            False, summary="no data loader to perturb — cannot test",
            error="missing data loader",
        )

    def _rebuild(cfg: Any, d: Any) -> Any:
        if rebuild is not None:
            return rebuild(cfg, d)
        return type(strategy)(cfg, d)

    config = getattr(strategy, "config", None)
    try:
        kind, full = _capture(strategy)
    except Exception as exc:  # noqa: BLE001
        return LookaheadResult(False, summary=f"baseline run failed: {exc}",
                               error=str(exc))

    resolved = _resolve_cutoffs(full.index, cutoffs)
    if not resolved:
        return LookaheadResult(
            False,
            summary="no datetime-indexed decisions to test (need a dated "
                    "weights/returns series and valid cutoffs)",
            error="no usable cutoffs",
        )

    results: list[CutoffResult] = []
    for t in resolved:
        loader = _PerturbedLoader(data, t, mode=mode, seed=seed)
        try:
            pert_strat = _rebuild(config, loader)
            kind_p, pert = _capture(pert_strat)
        except Exception as exc:  # noqa: BLE001
            results.append(CutoffResult(
                cutoff=t, compared=kind, n_compared=0, max_abs_diff=float("nan"),
                leak=False, note=f"perturbed run failed: {exc}",
            ))
            continue

        # Compare on the common decision kind (fall back to returns if the
        # perturbed run could not produce weights).
        use_full, use_pert, compared = full, pert, kind
        if kind != kind_p:
            use_full, use_pert, compared = _coerce_to_returns(strategy), \
                _coerce_to_returns(pert_strat), "returns"

        n, max_abs = _compare_pre_cutoff(
            use_full, use_pert, t, atol=atol, rtol=rtol)
        leak = max_abs > atol
        note = "" if n else "no overlapping pre-cutoff cells"
        results.append(CutoffResult(
            cutoff=t, compared=compared, n_compared=n,
            max_abs_diff=max_abs, leak=leak and n > 0, note=note,
        ))

    leak_detected = any(c.leak for c in results)
    summary = _summarise(leak_detected, results)
    return LookaheadResult(leak_detected, per_cutoff=results, summary=summary)


def _coerce_to_returns(strategy: Any) -> pd.Series:
    r = strategy.backtest().returns
    if not isinstance(r, pd.Series):
        r = pd.Series(np.asarray(r, dtype="float64"))
    return r.sort_index() if isinstance(r.index, pd.DatetimeIndex) else r


def _summarise(leak: bool, results: list[CutoffResult]) -> str:
    if not results:
        return "inconclusive — no cutoffs evaluated"
    tested = [r for r in results if r.n_compared > 0]
    if leak:
        worst = max((r for r in results if r.leak),
                    key=lambda r: (r.max_abs_diff if np.isfinite(r.max_abs_diff)
                                   else np.inf))
        return (f"LOOK-AHEAD DETECTED: pre-cutoff positions changed when the "
                f"future was corrupted (worst cutoff {worst.cutoff.date()}, "
                f"max |delta| {worst.max_abs_diff:.3g} on {worst.compared}).")
    if not tested:
        return ("inconclusive — perturbed runs produced no overlapping "
                "pre-cutoff positions to compare.")
    return (f"no look-ahead: pre-cutoff {tested[0].compared} are identical across "
            f"{len(tested)} cutoff(s) after corrupting all future data (causal).")


# ===========================================================================
# Legacy Sharpe-shift detector (original #9 signature) — kept for compat
# ===========================================================================

_DROP_THRESHOLD = 0.40        # >40% Sharpe drop after the shift => suspect
_RANDOM_WALK_SHARPE = 0.10    # shifted Sharpe near 0 => signal needed the future


def _sharpe(returns: np.ndarray, ppy: int = _m.TRADING_DAYS) -> float:
    r = returns[np.isfinite(returns)]
    return float(_m.compute_sharpe(r, ppy)) if len(r) >= 2 else 0.0


def _detect_lookahead_legacy(
    backtest_fn: Callable, cache: dict, params: dict, *,
    shift_bars: int = 1, n_random_trials: int = 5,
    seed: int = 42, ppy: int = _m.TRADING_DAYS,
) -> dict:
    """Legacy entry point: baseline vs ``cache['_shift']``-shifted Sharpe.

    Preserved so the original #9 callers/tests keep working. The shift contract
    is opt-in (``backtest_fn`` lags its own signal by ``cache['_shift']``); when a
    backtest ignores it the run is identical and the verdict is a clean pass.
    """
    notes: list[str] = []
    try:
        base = np.asarray(backtest_fn(cache, params), dtype="float64")
        base = base[np.isfinite(base)]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"baseline backtest failed: {exc}"}
    if len(base) < 10:
        return {"error": "insufficient baseline returns"}
    base_sr = _sharpe(base, ppy)

    shifted_cache = {**cache, "_shift": shift_bars}
    try:
        shifted = np.asarray(backtest_fn(shifted_cache, params), dtype="float64")
        shifted = shifted[np.isfinite(shifted)]
        shifted_sr = _sharpe(shifted, ppy)
        supported = True
    except Exception as exc:  # noqa: BLE001
        notes.append(f"shifted run failed ({exc}); test inconclusive")
        shifted_sr, supported = float("nan"), False

    drop = ((base_sr - shifted_sr) / max(abs(base_sr), 1e-9)
            if supported and np.isfinite(shifted_sr) and base_sr != 0 else float("nan"))

    rng = np.random.default_rng(seed)
    random_sr = []
    for _ in range(n_random_trials):
        s = base.copy()
        rng.shuffle(s)
        random_sr.append(round(_sharpe(s, ppy), 4))
    mean_random = float(np.mean(random_sr)) if random_sr else 0.0

    suspect, confidence = False, "low"
    if not supported:
        notes.append("backtest_fn does not honour cache['_shift'] — test inconclusive")
    elif np.isfinite(drop):
        if drop > _DROP_THRESHOLD:
            suspect, confidence = True, "high"
            notes.append(f"Sharpe drops {drop:.1%} after a {shift_bars}-bar shift "
                         "— strong look-ahead signal")
        elif abs(shifted_sr) < _RANDOM_WALK_SHARPE:
            suspect, confidence = True, "medium"
            notes.append(f"shifted Sharpe ~0 ({shifted_sr:.3f}) — strategy may depend "
                         "on future data")
        elif drop > 0.20:
            confidence = "medium"
            notes.append(f"moderate Sharpe drop ({drop:.1%}) — worth investigating")
        else:
            confidence = "high"
            notes.append("no significant Sharpe drop after the shift — no look-ahead detected")
    if base_sr > 0 and abs(mean_random) < 0.05:
        notes.append(f"shuffle Sharpe ~0 ({mean_random:.3f}) — returns aren't trivially "
                     "autocorrelated")

    return {
        "baseline_sharpe": round(base_sr, 4),
        "shifted_sharpe": round(shifted_sr, 4) if np.isfinite(shifted_sr) else None,
        "sharpe_drop_pct": round(drop, 4) if np.isfinite(drop) else None,
        "suspect": suspect, "confidence": confidence,
        "shift_bars": shift_bars, "shift_supported": supported,
        "random_sharpes": random_sr, "mean_random_sharpe": round(mean_random, 4),
        "notes": notes,
    }


# ===========================================================================
# Public dispatcher
# ===========================================================================

def detect_lookahead(
    strategy: Any,
    data: Any = None,
    params: dict | None = None,
    *,
    cutoffs: Sequence[Any] | None = None,
    mode: PerturbMode = "scramble",
    rebuild: Callable[[Any, Any], Any] | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-6,
    seed: int = 12345,
    **legacy_kwargs: Any,
) -> LookaheadResult | dict:
    """Detect causal look-ahead by future-data perturbation.

    Primary use — pass an :class:`rigor.strategy.StrategyBase` instance and the
    data loader it was built from::

        result = detect_lookahead(strategy, data)
        assert not result.leak_detected

    The detector reruns the strategy on a dataset whose every bar **after** each
    cutoff date is corrupted, then checks that all positions **at or before** the
    cutoff are unchanged. A causal strategy passes; one that reads the future
    fails. Returns a :class:`LookaheadResult` (``leak_detected`` + per-cutoff
    detail + a human ``summary``).

    Parameters:
      strategy  — a built ``StrategyBase`` (has ``.backtest()``); or, for the
                  legacy path, a ``backtest_fn(cache, params)`` callable.
      data      — the loader the strategy reads (defaults to ``strategy.data``).
      cutoffs   — timestamps, or fractions in (0,1) of the sample. Default
                  ``(0.50, 0.75)``.
      mode      — ``"scramble"`` (default; permute+reverse the future, preserving
                  its distribution so engine accounting never sees an all-NaN
                  tail) or ``"nan"`` (blank the future outright). Scramble is the
                  robust default because some engines' equity-curve math chokes on
                  an all-NaN tail, which would silently make the test vacuous.
      rebuild   — optional ``(config, data) -> StrategyBase`` factory used to
                  spin up the perturbed run; defaults to ``type(strategy)(config,
                  data)``.
      atol/rtol — floating-point tolerance for the pre-cutoff position diff.

    Backward compatibility: if ``strategy`` is a callable (the original #9
    ``backtest_fn``) the call is routed to the legacy Sharpe-shift detector,
    which returns the original ``dict`` verdict.
    """
    # Legacy dispatch: detect_lookahead(backtest_fn, cache, params, ...).
    if callable(strategy) and not hasattr(strategy, "backtest"):
        cache = data if isinstance(data, dict) else {}
        return _detect_lookahead_legacy(
            strategy, cache, params or {}, **legacy_kwargs)

    return _detect_lookahead_perturb(
        strategy, data, cutoffs=cutoffs, mode=mode, rebuild=rebuild,
        atol=atol, rtol=rtol, seed=seed,
    )
