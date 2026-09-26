"""Deterministic split/dividend back-adjustment, computed in-framework.

Why not just use EODHD's ``adjusted_close``? Because vendor-adjusted series
*restate* whenever a new corporate action arrives -- so the same code pulling on
two different days would see different history. By storing raw OHLCV + the raw
action feed and applying the adjustment *here*, the result is a pure function of
frozen inputs: same raw bytes -> same adjusted series, forever, on any machine.

Convention: prices are *back-adjusted* and anchored so the most recent bar equals
its raw value (factor = 1 at the latest date); history is scaled down. This
matches Norgate's back-adjusted convention and keeps recent prices realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Method = str  # "total_return" | "split" | "none"


def parse_splits(raw_splits: list[dict]) -> dict[str, float]:
    """EODHD split records -> {ex_date: ratio} where ratio = new/old shares.

    e.g. a 4-for-1 split is recorded "4.000000/1.000000" -> 4.0.
    """
    out: dict[str, float] = {}
    for rec in raw_splits:
        d = rec.get("date")
        s = rec.get("split")
        if not d or not s:
            continue
        try:
            num, den = (float(x) for x in str(s).split("/"))
            if den:
                out[d] = num / den
        except (ValueError, ZeroDivisionError):
            continue
    return out


def parse_dividends(raw_divs: list[dict]) -> dict[str, float]:
    """EODHD dividend records -> {ex_date: cash_amount} in historical (raw) units.

    Uses ``unadjustedValue`` (the actual cash paid at the time) so it is
    consistent with the raw close on the day before the ex-date.
    """
    out: dict[str, float] = {}
    for rec in raw_divs:
        d = rec.get("date")
        if not d:
            continue
        val = rec.get("unadjustedValue", rec.get("value"))
        if val is None:
            continue
        try:
            amt = float(val)
        except (TypeError, ValueError):
            continue
        if amt:
            out[d] = out.get(d, 0.0) + amt
    return out


def adjustment_factors(
    dates: list[str],
    close: np.ndarray,
    splits: dict[str, float],
    dividends: dict[str, float],
    include_dividends: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (price_factor, split_only_factor), each anchored to 1 at the last bar.

    ``price_factor`` multiplies O/H/L/C; ``split_only_factor`` is used to adjust
    volume (volume is unaffected by dividends).
    """
    n = len(dates)
    price_factor = np.ones(n)
    split_factor = np.ones(n)
    if n == 0:
        return price_factor, split_factor

    # Walk backward: factor[i] = factor[i+1] * (effect of an action dated i+1).
    for i in range(n - 2, -1, -1):
        d_next = dates[i + 1]
        sp = 1.0
        if d_next in splits and splits[d_next] > 0:
            sp = 1.0 / splits[d_next]
        div = 1.0
        if include_dividends and d_next in dividends:
            prev_close = close[i]
            if prev_close > 0:
                div = max(0.0, 1.0 - dividends[d_next] / prev_close)
        split_factor[i] = split_factor[i + 1] * sp
        price_factor[i] = price_factor[i + 1] * sp * div
    return price_factor, split_factor


def adjust_ohlcv(
    df: pd.DataFrame,
    raw_splits: list[dict],
    raw_dividends: list[dict],
    method: Method = "total_return",
) -> pd.DataFrame:
    """Return a copy of a raw OHLCV frame with adjusted columns added.

    Input ``df`` must have columns: date, open, high, low, close, volume
    (date ascending). Adds: adj_open, adj_high, adj_low, adj_close, adj_volume.
    """
    out = df.copy().reset_index(drop=True)
    if out.empty:
        for c in ("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"):
            out[c] = pd.Series(dtype="float64")
        return out

    if method == "none":
        price_factor = np.ones(len(out))
        split_factor = np.ones(len(out))
    else:
        splits = parse_splits(raw_splits)
        dividends = parse_dividends(raw_dividends)
        price_factor, split_factor = adjustment_factors(
            out["date"].tolist(),
            out["close"].to_numpy(dtype="float64"),
            splits,
            dividends,
            include_dividends=(method == "total_return"),
        )

    for col in ("open", "high", "low", "close"):
        out[f"adj_{col}"] = out[col].to_numpy(dtype="float64") * price_factor
    # Volume scales inversely to the split-only price factor (not dividends).
    with np.errstate(divide="ignore", invalid="ignore"):
        out["adj_volume"] = np.where(
            split_factor > 0, out["volume"].to_numpy(dtype="float64") / split_factor, np.nan
        )
    return out
