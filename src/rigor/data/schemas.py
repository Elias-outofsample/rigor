"""Data-contract validation at the data boundary.

A backtest is only as trustworthy as the bars it runs on. Vendor feeds
occasionally ship corrupt rows — a ``high`` below the ``low``, a negative close,
a duplicated or out-of-order date — and a silent bad bar can quietly poison
every downstream metric. This module enforces a small, explicit *contract* on an
OHLCV frame the moment it crosses into the framework, so that genuinely corrupt
data fails loudly (with a message naming the ticker, date, and violated rule)
instead of propagating into the engine.

Why pure pandas/numpy (not Pandera)
-----------------------------------
The contract is six concrete rules; a dependency-light validator expresses them
in ~40 lines, keeps the core install reproducible and minimal
(the repo deliberately confines heavy libraries to optional extras), and —
crucially — lets us raise a *precise, actionable* error (``OHLC violation: SPY
2020-03-16 high<low``) rather than a generic schema dump. The deliverable is the
enforced contract and clear errors, not any particular library.

The OHLCV contract
------------------
For a tidy, date-ascending frame with columns
``date, open, high, low, close[, volume]``:

1. **Non-negative prices** — open/high/low/close are ``>= 0`` wherever present.
   NaN *and* exact 0 are tolerated: both are real vendor artifacts on thin /
   halted / delisted names (a no-trade session shows a 0 low or a NaN bar) that
   the loader/engine neutralise downstream. Only a *negative* price is
   structurally impossible, and that is the corruption this rule catches.
2. **OHLC consistency** — ``low <= min(open, close)`` and
   ``high >= max(open, close)`` (hence ``high >= low``) on every fully-populated
   bar, skipping any bar with a NaN leg.
3. **Clean dates** — dates are unique and strictly monotonic-increasing.

Volume is intentionally unconstrained, and a NaN/0 close is tolerated (both are
legitimate on thin or delisted names). The contract deliberately flags only the
*structurally impossible* — a negative price, a high below the low, or a
duplicate/out-of-order date — never a merely-imperfect bar the engine already
handles. Use :func:`validate_ohlcv` at a once-per-ticker seam (fetch /
cache-write), never on a hot read path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["DataContractError", "validate_ohlcv", "validate_returns"]

# Price columns that must be positive where present. ``volume`` is deliberately
# excluded (0/NaN volume is legitimate); ``adjusted_close`` is a vendor extra we
# don't contract on (the framework computes its own adjusted series).
_PRICE_COLS = ("open", "high", "low", "close")


class DataContractError(ValueError):
    """A loaded frame violated the OHLCV data contract.

    The message names the ticker, the offending date (when one row is at fault),
    and the rule that was broken — enough for a user to act on without
    opening a debugger.
    """


def _first_bad_date(df: pd.DataFrame, mask: pd.Series) -> str:
    """Human-readable date of the first ``True`` in ``mask`` (for error messages).

    Falls back to the row's positional index if there is no ``date`` column or
    the value is missing, so the message is always informative.
    """
    if not mask.any():
        return "?"
    pos = int(np.argmax(mask.to_numpy()))
    if "date" in df.columns:
        val = df["date"].iloc[pos]
        if pd.notna(val):
            # Normalise Timestamps to plain YYYY-MM-DD; pass strings through.
            return str(val.date()) if isinstance(val, pd.Timestamp) else str(val)
    return f"row {pos}"


def validate_ohlcv(df: pd.DataFrame, *, ticker: str = "?") -> pd.DataFrame:
    """Validate one symbol's OHLCV frame against the data contract.

    Returns the frame unchanged on success so the call can be used inline
    (``df = validate_ohlcv(df, ticker=sym)``). Raises :class:`DataContractError`
    with a precise, actionable message on the first violation found.

    Parameters
    ----------
    df:
        Tidy, date-ascending OHLCV frame with columns
        ``date, open, high, low, close`` (``volume`` optional). An empty frame is
        valid (a symbol with no history is not a contract breach).
    ticker:
        Symbol name, used only to make error messages actionable.

    Notes
    -----
    Designed for a *once-per-ticker* seam (fetch / cache-write). It is a few
    vectorised passes over the frame, but should not sit on a per-read hot path.
    """
    if df.empty:
        return df

    missing = [c for c in (*_PRICE_COLS, "date") if c not in df.columns]
    if missing:
        raise DataContractError(
            f"schema violation: {ticker} OHLCV frame missing column(s) {missing}"
        )

    # 1. Non-negative prices. NaN AND exact 0 are tolerated: both are real vendor
    #    artifacts on thin / halted / delisted names (EODHD ships a 0 low or a NaN
    #    bar for a no-trade session) that the loader/engine neutralise downstream.
    #    Only a *negative* price is structurally impossible — that is the corruption
    #    this rule catches.
    for col in _PRICE_COLS:
        s = df[col]
        bad = s.notna() & (s < 0)
        if bad.any():
            when = _first_bad_date(df, bad)
            raise DataContractError(
                f"price violation: {ticker} {when} {col}={s[bad].iloc[0]!r} "
                f"is negative (open/high/low/close must be >= 0)"
            )

    # 2. OHLC consistency on fully-populated bars (skip rows with a NaN leg so a
    #    vendor gap is not misreported as an ordering breach).
    o, h, low_, c = (df[col] for col in _PRICE_COLS)
    populated = o.notna() & h.notna() & low_.notna() & c.notna()
    low_bad = populated & (low_ > np.minimum(o, c))
    if low_bad.any():
        when = _first_bad_date(df, low_bad)
        raise DataContractError(
            f"OHLC violation: {ticker} {when} low>min(open,close) "
            "(low must be <= both open and close)"
        )
    high_bad = populated & (h < np.maximum(o, c))
    if high_bad.any():
        when = _first_bad_date(df, high_bad)
        raise DataContractError(
            f"OHLC violation: {ticker} {when} high<max(open,close) "
            "(high must be >= both open and close)"
        )
    # Catch the textbook high<low case explicitly for the clearest message.
    hl_bad = populated & (h < low_)
    if hl_bad.any():
        when = _first_bad_date(df, hl_bad)
        raise DataContractError(f"OHLC violation: {ticker} {when} high<low")

    # 4. Clean dates: unique and strictly increasing.
    dates = df["date"]
    if dates.duplicated().any():
        dup = dates[dates.duplicated()].iloc[0]
        raise DataContractError(
            f"date violation: {ticker} duplicate date {dup!r} "
            "(dates must be unique)"
        )
    if not dates.is_monotonic_increasing:
        # Find the first out-of-order step for an actionable message.
        order_bad = dates.to_numpy()[1:] <= dates.to_numpy()[:-1]
        pos = int(np.argmax(order_bad)) + 1
        raise DataContractError(
            f"date violation: {ticker} {dates.iloc[pos]!r} is not after "
            f"{dates.iloc[pos - 1]!r} (dates must be strictly increasing)"
        )

    return df


def validate_returns(s: pd.Series, *, name: str = "?") -> pd.Series:
    """Validate a returns Series: finite, no NaN (an optional, lighter contract).

    Returns the Series unchanged on success. Raises :class:`DataContractError`
    if any value is NaN or non-finite (``inf``) — the two failure modes a bad
    price bar produces downstream. Bounds on magnitude are intentionally *not*
    enforced: a real one-day move can legitimately exceed any fixed threshold.
    """
    if s.empty:
        return s
    bad = ~np.isfinite(s.to_numpy(dtype="float64"))
    if bad.any():
        pos = int(np.argmax(bad))
        label = s.index[pos]
        raise DataContractError(
            f"returns violation: {name} {label!r} is {s.iloc[pos]!r} "
            "(returns must be finite and non-NaN)"
        )
    return s
