"""VWAP-paper intraday engine — four academic intraday mechanisms on CME futures.

Shared, reusable engine for a family of **session-anchored VWAP** intraday strategies
ported from four papers. Every strategy in the family is flat overnight (entries and
exits inside one RTH session), reads 1-/5-minute Databento bars through
``data.intraday``, and emits **one net return per trading session**, so the metrics
annualise at 252 and are directly comparable to every other strategy in the book.

The four mechanisms (``variant`` in ``config.extra``)
-----------------------------------------------------

``vwap_trend`` — **A. Zarattini & Aziz (SSRN 4631351), "VWAP Trend".**
    Long while price is above the session VWAP, short while below; the position flips
    on every VWAP cross. The paper reports Sharpe ~2.1 on QQQ/TQQQ 2018-2023 — a single
    asset family over six years, so the out-of-sample question is wide open. The paper
    does **not** specify a stop; one is added here (``stop_atr_mult`` x the prior-day
    ATR) because without it the per-day risk is unbounded, which makes the result
    impossible to size or to evaluate against prop-trading risk rules.

    ``single_entry_per_day`` (not in the paper): take only the FIRST valid signal of the
    session and hold it to stop-or-close, ignoring later crosses. ``min_bars_cooldown``
    (not in the paper): wait N bars after a close before re-entering.

``opening_fade`` — **C. Grant, Wolf & Yu (SSRN 689282, JBF 2005), opening reversal.**
    Large moves in the first minutes of the session tend to revert. Fade the first
    ``open_window_min`` minutes when that move exceeds ``threshold_atr_mult`` x ATR;
    stop at the size of the faded move, target ``reward_r`` x that distance. The authors
    themselves warn that significance "drops a lot once transaction costs are applied" —
    which is exactly why this engine charges a round-trip cost on every entry.

``adx_vwap_reversion`` — **B. Bhatti (SSRN 6454659), ADX-conditioned reversion.**
    Fade an extreme of the prior session's range when price is far from VWAP *and* ADX
    is falling from a peak (momentum exhaustion). The paper is FX and explicitly leaves
    "backtesting for future work" — nobody, including the authors, has tested it. The
    thresholds (``adx_min_prior=25``, the classic "established trend" level, and a
    ``dev_threshold_mult=1`` intraday sigma) are a reasonable reading, not paper values.

``regime_routed`` — **D. Lee (SSRN 6438039), volatility-normalised VWAP-deviation regime.**
    Not a strategy but a filter: classify each day trend vs reversion from the mean
    absolute VWAP deviation, normalised by intraday volatility, over the first
    ``regime_n_bars`` bars. Route trend days to ``vwap_trend`` and reversion days to
    ``opening_fade``.

    **Look-ahead caveat, measured not assumed.** The regime is only known after
    ``regime_n_bars`` bars, but ``vwap_trend`` starts trading at bar 1. Routing a whole
    session's ``vwap_trend`` P&L on a label that did not exist until bar N is a genuine
    leak. ``regime_causal=True`` closes it by making the routed ``vwap_trend`` branch
    start trading only *after* the classification bar. The default is ``False`` — the
    paper-faithful wiring — and both are run and reported so the size of the leak is a
    number, not an opinion.

Causality
---------
Every input is strictly causal. ``daily_atr`` is a Wilder ATR over *daily* bars shifted
one session (known at the open of the session that uses it); ``prior_high``/``prior_low``
are the previous session's extremes; the session VWAP and the expanding deviation-sigma
are cumulative *within* the session; ADX is Wilder-smoothed over past bars only. Entries
fill at the CLOSE of the triggering bar and stops/targets are evaluated from the NEXT bar
onward — never on the entry bar itself.

Accounting
----------
The reference logic works in **index points**; a session's points P&L is converted to a
return on notional by dividing by that session's opening price (one contract held, the
same fixed-notional convention as the other futures strategies in the book). Each entry
is charged a round trip of ``2 x commission_bps``. Roll sessions are skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..engine import BacktestResult, result_from_returns
from ..strategy import StrategyBase, StrategyConfig

VARIANTS = ("vwap_trend", "opening_fade", "adx_vwap_reversion", "regime_routed")


# ---------------------------------------------------------------------------
# Causal feature preparation
# ---------------------------------------------------------------------------
def wilder_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
) -> np.ndarray:
    """Wilder's ADX over a bar stream. Causal: bar i uses only bars <= i.

    Smoothing is the Wilder EMA (alpha = 1/period). Computed continuously across the
    concatenated RTH stream; the session boundary produces one gap bar per day, the
    standard treatment for RTH-only intraday ADX.
    """
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    prev_close = np.concatenate(([np.nan], close[:-1]))
    prev_high = np.concatenate(([np.nan], high[:-1]))
    prev_low = np.concatenate(([np.nan], low[:-1]))

    tr = np.nanmax(
        np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0
    )
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()
    p_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()
    m_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * p_s / tr_s
        minus_di = 100.0 * m_s / tr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isfinite(dx), dx, np.nan)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()
    return np.asarray(adx, dtype="float64")


def prepare_bars(df: pd.DataFrame, atr_period: int = 14, adx_period: int = 14) -> dict[str, Any]:
    """Intraday OHLCV frame (``dt``/``session``/OHLCV) -> the arrays the mechanisms read.

    Returns flat per-bar arrays plus per-session index bounds and per-session causal
    scalars (prior-day ATR, prior-session high/low).
    """
    df = df.sort_values("dt").reset_index(drop=True)
    session = pd.DatetimeIndex(df["session"])
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    vol = df["volume"].to_numpy(dtype="float64")

    # Session-anchored VWAP: cumulative typical-price*volume / cumulative volume, reset
    # at each session open. Cumulative-within-session => causal by construction.
    typical = (h + low + c) / 3.0
    g = df.groupby("session", sort=False)
    pv = pd.Series(typical * vol).groupby(session, sort=False).cumsum().to_numpy()
    cv = pd.Series(vol).groupby(session, sort=False).cumsum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cv > 0, pv / cv, c)

    # Deviation from VWAP and its expanding within-session sigma (causal).
    dev = c - vwap
    dev_std = (
        pd.Series(dev).groupby(session, sort=False).expanding().std().reset_index(drop=True)
    ).to_numpy(dtype="float64")

    adx = wilder_adx(h, low, c, adx_period)

    # Per-session bounds.
    codes, uniq = pd.factorize(session, sort=False)
    n_sessions = len(uniq)
    starts = np.searchsorted(codes, np.arange(n_sessions), side="left")
    ends = np.searchsorted(codes, np.arange(n_sessions), side="right")

    # Minutes elapsed since the session's first bar (for the opening window).
    ts = df["dt"].to_numpy(dtype="datetime64[ns]")
    first_ts = ts[starts][codes]
    minutes = (ts - first_ts) / np.timedelta64(1, "m")

    # Daily bars -> Wilder ATR, SHIFTED one session so the value is known at the open.
    d_hi = g["high"].max().to_numpy(dtype="float64")
    d_lo = g["low"].min().to_numpy(dtype="float64")
    d_cl = g["close"].last().to_numpy(dtype="float64")
    d_pc = np.concatenate(([np.nan], d_cl[:-1]))
    d_tr = np.nanmax(
        np.vstack([d_hi - d_lo, np.abs(d_hi - d_pc), np.abs(d_lo - d_pc)]), axis=0
    )
    atr_full = (
        pd.Series(d_tr).ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()
    ).to_numpy(dtype="float64")
    daily_atr = np.concatenate(([np.nan], atr_full[:-1]))          # causal: prior-day ATR
    prior_high = np.concatenate(([np.nan], d_hi[:-1]))
    prior_low = np.concatenate(([np.nan], d_lo[:-1]))

    return {
        "o": o, "h": h, "l": low, "c": c, "vwap": vwap, "dev": dev, "dev_std": dev_std,
        "adx": adx, "minutes": minutes, "starts": starts, "ends": ends,
        "daily_atr": daily_atr, "prior_high": prior_high, "prior_low": prior_low,
        "dates": pd.DatetimeIndex(uniq),
    }


# ---------------------------------------------------------------------------
# Trade simulation primitive (numpy mirror of the reference `_simulate_single_trade`)
# ---------------------------------------------------------------------------
def _simulate_single_trade(
    entry_price: float, side: int, stop_price: float, target_price: float,
    h: np.ndarray, low: np.ndarray, c: np.ndarray,
) -> tuple[float, float]:
    """Simulate one trade over the supplied forward bars -> (pnl_points, worst_mark).

    Stop is checked before target within a bar (the conservative tie-break when a single
    bar spans both). The worst mark uses each bar's adverse extreme (low if long, high if
    short), not its close.
    """
    worst_mark = 0.0
    n = len(c)
    for i in range(n):
        adverse = low[i] if side == 1 else h[i]
        worst_mark = min(worst_mark, (adverse - entry_price) * side)
        hit_stop = (low[i] <= stop_price) if side == 1 else (h[i] >= stop_price)
        hit_target = (h[i] >= target_price) if side == 1 else (low[i] <= target_price)
        if hit_stop:
            pnl = (stop_price - entry_price) * side
            return pnl, min(worst_mark, pnl)
        if hit_target:
            pnl = (target_price - entry_price) * side
            return pnl, worst_mark
    if n == 0:
        return 0.0, 0.0
    pnl = (c[-1] - entry_price) * side
    return pnl, min(worst_mark, pnl)


# ---------------------------------------------------------------------------
# A — VWAP trend (Zarattini & Aziz)
# ---------------------------------------------------------------------------
def _banded_sides(c: np.ndarray, vwap: np.ndarray, a: int, b: int, band: float) -> np.ndarray:
    """Per-bar side with a **hysteresis band** around VWAP.

    ``band = 0`` reproduces the paper's bare classifier exactly (``+1`` above VWAP,
    ``-1`` below). With ``band > 0`` a bar only *changes* the side once price clears
    VWAP by ``band`` (in price units); inside the band the previous side is carried
    forward, so a wobble across the line produces no transition at all.

    This is deliberately applied to the side *classifier* rather than as a veto on the
    entry: vetoing the entry would consume the cross (the next bar's ``side_prev``
    would already be the new side) and the signal would be lost for the rest of the
    session. Carrying the side forward keeps the transition intact until price
    genuinely commits.
    """
    n = b - a
    if band <= 0:
        return np.where(c[a:b] > vwap[a:b], 1, -1).astype("int64")
    out = np.empty(n, dtype="int64")
    prev = 1 if c[a] > vwap[a] else -1
    for i in range(n):
        d = c[a + i] - vwap[a + i]
        if d > band:
            prev = 1
        elif d < -band:
            prev = -1
        out[i] = prev
    return out


def run_vwap_trend(
    cache: dict[str, Any], *, stop_atr_mult: float = 1.0, single_entry_per_day: bool = False,
    min_bars_cooldown: int = 0, max_entries_per_day: int | None = None, start_bar: int = 1,
    entry_band_atr: float = 0.0, slope_bars: int = 0, flat_before_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (pnl_points, worst_mark, n_entries) per session.

    ``start_bar`` lets the regime router defer the first entry until the classification
    bar (causal mode).

    Entry-quality knobs (all OFF by default, so the defaults are bit-identical to the
    paper's rule — they exist to be swept in-sample, not to be assumed):

    ``entry_band_atr``   hysteresis band around VWAP, in units of the prior-day ATR.
                         Attacks the mechanism's actual failure mode: crosses cluster
                         where price sits *on* VWAP, and each one pays a round trip.
    ``slope_bars``       require the session VWAP itself to be sloping with the trade
                         over the last N bars. A rejected signal still CLOSES an open
                         position (the old direction is no longer wanted) but does not
                         open the new one — "flat, not confident enough to reverse".
    ``flat_before_min``  force flat this many minutes into the session and take no
                         further entries (avoids carrying risk into the close).
    """
    h, low, c = cache["h"], cache["l"], cache["c"]
    vwap, starts, ends = cache["vwap"], cache["starts"], cache["ends"]
    minutes = cache["minutes"]
    atr_arr = cache["daily_atr"]
    n_sessions = len(starts)
    pnl_out = np.zeros(n_sessions)
    worst_out = np.zeros(n_sessions)
    ent_out = np.zeros(n_sessions)
    eff_max = 1 if single_entry_per_day else max_entries_per_day

    for k in range(n_sessions):
        a, b = starts[k], ends[k]
        n = b - a
        atr = atr_arr[k]
        if not np.isfinite(atr) or atr <= 0 or n < 3:
            continue
        stop_dist = atr * stop_atr_mult
        sides = _banded_sides(c, vwap, a, b, entry_band_atr * atr)
        position, entry_price, stop_price = 0, 0.0, 0.0
        realized, worst, n_entries, cooldown_until = 0.0, 0.0, 0, -1

        for i in range(max(1, start_bar), n):
            j = a + i
            side_now, side_prev = int(sides[i]), int(sides[i - 1])

            if position != 0:
                hit_stop = (low[j] <= stop_price) if position == 1 else (h[j] >= stop_price)
                adverse = low[j] if position == 1 else h[j]
                worst = min(worst, realized + (adverse - entry_price) * position)
                if hit_stop:
                    realized += (stop_price - entry_price) * position
                    position = 0
                    cooldown_until = i + min_bars_cooldown

            # Time exit: flat and done for the session.
            if flat_before_min is not None and minutes[j] >= flat_before_min:
                if position != 0:
                    worst = min(worst, realized + (c[j] - entry_price) * position)
                    realized += (c[j] - entry_price) * position
                    position = 0
                break

            available = eff_max is None or n_entries < eff_max
            if available and side_now != side_prev and position != side_now and i >= cooldown_until:
                confirmed = True
                if slope_bars > 0:
                    p = j - slope_bars
                    confirmed = p >= a and (
                        vwap[j] > vwap[p] if side_now == 1 else vwap[j] < vwap[p]
                    )
                if position != 0:
                    realized += (c[j] - entry_price) * position
                    cooldown_until = i + min_bars_cooldown
                    position = 0
                if confirmed:
                    position = side_now
                    entry_price = c[j]
                    stop_price = entry_price - stop_dist * position
                    n_entries += 1

        if position != 0:
            last = b - 1
            adverse = low[last] if position == 1 else h[last]
            worst = min(worst, realized + (adverse - entry_price) * position)
            realized += (c[last] - entry_price) * position

        pnl_out[k] = realized
        worst_out[k] = min(worst, realized)
        ent_out[k] = n_entries
    return pnl_out, worst_out, ent_out


# ---------------------------------------------------------------------------
# C — Opening fade (Grant, Wolf & Yu)
# ---------------------------------------------------------------------------
def run_opening_fade(
    cache: dict[str, Any], *, open_window_min: int = 30, reward_r: float = 1.5,
    threshold_atr_mult: float = 0.4, max_threshold_atr_mult: float | None = None,
    stop_cap_atr: float | None = None, min_entry_bar: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entry-quality knobs (OFF by default, so defaults reproduce the paper exactly):

    ``max_threshold_atr_mult``  skip the fade when the opening move is *too* large — an
                                extreme open is more often a real trend day than an
                                over-reaction, so the mechanism has an upper band too.
    ``stop_cap_atr``            cap the stop at N x ATR. The paper's stop is the size of
                                the faded move, which on a violent open is enormous and
                                makes the risk per trade wildly inconsistent.

    ``min_entry_bar`` is a CAUSALITY gate, not a tuning knob. The regime router computes
    its label from the session's first K bars; if this branch's natural entry lands
    before bar K it would be trading on a label that does not exist yet. Setting
    ``min_entry_bar=K`` pushes the entry (and the measured opening move with it) out to
    bar K. Default 0 = the paper's behaviour, unchanged.
    """
    o, h, low, c = cache["o"], cache["h"], cache["l"], cache["c"]
    minutes, starts, ends = cache["minutes"], cache["starts"], cache["ends"]
    atr_arr = cache["daily_atr"]
    n_sessions = len(starts)
    pnl_out = np.zeros(n_sessions)
    worst_out = np.zeros(n_sessions)
    ent_out = np.zeros(n_sessions)

    for k in range(n_sessions):
        a, b = starts[k], ends[k]
        n = b - a
        atr = atr_arr[k]
        if not np.isfinite(atr) or atr <= 0 or n < 5:
            continue
        session_open = o[a]
        nw = int(np.count_nonzero(minutes[a:b] <= open_window_min))
        if nw == 0:
            continue
        nw = max(nw, min_entry_bar + 1)       # causality gate (no-op when min_entry_bar=0)
        if nw >= n:
            continue
        opening_ret = c[a + nw - 1] - session_open
        if abs(opening_ret) < threshold_atr_mult * atr:
            continue
        if max_threshold_atr_mult is not None and abs(opening_ret) > max_threshold_atr_mult * atr:
            continue
        side = -1 if opening_ret > 0 else 1
        entry_price = c[a + nw - 1]
        stop_dist = abs(opening_ret)
        if stop_cap_atr is not None:
            stop_dist = min(stop_dist, stop_cap_atr * atr)
        stop_price = entry_price - stop_dist * side
        target_price = entry_price + stop_dist * reward_r * side
        pnl, worst = _simulate_single_trade(
            entry_price, side, stop_price, target_price, h[a + nw:b], low[a + nw:b], c[a + nw:b]
        )
        pnl_out[k] = pnl
        worst_out[k] = worst
        ent_out[k] = 1
    return pnl_out, worst_out, ent_out


# ---------------------------------------------------------------------------
# B — ADX-conditioned VWAP reversion (Bhatti)
# ---------------------------------------------------------------------------
def run_adx_vwap_reversion(
    cache: dict[str, Any], *, adx_decline_lookback: int = 3, adx_min_prior: float = 25.0,
    dev_threshold_mult: float = 1.0, reward_r: float = 1.5, stop_atr_mult: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``stop_atr_mult`` exposes the stop the paper hardcodes at 0.5 x ATR, so the
    risk/reward geometry can be swept in-sample rather than assumed."""
    h, low, c = cache["h"], cache["l"], cache["c"]
    dev, dev_std, adx = cache["dev"], cache["dev_std"], cache["adx"]
    starts, ends = cache["starts"], cache["ends"]
    atr_arr, p_hi, p_lo = cache["daily_atr"], cache["prior_high"], cache["prior_low"]
    n_sessions = len(starts)
    pnl_out = np.zeros(n_sessions)
    worst_out = np.zeros(n_sessions)
    ent_out = np.zeros(n_sessions)

    for k in range(n_sessions):
        a, b = starts[k], ends[k]
        n = b - a
        atr, prior_high, prior_low = atr_arr[k], p_hi[k], p_lo[k]
        if (not np.isfinite(atr) or atr <= 0
                or not np.isfinite(prior_high) or not np.isfinite(prior_low)):
            continue
        for i in range(adx_decline_lookback, n):
            j = a + i
            adx_now, adx_prior = adx[j], adx[j - adx_decline_lookback]
            sd = dev_std[j]
            if (not np.isfinite(adx_now) or not np.isfinite(adx_prior)
                    or not np.isfinite(sd) or sd == 0):
                continue
            if not (adx_prior >= adx_min_prior and adx_now < adx_prior):
                continue
            big_pos = dev[j] >= dev_threshold_mult * sd
            big_neg = dev[j] <= -dev_threshold_mult * sd
            if h[j] >= prior_high and big_pos:
                side = -1
            elif low[j] <= prior_low and big_neg:
                side = 1
            else:
                continue

            entry_price = c[j]
            stop_dist = stop_atr_mult * atr
            stop_price = entry_price - stop_dist * side
            target_price = entry_price + stop_dist * reward_r * side
            pnl, worst = _simulate_single_trade(
                entry_price, side, stop_price, target_price, h[j + 1:b], low[j + 1:b], c[j + 1:b]
            )
            pnl_out[k] = pnl
            worst_out[k] = worst
            ent_out[k] = 1
            break
    return pnl_out, worst_out, ent_out


# ---------------------------------------------------------------------------
# D — Lee regime classifier + router
# ---------------------------------------------------------------------------
def daily_regime_vwap_deviation(
    cache: dict[str, Any], *, dev_threshold: float = 1.5, n_bars_check: int = 12,
) -> np.ndarray:
    """Per-session regime: 1 = trend, 0 = reversion, -1 = undefined.

    Mean |VWAP deviation / expanding intraday sigma| over the first ``n_bars_check``
    bars; >= ``dev_threshold`` => trend.
    """
    dev, dev_std = cache["dev"], cache["dev_std"]
    starts, ends = cache["starts"], cache["ends"]
    n_sessions = len(starts)
    out = np.full(n_sessions, -1, dtype="int64")
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(dev_std > 0, np.abs(dev / dev_std), np.nan)
    for k in range(n_sessions):
        a, b = starts[k], ends[k]
        seg = norm[a:min(a + n_bars_check, b)]
        if len(seg) == 0 or np.all(~np.isfinite(seg)):
            continue
        m = float(np.nanmean(seg))
        if not np.isfinite(m):
            continue
        out[k] = 1 if m >= dev_threshold else 0
    return out


class Strategy(StrategyBase):
    """StrategyBase wrapper — ``config.extra["variant"]`` selects the mechanism."""

    def __init__(self, config: StrategyConfig, data: Any) -> None:
        super().__init__(config)
        self.data = data
        e = dict(config.extra)
        self._e = e
        self.symbol = str(e.get("symbol", "ES"))
        self.interval = str(e.get("interval", "5m"))
        self.variant = str(e.get("variant", "vwap_trend"))
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {VARIANTS}")
        self.commission_bps = float(config.commission_bps)
        self.start_date = config.start_date
        self.end_date = config.end_date

    def _p(self, params: dict[str, Any], key: str, default: Any) -> Any:
        return params[key] if key in params else self._e.get(key, default)

    def param_grid(self) -> dict[str, list]:
        """Search space for `rigor optimize`. The first cell of every axis is the
        paper's own value, so the untuned default is always IN the grid and the
        optimiser can benchmark the winner against it (see docs/optimization.md)."""
        if self.variant in ("vwap_trend", "regime_routed"):
            g: dict[str, list] = {
                "stop_atr_mult": [1.0, 0.5, 1.5, 2.0],
                "entry_band_atr": [0.0, 0.05, 0.10, 0.20, 0.35],
                "min_bars_cooldown": [0, 3, 6],
                "max_entries_per_day": [0, 1, 2, 3],       # 0 = unlimited (paper)
                "slope_bars": [0, 3, 6],
            }
            if self.variant == "regime_routed":
                # Trimmed on the A-axes: the router multiplies mechanism A's grid by its
                # own two axes, so the full product would be ~4300 cells per instrument.
                g = {"stop_atr_mult": [1.0, 0.5, 1.5],
                     "entry_band_atr": [0.0, 0.10, 0.20],
                     "max_entries_per_day": [0, 1, 2],
                     "regime_dev_threshold": [1.5, 1.0, 2.0],
                     "regime_n_bars": [12, 30]}
            return g
        if self.variant == "opening_fade":
            return {"open_window_min": [30, 15, 60], "threshold_atr_mult": [0.4, 0.3, 0.6],
                    "reward_r": [1.5, 1.0, 2.0, 3.0],
                    "max_threshold_atr_mult": [0.0, 1.0, 1.5],   # 0 = no upper band (paper)
                    "stop_cap_atr": [0.0, 0.5, 1.0]}             # 0 = uncapped (paper)
        return {"adx_min_prior": [25.0, 20.0, 30.0], "dev_threshold_mult": [1.0, 1.5, 2.0],
                "reward_r": [1.5, 1.0, 2.0, 3.0], "stop_atr_mult": [0.5, 0.25, 1.0]}

    def build_cache(self) -> dict[str, Any]:
        df = self.data.intraday(self.symbol, interval=self.interval, session="rth")
        if df.empty:
            return {"empty": True, "dates": pd.DatetimeIndex([])}
        if self.start_date:
            df = df[df["dt"] >= pd.Timestamp(self.start_date)]
        if self.end_date:
            df = df[df["dt"] <= pd.Timestamp(self.end_date) + pd.Timedelta(days=1)]
        if df.empty:
            return {"empty": True, "dates": pd.DatetimeIndex([])}

        cache = prepare_bars(df)
        roll_fn = getattr(self.data, "roll_dates", None)
        rolls = set(pd.DatetimeIndex(roll_fn(self.symbol))) if roll_fn else set()
        cache["roll"] = np.asarray([d in rolls for d in cache["dates"]], dtype=bool)
        cache["session_open"] = cache["o"][cache["starts"]]
        cache["empty"] = False
        return cache

    def _trend_kw(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mechanism-A kwargs. ``max_entries_per_day`` uses 0 as the sentinel for
        "unlimited" so the axis stays type-consistent for the optimize gate."""
        mx = int(self._p(params, "max_entries_per_day", 0))
        return {
            "stop_atr_mult": float(self._p(params, "stop_atr_mult", 1.0)),
            "single_entry_per_day": bool(self._p(params, "single_entry_per_day", False)),
            "min_bars_cooldown": int(self._p(params, "min_bars_cooldown", 0)),
            "max_entries_per_day": None if mx <= 0 else mx,
            "entry_band_atr": float(self._p(params, "entry_band_atr", 0.0)),
            "slope_bars": int(self._p(params, "slope_bars", 0)),
            "flat_before_min": (lambda v: None if v <= 0 else float(v))(
                float(self._p(params, "flat_before_min", 0.0))),
        }

    def _fade_kw(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mechanism-C kwargs. 0 is the sentinel for "off" on the two optional bands."""
        mx = float(self._p(params, "max_threshold_atr_mult", 0.0))
        cap = float(self._p(params, "stop_cap_atr", 0.0))
        return {
            "open_window_min": int(self._p(params, "open_window_min", 30)),
            "reward_r": float(self._p(params, "reward_r", 1.5)),
            "threshold_atr_mult": float(self._p(params, "threshold_atr_mult", 0.4)),
            "max_threshold_atr_mult": None if mx <= 0 else mx,
            "stop_cap_atr": None if cap <= 0 else cap,
        }

    def _points_for(
        self, cache: dict[str, Any], params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """-> (pnl_points, n_entries) per session for the configured variant."""
        v = self.variant
        if v == "vwap_trend":
            pnl, _, ent = run_vwap_trend(cache, **self._trend_kw(params))
            return pnl, ent
        if v == "opening_fade":
            pnl, _, ent = run_opening_fade(cache, **self._fade_kw(params))
            return pnl, ent
        if v == "adx_vwap_reversion":
            pnl, _, ent = run_adx_vwap_reversion(
                cache,
                adx_decline_lookback=int(self._p(params, "adx_decline_lookback", 3)),
                adx_min_prior=float(self._p(params, "adx_min_prior", 25.0)),
                dev_threshold_mult=float(self._p(params, "dev_threshold_mult", 1.0)),
                reward_r=float(self._p(params, "reward_r", 1.5)),
                stop_atr_mult=float(self._p(params, "stop_atr_mult", 0.5)),
            )
            return pnl, ent

        # regime_routed
        n_bars = int(self._p(params, "regime_n_bars", 12))
        causal = bool(self._p(params, "regime_causal", False))
        regime = daily_regime_vwap_deviation(
            cache, dev_threshold=float(self._p(params, "regime_dev_threshold", 1.5)),
            n_bars_check=n_bars,
        )
        # In causal mode BOTH branches must wait for the label. The trend branch defers
        # its first entry to bar n_bars; the fade branch defers its entry the same way —
        # its natural entry (end of the 30-minute window) otherwise lands BEFORE the
        # label exists, which is the same leak in the other branch.
        pnl_t, _, ent_t = run_vwap_trend(
            cache, **(self._trend_kw(params) | {"start_bar": n_bars if causal else 1})
        )
        pnl_r, _, ent_r = run_opening_fade(
            cache, **(self._fade_kw(params) | {"min_entry_bar": n_bars if causal else 0})
        )
        pnl = np.where(regime == 1, pnl_t, np.where(regime == 0, pnl_r, 0.0))
        ent = np.where(regime == 1, ent_t, np.where(regime == 0, ent_r, 0.0))
        return pnl, ent

    def run_backtest(self, cache: dict[str, Any], params: dict[str, Any]) -> BacktestResult:
        if cache.get("empty"):
            return result_from_returns(pd.Series(dtype="float64", name="returns"))
        pnl, ent = self._points_for(cache, params)
        notional = cache["session_open"]
        with np.errstate(divide="ignore", invalid="ignore"):
            gross = np.where(notional > 0, pnl / notional, 0.0)
        cost = ent * 2.0 * self.commission_bps / 1e4        # round trip per entry
        net = np.where(cache["roll"], 0.0, gross - cost)     # roll sessions stand aside
        net = np.where(np.isfinite(net), net, 0.0)
        ser = pd.Series(net, index=cache["dates"], name="returns")
        # `eval_start` reports only the out-of-sample window while still building every
        # feature (ATR, ADX, prior extremes) on the FULL series. Trimming `start_date`
        # instead would throw away the warmup and silently change the first ~14 sessions,
        # so the committed report card would not equal the OOS number that was measured.
        ev = self._e.get("eval_start")
        if ev:
            ser = ser[ser.index >= pd.Timestamp(str(ev))]
        return result_from_returns(ser)


def build(config: StrategyConfig, data: Any) -> Strategy:
    return Strategy(config, data)
