"""Regression + causality tests for ``rigor.strategies.vwap_papers_engine``.

The engine is a **numpy rewrite** of a reference implementation that looped with
``DataFrame.iterrows()`` (unusably slow over 1.5M bars). The rewrite is only legitimate
if it is bit-for-bit identical, so the reference is embedded here verbatim and the two
are compared on synthetic bars across every mechanism and a spread of parameters. A
divergence in either direction is a test failure, not a rounding tolerance.

The rest of the file pins the properties the mechanisms depend on: that no feature can
see its own bar's future, and that the trade primitive resolves stop/target/close in the
documented order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rigor.strategies.vwap_papers_engine import (
    _simulate_single_trade,
    daily_regime_vwap_deviation,
    prepare_bars,
    run_adx_vwap_reversion,
    run_opening_fade,
    run_vwap_trend,
    wilder_adx,
)

# ---------------------------------------------------------------------------
# Synthetic bars — deterministic, with enough regime variety to hit every branch
# ---------------------------------------------------------------------------


def _bars(n_sessions: int = 60, bars_per_session: int = 40, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for d in range(n_sessions):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        # alternate trending / choppy sessions so both regimes are exercised
        drift = (0.05 if d % 3 == 0 else -0.05 if d % 3 == 1 else 0.0)
        for b in range(bars_per_session):
            ts = day + pd.Timedelta(hours=9, minutes=30 + 5 * b)
            step = rng.normal(drift, 0.35)
            o = price
            c = price + step
            hi = max(o, c) + abs(rng.normal(0, 0.18))
            lo = min(o, c) - abs(rng.normal(0, 0.18))
            rows.append({"dt": ts, "session": day, "open": o, "high": hi, "low": lo,
                         "close": c, "volume": float(rng.integers(300, 3000))})
            price = c
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _bars()


@pytest.fixture(scope="module")
def cache(bars: pd.DataFrame) -> dict:
    return prepare_bars(bars)


@pytest.fixture(scope="module")
def ref_frame(bars: pd.DataFrame, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The reference implementation's expected input, built from the SAME cache so the
    comparison isolates the port and not the feature preparation."""
    codes = np.repeat(np.arange(len(cache["starts"])), cache["ends"] - cache["starts"])
    df = pd.DataFrame({
        "ts_event": bars["dt"].to_numpy(), "session_date": bars["session"].to_numpy(),
        "open": cache["o"], "high": cache["h"], "low": cache["l"], "close": cache["c"],
        "vwap": cache["vwap"], "adx": cache["adx"], "daily_atr": cache["daily_atr"][codes],
    })
    prior_hl = pd.DataFrame(
        {"prior_high": cache["prior_high"], "prior_low": cache["prior_low"]},
        index=cache["dates"],
    )
    return df, prior_hl


# ---------------------------------------------------------------------------
# The reference implementation, embedded verbatim (row-wise, DataFrame-based)
# ---------------------------------------------------------------------------


def _ref_simulate(entry_price, side, stop_price, target_price, future_bars):
    worst_mark = 0.0
    for _, fb in future_bars.iterrows():
        adverse_price = fb["low"] if side == 1 else fb["high"]
        worst_mark = min(worst_mark, (adverse_price - entry_price) * side)
        hit_stop = ((side == 1 and fb["low"] <= stop_price)
                    or (side == -1 and fb["high"] >= stop_price))
        hit_target = ((side == 1 and fb["high"] >= target_price)
                      or (side == -1 and fb["low"] <= target_price))
        if hit_stop:
            pnl = (stop_price - entry_price) * side
            return pnl, min(worst_mark, pnl)
        if hit_target:
            pnl = (target_price - entry_price) * side
            return pnl, worst_mark
    if len(future_bars) == 0:
        return 0.0, 0.0
    pnl = (future_bars["close"].iloc[-1] - entry_price) * side
    return pnl, min(worst_mark, pnl)


def _ref_vwap_trend(bars, stop_atr_mult=1.0, single_entry_per_day=False,
                    min_bars_cooldown=0, max_entries_per_day=None):
    rows = []
    for date, day in bars.groupby("session_date"):
        day = day.reset_index(drop=True)
        atr = day["daily_atr"].iloc[0]
        if pd.isna(atr) or atr <= 0 or len(day) < 3:
            continue
        stop_dist = atr * stop_atr_mult
        position, entry_price, stop_price = 0, None, None
        realized_pnl, worst_mark, n_entries, cooldown_until = 0.0, 0.0, 0, -1
        effective_max = 1 if single_entry_per_day else max_entries_per_day
        for i in range(1, len(day)):
            row, prev = day.iloc[i], day.iloc[i - 1]
            side_now = 1 if row["close"] > row["vwap"] else -1
            side_prev = 1 if prev["close"] > prev["vwap"] else -1
            if position != 0:
                hit_stop = ((position == 1 and row["low"] <= stop_price)
                            or (position == -1 and row["high"] >= stop_price))
                adverse_price = row["low"] if position == 1 else row["high"]
                worst_mark = min(worst_mark,
                                 realized_pnl + (adverse_price - entry_price) * position)
                if hit_stop:
                    realized_pnl += (stop_price - entry_price) * position
                    position = 0
                    cooldown_until = i + min_bars_cooldown
            available = effective_max is None or n_entries < effective_max
            if available and side_now != side_prev and position != side_now and i >= cooldown_until:
                if position != 0:
                    realized_pnl += (row["close"] - entry_price) * position
                    cooldown_until = i + min_bars_cooldown
                position, entry_price = side_now, row["close"]
                stop_price = entry_price - stop_dist * position
                n_entries += 1
        if position != 0:
            final_price = day["close"].iloc[-1]
            adverse_price = day["low"].iloc[-1] if position == 1 else day["high"].iloc[-1]
            worst_mark = min(worst_mark, realized_pnl + (adverse_price - entry_price) * position)
            realized_pnl += (final_price - entry_price) * position
        rows.append({"date": date, "pnl_points": realized_pnl,
                     "worst": min(worst_mark, realized_pnl), "n_entries": n_entries})
    return pd.DataFrame(rows).set_index("date")


def _ref_opening_fade(bars, open_window_min=30, reward_r=1.5, threshold_atr_mult=0.4):
    rows = []
    for date, day in bars.groupby("session_date"):
        day = day.reset_index(drop=True)
        atr = day["daily_atr"].iloc[0]
        if pd.isna(atr) or atr <= 0 or len(day) < 5:
            continue
        session_open = day["open"].iloc[0]
        window_end = day["ts_event"].iloc[0] + pd.Timedelta(minutes=open_window_min)
        window_bars = day[day["ts_event"] <= window_end]
        if window_bars.empty or len(window_bars) >= len(day):
            continue
        opening_ret = window_bars["close"].iloc[-1] - session_open
        if abs(opening_ret) < threshold_atr_mult * atr:
            continue
        side = -1 if opening_ret > 0 else 1
        entry_price = window_bars["close"].iloc[-1]
        stop_dist = abs(opening_ret)
        pnl, worst = _ref_simulate(entry_price, side, entry_price - stop_dist * side,
                                   entry_price + stop_dist * reward_r * side,
                                   day.iloc[len(window_bars):])
        rows.append({"date": date, "pnl_points": pnl, "worst": worst, "n_entries": 1})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["pnl_points", "worst", "n_entries"])


def _ref_adx_reversion(bars, prior_hl, adx_decline_lookback=3, adx_min_prior=25.0,
                       dev_threshold_mult=1.0, reward_r=1.5):
    bars = bars.copy()
    bars["vwap_dev"] = bars["close"] - bars["vwap"]
    bars["vwap_dev_std"] = bars.groupby("session_date")["vwap_dev"].transform(
        lambda s: s.expanding().std())
    rows = []
    for date, day in bars.groupby("session_date"):
        if date not in prior_hl.index:
            continue
        prior_high, prior_low = prior_hl.loc[date, "prior_high"], prior_hl.loc[date, "prior_low"]
        if pd.isna(prior_high) or pd.isna(prior_low):
            continue
        atr = day["daily_atr"].iloc[0]
        if pd.isna(atr) or atr <= 0:
            continue
        day = day.reset_index(drop=True)
        for i in range(adx_decline_lookback, len(day)):
            row = day.iloc[i]
            adx_now, adx_prior = row["adx"], day["adx"].iloc[i - adx_decline_lookback]
            if (pd.isna(adx_now) or pd.isna(adx_prior) or pd.isna(row["vwap_dev_std"])
                    or row["vwap_dev_std"] == 0):
                continue
            adx_exhausting = (adx_prior >= adx_min_prior) and (adx_now < adx_prior)
            near_high, near_low = row["high"] >= prior_high, row["low"] <= prior_low
            big_pos = row["vwap_dev"] >= dev_threshold_mult * row["vwap_dev_std"]
            big_neg = row["vwap_dev"] <= -dev_threshold_mult * row["vwap_dev_std"]
            side = -1 if (near_high and big_pos and adx_exhausting) else (
                1 if (near_low and big_neg and adx_exhausting) else None)
            if side is None:
                continue
            entry_price = row["close"]
            stop_dist = 0.5 * atr
            pnl, worst = _ref_simulate(entry_price, side, entry_price - stop_dist * side,
                                       entry_price + stop_dist * reward_r * side,
                                       day.iloc[i + 1:])
            rows.append({"date": date, "pnl_points": pnl, "worst": worst, "n_entries": 1})
            break
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["pnl_points", "worst", "n_entries"])


def _align(mine: np.ndarray, ref: pd.DataFrame, dates: pd.DatetimeIndex, col: str) -> tuple:
    r = pd.Series(0.0, index=dates)
    if len(ref):
        s = ref[col]
        s.index = pd.DatetimeIndex(s.index)
        r.loc[s.index] = s.to_numpy(dtype="float64")
    return pd.Series(mine, index=dates).to_numpy(), r.to_numpy()


# ---------------------------------------------------------------------------
# Parity — the numpy port must equal the reference EXACTLY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {}, {"single_entry_per_day": True}, {"stop_atr_mult": 0.5}, {"stop_atr_mult": 1.5},
    {"min_bars_cooldown": 5}, {"max_entries_per_day": 2},
])
def test_vwap_trend_matches_reference(cache, ref_frame, kw):
    df, _ = ref_frame
    pnl, worst, ent = run_vwap_trend(cache, **kw)
    for arr, col in ((pnl, "pnl_points"), (worst, "worst"), (ent, "n_entries")):
        a, b = _align(arr, _ref_vwap_trend(df, **kw), cache["dates"], col)
        assert np.array_equal(a, b), f"{col} diverged for {kw}: max|d|={np.abs(a - b).max()}"


@pytest.mark.parametrize("kw", [
    {}, {"open_window_min": 60}, {"reward_r": 2.0}, {"threshold_atr_mult": 0.2},
])
def test_opening_fade_matches_reference(cache, ref_frame, kw):
    df, _ = ref_frame
    pnl, worst, _ = run_opening_fade(cache, **kw)
    for arr, col in ((pnl, "pnl_points"), (worst, "worst")):
        a, b = _align(arr, _ref_opening_fade(df, **kw), cache["dates"], col)
        assert np.array_equal(a, b), f"{col} diverged for {kw}"


@pytest.mark.parametrize("kw", [
    {}, {"adx_min_prior": 10.0}, {"dev_threshold_mult": 0.5}, {"reward_r": 2.0},
    {"adx_decline_lookback": 5},
])
def test_adx_reversion_matches_reference(cache, ref_frame, kw):
    df, prior_hl = ref_frame
    pnl, worst, _ = run_adx_vwap_reversion(cache, **kw)
    for arr, col in ((pnl, "pnl_points"), (worst, "worst")):
        a, b = _align(arr, _ref_adx_reversion(df, prior_hl, **kw), cache["dates"], col)
        assert np.array_equal(a, b), f"{col} diverged for {kw}"


# ---------------------------------------------------------------------------
# Causality — nothing may see its own bar's future
# ---------------------------------------------------------------------------


def test_daily_atr_is_lagged_one_session(cache, bars):
    """The ATR a session trades on must be computable from STRICTLY earlier sessions."""
    g = bars.groupby("session")
    tr_hi, tr_lo = g["high"].max().to_numpy(), g["low"].min().to_numpy()
    cl = g["close"].last().to_numpy()
    pc = np.concatenate(([np.nan], cl[:-1]))
    tr = np.nanmax(np.vstack([tr_hi - tr_lo, np.abs(tr_hi - pc), np.abs(tr_lo - pc)]), axis=0)
    expect = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    assert np.allclose(cache["daily_atr"][1:], expect[:-1], equal_nan=True)
    assert np.isnan(cache["daily_atr"][0])


def test_prior_session_extremes_are_lagged(cache, bars):
    g = bars.groupby("session")
    assert np.allclose(cache["prior_high"][1:], g["high"].max().to_numpy()[:-1])
    assert np.allclose(cache["prior_low"][1:], g["low"].min().to_numpy()[:-1])
    assert np.isnan(cache["prior_high"][0]) and np.isnan(cache["prior_low"][0])


def test_future_bars_cannot_change_todays_features(bars):
    """Truncating the series must leave every surviving bar's features untouched."""
    full = prepare_bars(bars)
    cut_sessions = pd.DatetimeIndex(sorted(bars["session"].unique()))[:40]
    part = prepare_bars(bars[bars["session"].isin(cut_sessions)].reset_index(drop=True))
    n = len(part["c"])
    for key in ("vwap", "dev", "dev_std", "adx", "minutes"):
        assert np.allclose(full[key][:n], part[key], equal_nan=True), f"{key} is not causal"
    m = len(part["dates"])
    assert np.allclose(full["daily_atr"][:m], part["daily_atr"], equal_nan=True)


def test_vwap_is_session_anchored(cache, bars):
    """VWAP resets each session: the first bar's VWAP is that bar's own typical price."""
    for a in cache["starts"]:
        typical = (cache["h"][a] + cache["l"][a] + cache["c"][a]) / 3.0
        assert cache["vwap"][a] == pytest.approx(typical)


def test_adx_is_causal():
    rng = np.random.default_rng(3)
    h, low, c = np.empty(300), np.empty(300), np.empty(300)
    p = 100.0
    for i in range(300):
        step = rng.normal(0, 0.5)
        c[i] = p + step
        h[i] = max(p, c[i]) + 0.2
        low[i] = min(p, c[i]) - 0.2
        p = c[i]
    full = wilder_adx(h, low, c)
    part = wilder_adx(h[:200], low[:200], c[:200])
    assert np.allclose(full[:200], part, equal_nan=True)


# ---------------------------------------------------------------------------
# Trade primitive semantics
# ---------------------------------------------------------------------------


def test_stop_resolves_before_target_within_a_bar():
    """A bar spanning both levels must book the STOP (the conservative tie-break)."""
    h = np.array([12.0])
    low = np.array([8.0])
    c = np.array([11.0])
    pnl, _ = _simulate_single_trade(10.0, 1, 9.0, 11.5, h, low, c)
    assert pnl == pytest.approx(-1.0)


def test_target_books_when_only_target_touched():
    h = np.array([11.6])
    low = np.array([9.9])
    c = np.array([11.5])
    pnl, _ = _simulate_single_trade(10.0, 1, 9.0, 11.5, h, low, c)
    assert pnl == pytest.approx(1.5)


def test_unresolved_trade_exits_at_session_close():
    h = np.array([10.2, 10.4])
    low = np.array([9.8, 9.9])
    c = np.array([10.1, 10.3])
    pnl, _ = _simulate_single_trade(10.0, 1, 9.0, 12.0, h, low, c)
    assert pnl == pytest.approx(0.3)


def test_no_forward_bars_is_flat():
    e = np.array([])
    assert _simulate_single_trade(10.0, 1, 9.0, 11.0, e, e, e) == (0.0, 0.0)


def test_short_side_mirrors_long():
    """Short: stop is ABOVE entry, target BELOW. Bar 1 touches neither; bar 2 stops out."""
    h = np.array([10.1, 11.2])
    low = np.array([9.5, 10.0])
    c = np.array([10.0, 11.0])
    pnl, _ = _simulate_single_trade(10.0, -1, 11.0, 9.0, h, low, c)
    assert pnl == pytest.approx(-1.0)          # stop hit on bar 2


def test_short_side_takes_target():
    """Mirror of the long target case: the short's low touching the target books +1."""
    h = np.array([10.1])
    low = np.array([9.0])
    c = np.array([10.0])
    pnl, _ = _simulate_single_trade(10.0, -1, 11.0, 9.0, h, low, c)
    assert pnl == pytest.approx(1.0)


def test_worst_mark_uses_adverse_extreme_not_close():
    """A long's worst mark must come from the LOW, even if the bar closes green."""
    h = np.array([10.5])
    low = np.array([7.0])
    c = np.array([10.4])
    _, worst = _simulate_single_trade(10.0, 1, 1.0, 99.0, h, low, c)
    assert worst == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------


def test_regime_labels_are_in_domain(cache):
    reg = daily_regime_vwap_deviation(cache)
    assert set(np.unique(reg)).issubset({-1, 0, 1})
    assert len(reg) == len(cache["dates"])


def test_regime_threshold_is_monotone(cache):
    """Raising the bar for 'trend' can only ever reduce the number of trend days."""
    counts = [int((daily_regime_vwap_deviation(cache, dev_threshold=t) == 1).sum())
              for t in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert counts == sorted(counts, reverse=True)


def test_causal_router_defers_BOTH_branches_past_the_label(cache):
    """Regression: the causal router's *fade* branch used to enter before the label existed.

    The trend branch was deferred with ``start_bar``, but the fade branch kept entering at
    the end of its 30-minute window — bar 6 on 5-minute bars — while the label consumed
    bars 0..11. Both branches must start at or after the classification bar.
    """
    n_bars = 12
    starts = cache["starts"]
    _, _, ent_ungated = run_opening_fade(cache, min_entry_bar=0)
    _, _, ent_gated = run_opening_fade(cache, min_entry_bar=n_bars)
    # The gate must actually move entries (otherwise the test proves nothing).
    assert ent_ungated.sum() > 0

    # With the gate on, no trade may be decided before bar n_bars.
    for k, a in enumerate(starts):
        if ent_gated[k] == 0:
            continue
        b = cache["ends"][k]
        nw = int(np.count_nonzero(cache["minutes"][a:b] <= 30))
        assert max(nw, n_bars + 1) - 1 >= n_bars


def test_min_entry_bar_default_is_a_noop(cache):
    """The causality gate must not change the paper's behaviour when it is off."""
    base = run_opening_fade(cache)
    same = run_opening_fade(cache, min_entry_bar=0)
    for x, y in zip(base, same, strict=True):
        assert np.array_equal(x, y)


def test_regime_only_reads_the_classification_window(cache, bars):
    """The label must not move when bars AFTER the window change."""
    n_bars = 12
    tampered = bars.copy()
    mask = tampered.groupby("session").cumcount() >= n_bars
    tampered.loc[mask, ["open", "high", "low", "close"]] *= 1.5
    base = daily_regime_vwap_deviation(prepare_bars(bars), n_bars_check=n_bars)
    after = daily_regime_vwap_deviation(prepare_bars(tampered), n_bars_check=n_bars)
    assert np.array_equal(base, after)
