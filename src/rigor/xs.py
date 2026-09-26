"""Cross-sectional backtest primitives.

Pure/computational — accepts caller-supplied data panels and optional
membership masks. No data-source dependencies; wire in your own loader.

Conventions (consistent with engine.simulate_weights):
  * **Lag**: weights decided on date t are held over (t, t+1], i.e. shifted 1
    bar before touching returns. This matches ``simulate_weights``'s w.shift(1).
  * **Membership mask**: a bool DataFrame (date × symbol). On each date only
    symbols where mask=True are eligible for position. If None, all symbols
    with non-NaN signal are eligible.
  * **NaN safety**: degenerate dates (empty universe, all-NaN) produce zero
    weights and flat returns — never raise.
  * **Turnover cost**: L1 weight change × commission_bps / 10_000 per bar
    (same formula as engine.simulate_weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .engine import simulate_weights

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _rank_long_short_weights(
    signal_row: pd.Series,
    eligible: pd.Index,
    top_k: int | None,
    long_short: bool,
) -> dict[str, float]:
    """Compute equal-weight long/(short) weights for one cross-section.

    Parameters
    ----------
    signal_row:
        Signal values for all symbols on one date. NaN = ineligible.
    eligible:
        Symbols that are allowed to be traded on this date (membership gate).
    top_k:
        Number of top/bottom names to hold. None = top half.
    long_short:
        If True, short the bottom-k as well (long-short book). If False,
        long-only.

    Returns
    -------
    dict symbol -> weight  (weights sum to +1 for long-only; long leg sums to
    +1 and short leg to -1 for long-short, so gross = 2).
    """
    # Restrict to eligible symbols with a finite signal.
    avail = signal_row.reindex(eligible).dropna()
    n = len(avail)
    if n == 0:
        return {}

    k = top_k if top_k is not None else max(1, n // 2)
    k = min(k, n)

    ranked = avail.rank(method="first", ascending=True)  # 1 = lowest signal
    n_total = len(ranked)

    # Long: top-k by signal (highest ranks)
    long_idx = ranked[ranked > n_total - k].index
    n_long = len(long_idx)
    if n_long == 0:
        return {}

    w_per = 1.0 / n_long
    weights: dict[str, float] = dict.fromkeys(long_idx, w_per)

    if long_short and n_total >= 2 * k:
        short_idx = ranked[ranked <= k].index
        n_short = len(short_idx)
        for sym in short_idx:
            weights[sym] = -1.0 / n_short

    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def xs_daily_basket(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_k: int | None = None,
    long_short: bool = True,
    membership_mask: pd.DataFrame | None = None,
    commission_bps: float = 1.0,
    rebalance: int = 1,
) -> dict:
    """Daily cross-sectional basket: rank signal → equal-weight long(/short).

    Parameters
    ----------
    signal:
        Date × symbol DataFrame of raw signal values. Higher = more bullish.
    prices:
        Date × symbol DataFrame of adjusted prices (for return computation).
    top_k:
        Symbols per leg. None → top-half / bottom-half of eligible universe.
    long_short:
        True  → long top-k, short bottom-k (dollar-neutral).
        False → long top-k only (gross = 1).
    membership_mask:
        Optional bool DataFrame (date × symbol). A symbol is eligible on date t
        only if mask.loc[t, symbol] is True. Rows/columns are re-indexed against
        ``signal``; missing = False (excluded).
    commission_bps:
        Round-trip cost in basis points per unit of L1 weight change.
    rebalance:
        Rebalance frequency in trading days (1 = daily). On non-rebalance bars
        the prior-bar weight is carried forward by ``simulate_weights``.

    Returns
    -------
    dict with keys:
        ``result``     : BacktestResult
        ``weights``    : pd.DataFrame (date × symbol, target weights)
        ``n_long``     : pd.Series (number of long names each rebalance date)
        ``n_short``    : pd.Series (number of short names each rebalance date)
        ``turnover``   : pd.Series (daily L1 weight change)
    """
    signal = signal.sort_index()
    prices = prices.sort_index()

    # Align on common symbols.
    common_syms = signal.columns.intersection(prices.columns)
    signal = signal[common_syms]
    prices = prices[common_syms]

    # Build aligned mask on signal's grid (date × symbol → bool).
    if membership_mask is not None:
        mask = membership_mask.reindex(index=signal.index, columns=common_syms).fillna(False)
    else:
        mask = pd.DataFrame(True, index=signal.index, columns=common_syms)

    # Rebalance dates: every `rebalance`-th bar (0-indexed).
    reb_dates = signal.index[::rebalance]

    weights_dict: dict[pd.Timestamp, dict[str, float]] = {}
    n_long_map: dict[pd.Timestamp, int] = {}
    n_short_map: dict[pd.Timestamp, int] = {}

    for date in reb_dates:
        if date not in signal.index:
            continue
        sig_row = signal.loc[date]
        eligible = mask.columns[mask.loc[date].to_numpy(dtype=bool)]
        w = _rank_long_short_weights(sig_row, eligible, top_k, long_short)
        weights_dict[date] = w
        longs = {s for s, v in w.items() if v > 0}
        shorts = {s for s, v in w.items() if v < 0}
        n_long_map[date] = len(longs)
        n_short_map[date] = len(shorts)

    # Build weights DataFrame: fill with 0, then ffill between rebalances.
    w_df = pd.DataFrame(0.0, index=signal.index, columns=common_syms)
    for date, w in sorted(weights_dict.items()):
        for sym, val in w.items():
            w_df.loc[date, sym] = val

    # Forward-fill target weights between rebalances (simulate_weights will
    # shift by 1 bar before applying to returns, so no look-ahead).
    # We reset to 0 at each rebalance date (already done above), then ffill
    # only the non-rebalance rows.
    if rebalance > 1:
        # Set non-rebalance rows to NaN so ffill works from the prior reb date.
        non_reb_mask = ~w_df.index.isin(reb_dates)
        w_df.loc[non_reb_mask] = np.nan
        w_df = w_df.ffill().fillna(0.0)

    result = simulate_weights(w_df, prices, commission_bps=commission_bps)

    return {
        "result": result,
        "weights": w_df,
        "n_long": pd.Series(n_long_map, name="n_long"),
        "n_short": pd.Series(n_short_map, name="n_short"),
        "turnover": result.costs / (commission_bps / 1e4 + 1e-15),
    }


def monthly_xs_quintile_ls(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    membership_mask: pd.DataFrame | None = None,
    commission_bps: float = 1.0,
) -> dict:
    """Monthly quintile long/short: rebalance at end-of-month.

    Ranks the cross-section into ``n_quantiles`` buckets once per month (using
    the last available signal date of each month). Long the top quintile, short
    the bottom quintile, equal-weight within each leg.

    Parameters
    ----------
    signal:
        Date × symbol DataFrame. Higher signal = more bullish.
    prices:
        Date × symbol adjusted price panel.
    n_quantiles:
        Number of buckets (default 5 = quintiles).
    membership_mask:
        Optional bool DataFrame (date × symbol) for PIT universe filtering.
    commission_bps:
        Round-trip cost per unit L1 weight change.

    Returns
    -------
    dict with keys:
        ``result``          : BacktestResult
        ``weights``         : pd.DataFrame (date × symbol)
        ``quintile_returns``: pd.DataFrame (month-end date × quintile label)
                              columns Q1 … Q{n_quantiles}, each is average
                              forward-month return for that bucket.
        ``ls_spread``       : pd.Series  monthly long-minus-short return
    """
    signal = signal.sort_index()
    prices = prices.sort_index()

    common_syms = signal.columns.intersection(prices.columns)
    signal = signal[common_syms]
    prices = prices[common_syms]

    if membership_mask is not None:
        mask = membership_mask.reindex(index=signal.index, columns=common_syms).fillna(False)
    else:
        mask = pd.DataFrame(True, index=signal.index, columns=common_syms)

    # Month-end rebalance dates: last trading day in each calendar month.
    month_ends = signal.groupby(signal.index.to_period("M")).apply(lambda g: g.index[-1])

    asset_returns = prices.pct_change(fill_method=None).reindex(signal.index)

    quintile_rows: list[dict] = []
    ls_dict: dict[pd.Timestamp, float] = {}

    w_df = pd.DataFrame(0.0, index=signal.index, columns=common_syms)

    for period, reb_date in sorted(month_ends.items()):
        sig_row = signal.loc[reb_date]
        eligible = mask.columns[mask.loc[reb_date].to_numpy(dtype=bool)]
        avail = sig_row.reindex(eligible).dropna()
        n = len(avail)
        if n < n_quantiles:
            continue

        # Assign quintile labels 1..n_quantiles (1=lowest signal).
        try:
            q_labels = pd.qcut(avail, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q_labels.isna().all():
            continue

        # Equal-weight long top quintile, short bottom quintile.
        top_syms = avail.index[q_labels == q_labels.max()]
        bot_syms = avail.index[q_labels == q_labels.min()]
        if len(top_syms) == 0 or len(bot_syms) == 0:
            continue

        for sym in top_syms:
            w_df.loc[reb_date, sym] = 1.0 / len(top_syms)
        for sym in bot_syms:
            w_df.loc[reb_date, sym] = -1.0 / len(bot_syms)

        # Compute forward-month returns for each quintile (for diagnostics).
        # Find next month-end date.
        future_dates = signal.index[signal.index > reb_date]
        if len(future_dates) == 0:
            continue
        next_reb_dates = [v for p2, v in sorted(month_ends.items()) if p2 > period]
        if not next_reb_dates:
            continue
        next_reb = next_reb_dates[0]
        window = asset_returns.loc[
            (asset_returns.index > reb_date) & (asset_returns.index <= next_reb)
        ]
        if window.empty:
            continue

        # Compound returns over the window.
        fwd = (1 + window).prod() - 1

        row: dict[str, object] = {"date": reb_date}
        for q_idx in range(int(q_labels.max()) + 1):
            q_syms = avail.index[q_labels == q_idx]
            if len(q_syms):
                row[f"Q{q_idx + 1}"] = float(fwd.reindex(q_syms).mean())
        quintile_rows.append(row)

        # Long-short monthly spread.
        long_ret = float(fwd.reindex(top_syms).mean())
        short_ret = float(fwd.reindex(bot_syms).mean())
        ls_dict[reb_date] = long_ret - short_ret

    # Forward-fill weights between month-ends (non-reb dates → NaN → ffill).
    non_reb_mask_idx = ~w_df.index.isin(month_ends.values)
    w_df.loc[non_reb_mask_idx] = np.nan
    w_df = w_df.ffill().fillna(0.0)

    result = simulate_weights(w_df, prices, commission_bps=commission_bps)

    quintile_returns = (
        pd.DataFrame(quintile_rows).set_index("date")
        if quintile_rows
        else pd.DataFrame(columns=[f"Q{i}" for i in range(1, n_quantiles + 1)])
    )
    ls_spread = pd.Series(ls_dict, name="ls_spread")

    return {
        "result": result,
        "weights": w_df,
        "quintile_returns": quintile_returns,
        "ls_spread": ls_spread,
    }


def xs_information_coefficient(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
) -> dict:
    """Cross-sectional rank IC (Spearman) between signal and forward returns.

    Computes, for each date, the Spearman correlation between the signal
    cross-section and the forward return cross-section. Dates with fewer than
    5 valid pairs are skipped (IC = NaN for that date).

    Parameters
    ----------
    signal:
        Date × symbol DataFrame of signal values (point-in-time, no lag applied
        here — the caller is responsible for any required shift).
    fwd_returns:
        Date × symbol DataFrame of forward returns. Typically
        ``prices.pct_change().shift(-h)`` for horizon h.

    Returns
    -------
    dict with keys:
        ``ic_series``   : pd.Series  daily Spearman IC (NaN on thin dates)
        ``mean_ic``     : float
        ``ic_std``      : float
        ``icir``        : float  mean_ic / ic_std
        ``t_stat``      : float  t = mean_ic / (ic_std / sqrt(n_dates))
        ``pct_positive``: float  fraction of dates with IC > 0
        ``n_dates``     : int    number of dates with valid IC
    """
    # Align on common grid.
    common_dates = signal.index.intersection(fwd_returns.index)
    common_syms = signal.columns.intersection(fwd_returns.columns)
    sig = signal.reindex(index=common_dates, columns=common_syms)
    fwd = fwd_returns.reindex(index=common_dates, columns=common_syms)

    ic_vals: dict[pd.Timestamp, float] = {}

    for date in common_dates:
        s_row = sig.loc[date].to_numpy(dtype="float64")
        f_row = fwd.loc[date].to_numpy(dtype="float64")
        valid = np.isfinite(s_row) & np.isfinite(f_row)
        if valid.sum() < 5:
            continue
        corr, _ = spearmanr(s_row[valid], f_row[valid])
        if np.isfinite(corr):
            ic_vals[date] = float(corr)

    ic_series = pd.Series(ic_vals, name="ic", dtype="float64")

    if len(ic_series) == 0:
        return {
            "ic_series": ic_series,
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "pct_positive": 0.0,
            "n_dates": 0,
        }

    arr = ic_series.to_numpy()
    mean_ic = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    t_stat = mean_ic / (ic_std / np.sqrt(len(arr))) if ic_std > 0 else 0.0

    return {
        "ic_series": ic_series,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "t_stat": t_stat,
        "pct_positive": float((arr > 0).mean()),
        "n_dates": len(arr),
    }
