"""Deterministic backtest engine: target weights -> portfolio returns.

This is the *standardised accounting core*. Strategies decide WHAT to hold
(target weights per date); the engine decides how that becomes a return stream —
identically for everyone. That uniformity is the point: two strategies differing
only in signal logic get exactly the same cost/lag treatment, so their report
cards are comparable.

Conventions (deliberate, documented, and the same on every machine):
  * No look-ahead: the weight in effect over (t-1, t] is the weight decided at
    t-1. Weights are shifted one bar before being applied to returns.
  * Turnover cost: ``commission_bps`` charged on L1 weight change each rebalance,
    i.e. cost_t = bps/1e4 * sum_i |w_{i,t} - w_{i,t-1}|.
  * Periodic rebalance to target (intra-period drift not modelled) — a standard
    v1 simplification; revisit if a strategy needs buy-and-hold drift accounting.

Pure numpy/pandas. No RNG, no threads, no Numba — bit-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as _metrics


@dataclass
class Trade:
    """One completed trade, recorded by an event-driven strategy's own loop.

    Use this when a strategy can't express its positions as a daily weights
    matrix (intraday limit/stop fills, state machines, pyramiding) — it knows
    each trade's entry/exit directly, so it records them into a :class:`TradeLog`
    and the position ledger turns them into per-trade MAE/MFE.

    Fields:
      symbol/side  — instrument and "long"/"short".
      entry_date/exit_date — timestamps of the fills.
      entry_price/exit_price — the ACTUAL fill prices (limit/stop/open — whatever
        fired), NOT the prior close. This is what makes intraday MAE/MFE correct.
      bar_prices   — the price marks WHILE HELD (e.g. daily closes, or intraday
        bars), as a Series from entry to exit. MAE/MFE are measured along this
        path relative to ``entry_price``.
      size         — optional position size/weight (informational; ledger weight
        columns are left NaN for trade-log strategies).
      system       — optional tag (e.g. Turtle "S1"/"S2"); one Trade per unit
        when pyramiding.
    """
    symbol: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    bar_prices: pd.Series = field(repr=False, default=None)
    size: float = 1.0
    system: str = ""


@dataclass
class TradeLog:
    """Accumulator a strategy's loop appends completed :class:`Trade` objects to.

    A strategy opts in by adding ``trade_log=None`` to its ``run_backtest``
    signature and calling ``trade_log.record(...)`` at each exit; ``backtest()``
    supplies the instance and attaches it to the result. Strategies that don't
    accept the argument are completely unaffected.
    """
    trades: list[Trade] = field(default_factory=list)

    def record(
        self,
        *,
        symbol: str,
        side: str,
        entry_date,
        exit_date,
        entry_price: float,
        exit_price: float,
        bar_prices=None,
        size: float = 1.0,
        system: str = "",
    ) -> None:
        """Append one completed trade. For pyramiding, call once per unit."""
        self.trades.append(Trade(
            symbol=symbol, side=side, entry_date=entry_date, exit_date=exit_date,
            entry_price=float(entry_price), exit_price=float(exit_price),
            bar_prices=bar_prices, size=float(size), system=system,
        ))

    def __len__(self) -> int:
        return len(self.trades)

    @property
    def empty(self) -> bool:
        return not self.trades


@dataclass
class BacktestResult:
    returns: pd.Series          # net daily portfolio returns
    equity: pd.Series           # cumulative growth (starts ~1.0)
    metrics: dict               # canonical compute_core_metrics output
    gross_returns: pd.Series = field(repr=False, default=None)
    costs: pd.Series = field(repr=False, default=None)
    weights: pd.DataFrame = field(repr=False, default=None)
    # Per-asset return panel (date×symbol) — kept so the position ledger can
    # compute per-trade MAE/MFE without re-fetching prices. None for
    # returns-only strategies.
    asset_returns: pd.DataFrame = field(repr=False, default=None)
    # Trades recorded by an event-driven strategy (the alternative to weights for
    # building the position ledger). None for weights-based / plain returns.
    trade_log: TradeLog | None = field(repr=False, default=None)

    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestResult(n={m.get('n_obs')}, sharpe={m.get('sharpe'):.2f}, "
            f"cagr={m.get('cagr'):.2%}, maxdd={m.get('max_drawdown'):.2%})"
        )


def simulate_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float = 10.0,
    periods_per_year: int = _metrics.TRADING_DAYS,
) -> BacktestResult:
    """Simulate a target-weight portfolio over an adjusted-price panel.

    ``weights`` and ``prices`` are date-indexed, symbol-columned frames. They are
    aligned on their common dates/symbols; missing weights are treated as 0.
    """
    prices = prices.sort_index()
    asset_returns = prices.pct_change(fill_method=None)

    # Align weights onto the return calendar/symbols; absent => no position.
    w = weights.reindex(index=asset_returns.index, columns=asset_returns.columns)
    w = w.ffill().fillna(0.0)

    # No look-ahead: hold yesterday's target through today's return.
    w_held = w.shift(1).fillna(0.0)
    gross = (w_held * asset_returns.fillna(0.0)).sum(axis=1)

    # Turnover cost on the L1 change in target weights.
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (commission_bps / 1e4)

    net = (gross - costs).astype("float64")
    net.name = "returns"
    # Drop the first row (no prior weight / NaN return).
    net = net.iloc[1:]
    gross = gross.iloc[1:]
    costs = costs.iloc[1:]

    equity = pd.Series(
        _metrics.compute_equity_curve(net.to_numpy()), index=net.index, name="equity"
    )
    ppy = _metrics.periods_per_year_of(net.index, default=periods_per_year)
    return BacktestResult(
        returns=net,
        equity=equity,
        metrics=_metrics.compute_core_metrics(net.to_numpy(), periods_per_year=ppy),
        gross_returns=gross,
        costs=costs,
        weights=w,
        asset_returns=asset_returns,
    )


def result_from_returns(
    returns, index: pd.DatetimeIndex | None = None,
    periods_per_year: int = _metrics.TRADING_DAYS,
    trade_log: TradeLog | None = None,
    weights: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    """Wrap a precomputed returns array/series into a BacktestResult + metrics.

    For strategies that produce returns directly (the returns-first contract)
    rather than going through ``simulate_weights``.

    A strategy that computes its OWN returns but also has a date×symbol target
    ``weights`` matrix (and ``asset_returns`` panel) can pass them here to get a
    position ledger with MAE/MFE *without changing its returns* — the engine only
    reads them to reconstruct trades. Alternatively, event-driven strategies pass
    a ``trade_log`` of recorded trades.
    """
    if isinstance(returns, pd.Series):
        ser = returns.astype("float64")
    else:
        arr = np.asarray(returns, dtype="float64")
        ser = pd.Series(arr, index=index if index is not None else pd.RangeIndex(len(arr)))
    ser.name = "returns"
    ppy = _metrics.periods_per_year_of(ser.index, default=periods_per_year)
    equity = pd.Series(
        _metrics.compute_equity_curve(ser.to_numpy()), index=ser.index, name="equity"
    )
    return BacktestResult(
        returns=ser,
        equity=equity,
        metrics=_metrics.compute_core_metrics(ser.to_numpy(), periods_per_year=ppy),
        trade_log=trade_log,
        weights=weights,
        asset_returns=asset_returns,
    )
