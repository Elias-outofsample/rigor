"""Per-position spell ledger: extract and summarise position history from backtest weights.

A "position spell" is a maximal run of consecutive dates where the absolute weight
for a symbol exceeds 1e-9. Each spell becomes one row in the ledger.  Aggregated
stats are available via ``summarize_ledger``.  The canonical disk output is written
by ``write_position_ledger``, which places the file alongside the other strategy
artifacts under ``artifacts/<slug>_positions.csv``.

**MAE / MFE definitions** (computed only when *asset_returns* is supplied):

``max_adverse_excursion`` (MAE)
    The worst unrealised loss reached from entry to exit.  At each bar within the
    spell the cumulative position P&L is ``sum(weight[t-1] * return[t])`` from the
    first bar to bar *t*.  MAE = ``min(cumulative_pnl)`` across all bars.  By
    convention MAE ≤ 0; a value of 0 means the position never moved against the
    trader.

``max_favorable_excursion`` (MFE)
    The best unrealised gain reached before exit.  MFE = ``max(cumulative_pnl)``
    across all bars.  By convention MFE ≥ 0; a value of 0 means the position never
    moved in the trader's favour.

Both metrics use the same *shifted-weight* convention as ``return_contribution``
(previous bar's weight applied to current bar's return) to avoid look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import layout

if TYPE_CHECKING:
    pass

# Column order used everywhere — keeps read-back consistent.
_LEDGER_COLS = [
    "symbol",
    "side",
    "entry_date",
    "exit_date",
    "holding_days",
    "avg_weight",
    "max_weight",
    "n_rebalances",
    "return_contribution",
    "max_adverse_excursion",
    "max_favorable_excursion",
]

_ZERO_THRESH = 1e-9


def _empty_ledger() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical column schema."""
    return pd.DataFrame(columns=_LEDGER_COLS)


def _extract_weights(result) -> pd.DataFrame | None:
    """Pull the weights frame from a BacktestResult or a raw DataFrame."""
    if isinstance(result, pd.DataFrame):
        return result
    weights = getattr(result, "weights", None)
    if weights is None or (isinstance(weights, pd.DataFrame) and weights.empty):
        return None
    return weights


def build_position_ledger(
    result,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-position-spell ledger from backtest weights.

    Accepts a ``BacktestResult`` (duck-typed via its ``.weights`` attribute) or a
    raw weights ``DataFrame``.  Returns a ``DataFrame`` with columns::

        symbol, side, entry_date, exit_date, holding_days, avg_weight,
        max_weight, n_rebalances, return_contribution,
        max_adverse_excursion, max_favorable_excursion

    ``max_adverse_excursion`` and ``max_favorable_excursion`` are NaN when
    *asset_returns* is not supplied; see module docstring for definitions.

    Sorted by ``entry_date``, ``symbol``.  Returns an empty ``DataFrame`` with the
    correct columns if there are no non-zero weights.

    Event-driven strategies that have no weights matrix instead record trades into
    a ``TradeLog`` (``result.trade_log``); when present, the ledger is built from
    those recorded trades (see ``_build_from_trade_log``).
    """
    weights = _extract_weights(result)
    if weights is None:
        # No weights frame — build from a recorded TradeLog if the strategy kept one.
        trade_log = getattr(result, "trade_log", None)
        if trade_log is not None and getattr(trade_log, "trades", None):
            return _build_from_trade_log(trade_log)
        return _empty_ledger()

    # Fall back to the per-asset return panel the engine stashed on the result
    # (simulate_weights keeps it), so MAE/MFE populate without re-fetching prices.
    if asset_returns is None and not isinstance(result, pd.DataFrame):
        asset_returns = getattr(result, "asset_returns", None)

    # Align asset_returns to the same index/columns as weights (if provided).
    if asset_returns is not None:
        asset_returns = asset_returns.reindex(
            index=weights.index, columns=weights.columns
        ).fillna(0.0)

    rows: list[dict] = []

    for symbol in weights.columns:
        w_col = weights[symbol]
        in_spell = False
        spell_start: int = 0

        for i, (_date, w) in enumerate(w_col.items()):
            active = abs(w) > _ZERO_THRESH
            if active and not in_spell:
                # Start of a new spell.
                in_spell = True
                spell_start = i
            elif not active and in_spell:
                # End of spell — record it.
                rows.append(
                    _build_spell_row(
                        symbol, w_col, spell_start, i - 1, asset_returns
                    )
                )
                in_spell = False

        if in_spell:
            # Spell ran to the last bar.
            rows.append(
                _build_spell_row(
                    symbol, w_col, spell_start, len(w_col) - 1, asset_returns
                )
            )

    if not rows:
        return _empty_ledger()

    return (
        pd.DataFrame(rows, columns=_LEDGER_COLS)
        .sort_values(["entry_date", "symbol"])
        .reset_index(drop=True)
    )


def _build_spell_row(
    symbol: str,
    w_col: pd.Series,
    start_idx: int,
    end_idx: int,
    asset_returns: pd.DataFrame | None,
) -> dict:
    """Compute one spell row from the slice ``w_col.iloc[start_idx:end_idx+1]``."""
    spell_w = w_col.iloc[start_idx : end_idx + 1]
    idx = w_col.index

    avg_w = float(spell_w.mean())
    max_w = float(spell_w.abs().max())
    holding_days = end_idx - start_idx + 1

    # Count weight changes within the spell (from the 2nd bar onward).
    if holding_days > 1:
        diffs = spell_w.diff().iloc[1:]
        n_rebalances = int((diffs.abs() > _ZERO_THRESH).sum())
    else:
        n_rebalances = 0

    # return_contribution, MAE, MFE: require asset_returns.
    rc = float("nan")
    mae = float("nan")
    mfe = float("nan")
    if asset_returns is not None and symbol in asset_returns.columns:
        # Shift weights one bar (no look-ahead): apply previous bar's weight
        # to current bar's return.  The very first bar of the full series has no
        # previous weight, so shift(1) naturally produces NaN there — we drop it.
        full_w = w_col.shift(1)
        full_ret = asset_returns[symbol]
        bar_pnl = (full_w * full_ret).iloc[start_idx : end_idx + 1]
        rc = float(bar_pnl.sum(skipna=True))

        # Cumulative P&L within the spell (NaN bars treated as 0 contribution).
        cum_pnl = bar_pnl.fillna(0.0).cumsum()

        # MAE: worst (most negative) cumulative P&L reached; capped at 0.
        # MFE: best (most positive) cumulative P&L reached; floored at 0.
        mae = float(min(cum_pnl.min(), 0.0))
        mfe = float(max(cum_pnl.max(), 0.0))

    return {
        "symbol": symbol,
        "side": "long" if avg_w > 0 else "short",
        "entry_date": idx[start_idx],
        "exit_date": idx[end_idx],
        "holding_days": holding_days,
        "avg_weight": avg_w,
        "max_weight": max_w,
        "n_rebalances": n_rebalances,
        "return_contribution": rc,
        "max_adverse_excursion": mae,
        "max_favorable_excursion": mfe,
    }


def _build_from_trade_log(trade_log) -> pd.DataFrame:
    """Build the ledger from recorded trades (event-driven strategies).

    Each :class:`rigor.engine.Trade` becomes one row. MAE/MFE are measured along the
    trade's ``bar_prices`` path relative to the ACTUAL ``entry_price`` (so an
    intraday limit/stop fill, which differs from the prior close, is handled
    correctly). ``avg_weight``/``max_weight`` are NaN (no portfolio weight matrix).
    """
    rows = [_spell_row_from_trade(t) for t in trade_log.trades]
    if not rows:
        return _empty_ledger()
    return (
        pd.DataFrame(rows, columns=_LEDGER_COLS)
        .sort_values(["entry_date", "symbol"])
        .reset_index(drop=True)
    )


def _spell_row_from_trade(t) -> dict:
    """One ledger row from a recorded trade; MAE/MFE from its held-bar price path."""
    sign = 1.0 if str(t.side).lower() == "long" else -1.0
    entry_p = float(t.entry_price)

    marks = [entry_p]
    if t.bar_prices is not None and len(t.bar_prices) > 0:
        marks.extend(float(p) for p in pd.Series(t.bar_prices).to_numpy())
    marks.append(float(t.exit_price))

    if entry_p != 0.0:
        rel = sign * (np.asarray(marks, dtype="float64") / entry_p - 1.0)
        mae = float(min(rel.min(), 0.0))
        mfe = float(max(rel.max(), 0.0))
        rc = float(sign * (float(t.exit_price) / entry_p - 1.0))
    else:
        mae = mfe = rc = float("nan")

    holding = len(t.bar_prices) if (t.bar_prices is not None and len(t.bar_prices) > 0) else 1
    return {
        "symbol": t.symbol,
        "side": "long" if sign > 0 else "short",
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "holding_days": int(holding),
        "avg_weight": float("nan"),
        "max_weight": float("nan"),
        "n_rebalances": 0,
        "return_contribution": rc,
        "max_adverse_excursion": mae,
        "max_favorable_excursion": mfe,
    }


def summarize_ledger(ledger: pd.DataFrame) -> dict:
    """Aggregate statistics from a position ledger.

    Returns a ``dict`` with keys:

    * ``n_trades`` — total number of position spells
    * ``n_symbols`` — distinct symbols traded
    * ``pct_long`` — fraction of spells that are long-side
    * ``avg_holding_days`` — mean spell length
    * ``median_holding_days`` — median spell length
    * ``max_holding_days`` — longest spell (bars)
    * ``turnover_proxy`` — mean absolute weight across all spells

    When ``return_contribution`` is present and non-NaN:

    * ``win_rate`` — fraction of spells with positive contribution
    * ``avg_win`` — mean contribution of winning spells
    * ``avg_loss`` — mean contribution of losing spells (will be <= 0)
    * ``profit_factor`` — abs(sum_wins) / abs(sum_losses); inf if no losses

    When ``max_adverse_excursion`` / ``max_favorable_excursion`` are present
    and non-NaN (requires *asset_returns* passed to ``build_position_ledger``):

    * ``mae_mean_losing`` — mean MAE across losing spells (≤ 0)
    * ``mae_median_losing`` — median MAE across losing spells
    * ``mae_worst`` — single worst (most negative) MAE across all losing spells
    * ``mae_mean_winning`` — mean MAE across winning spells
    * ``mae_median_winning`` — median MAE across winning spells
    * ``mfe_mean_winning`` — mean MFE across winning spells (≥ 0)
    * ``mfe_median_winning`` — median MFE across winning spells
    * ``mfe_mean_losing`` — mean MFE across losing spells
    * ``mfe_median_losing`` — median MFE across losing spells
    """
    if ledger.empty:
        return {
            "n_trades": 0,
            "n_symbols": 0,
            "pct_long": float("nan"),
            "avg_holding_days": float("nan"),
            "median_holding_days": float("nan"),
            "max_holding_days": float("nan"),
            "turnover_proxy": float("nan"),
        }

    n_trades = len(ledger)
    n_symbols = ledger["symbol"].nunique()
    pct_long = float((ledger["side"] == "long").mean()) if n_trades else float("nan")

    hd = ledger["holding_days"]
    avg_hd = float(hd.mean())
    med_hd = float(hd.median())
    max_hd = int(hd.max())
    turnover_proxy = float(ledger["avg_weight"].abs().mean())

    out: dict = {
        "n_trades": n_trades,
        "n_symbols": n_symbols,
        "pct_long": pct_long,
        "avg_holding_days": avg_hd,
        "median_holding_days": med_hd,
        "max_holding_days": max_hd,
        "turnover_proxy": turnover_proxy,
    }

    # Return-contribution stats — only when the column is present and usable.
    rc = ledger.get("return_contribution")
    if rc is not None:
        valid = rc.dropna()
        if not valid.empty:
            wins = valid[valid > 0]
            losses = valid[valid <= 0]
            win_rate = float(len(wins) / len(valid))
            avg_win = float(wins.mean()) if not wins.empty else float("nan")
            avg_loss = float(losses.mean()) if not losses.empty else float("nan")
            sum_wins = float(wins.sum()) if not wins.empty else 0.0
            sum_losses = float(losses.sum()) if not losses.empty else 0.0
            profit_factor = float("inf") if sum_losses == 0.0 else abs(sum_wins) / abs(sum_losses)
            out.update(
                {
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                }
            )

    # MAE / MFE stats — only when those columns are present and non-NaN.
    # Uses return_contribution to split spells into winners/losers (rc > 0 = win).
    mae_col = ledger.get("max_adverse_excursion")
    mfe_col = ledger.get("max_favorable_excursion")
    if mae_col is not None and mfe_col is not None and rc is not None:
        # Build a mask aligned on the same valid (non-NaN) rows across all three.
        excursion_mask = mae_col.notna() & mfe_col.notna() & rc.notna()
        if excursion_mask.any():
            valid_rc = rc[excursion_mask]
            valid_mae = mae_col[excursion_mask]
            valid_mfe = mfe_col[excursion_mask]

            win_mask = valid_rc > 0
            loss_mask = ~win_mask

            mae_losing = valid_mae[loss_mask]
            mae_winning = valid_mae[win_mask]
            mfe_winning = valid_mfe[win_mask]
            mfe_losing = valid_mfe[loss_mask]

            excursion_stats: dict = {}

            if not mae_losing.empty:
                excursion_stats["mae_mean_losing"] = float(mae_losing.mean())
                excursion_stats["mae_median_losing"] = float(mae_losing.median())
                excursion_stats["mae_worst"] = float(mae_losing.min())
            if not mae_winning.empty:
                excursion_stats["mae_mean_winning"] = float(mae_winning.mean())
                excursion_stats["mae_median_winning"] = float(mae_winning.median())
            if not mfe_winning.empty:
                excursion_stats["mfe_mean_winning"] = float(mfe_winning.mean())
                excursion_stats["mfe_median_winning"] = float(mfe_winning.median())
            if not mfe_losing.empty:
                excursion_stats["mfe_mean_losing"] = float(mfe_losing.mean())
                excursion_stats["mfe_median_losing"] = float(mfe_losing.median())

            out.update(excursion_stats)

    return out


def write_position_ledger(
    strategy_dir: Path | str,
    slug: str,
    result,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> Path | None:
    """Build a position ledger and write it to ``artifacts/<slug>_positions.csv``.

    Uses ``layout.artifacts_dir(strategy_dir)`` to locate the target directory.
    Returns the ``Path`` of the written file, or ``None`` if the result carries no
    weights (returns-only strategy) — in that case no file is written.
    """
    ledger = build_position_ledger(result, asset_returns=asset_returns)
    if ledger.empty:
        return None

    art_dir = layout.artifacts_dir(Path(strategy_dir))
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / f"{slug}_positions.csv"
    ledger.to_csv(out_path, index=False)
    return out_path


def main(argv=None) -> int:  # noqa: ARG001
    """CLI entry point — no-op placeholder.

    Use ``rigor run <strategy_dir>`` to produce position ledgers as part of a full
    strategy run.  A dedicated ``rigor positions`` sub-command may be added in a
    future release.
    """
    from ..console import force_utf8_stdio

    force_utf8_stdio()
    print("rigor.project.trades: use 'rigor run <strategy_dir>' to generate position ledgers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
