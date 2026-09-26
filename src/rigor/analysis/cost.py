"""Realistic cost & capacity modelling (#4).

Turns a gross backtest into a net one and asks "how much can this hold?":

  * ``estimate_spread_bps`` / ``estimate_market_impact_bps`` — bid-ask + Almgren-Chriss
    square-root impact.
  * ``compute_realistic_costs`` — full cost breakdown (spread + impact + commission +
    borrow + margin financing), and net Sharpe when given a return series.
  * ``compute_financing_drag`` / ``compute_borrow_cost`` / ``recall_risk`` — short
    borrow + margin interest + HTB recall hazard.
  * ``capacity_ceiling`` / ``compute_capital_scaling`` / ``estimate_aum_capacity`` /
    ``compute_strategy_capacity`` / ``aggregate_portfolio_capacity`` — participation-limit
    and impact-threshold AUM capacity.

Pure numpy/pandas. (The original Norgate-watchlist real-ADV capacity helpers are
intentionally omitted — this framework is single-source EODHD, not Norgate.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as _m

__all__ = [
    "estimate_spread_bps", "estimate_market_impact_bps", "compute_realistic_costs",
    "compute_financing_drag", "capacity_ceiling", "compute_cost_sensitivity_table",
    "BORROW_TIERS", "compute_borrow_cost", "recall_risk", "compute_capital_scaling",
    "estimate_aum_capacity", "compute_strategy_capacity", "aggregate_portfolio_capacity",
]

TRADING_DAYS_PER_YEAR = 252
MARGIN_RATE_ANNUAL = 0.0533            # broker-typical margin interest (~5.3%/yr)
BORROW_RATE_LIQUID_BPS = 50.0          # liquid large-cap stock borrow
BORROW_RATE_SMALL_CAP_BPS = 200.0
BORROW_RATE_HARD_TO_BORROW_BPS = 500.0

# Annualised borrow rates by hard-to-borrow tier (IB / S3 Partners typical).
BORROW_TIERS: dict[str, float] = {
    "easy": 0.005, "general": 0.015, "hard": 0.075, "specials": 0.20,
}

# Conservative median ADV per symbol (USD) per universe — participation-limit proxy.
_UNIV_ADV_PER_SLOT: dict[str, float] = {
    "equity_us": 100_000_000.0, "equity_global": 60_000_000.0,
    "commodities": 40_000_000.0, "fx": 200_000_000.0, "crypto": 20_000_000.0,
}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and np.isfinite(b) else default


def estimate_spread_bps(adv: float, price: float = 50.0) -> float:
    """Estimate the bid-ask spread (bps) from dollar ADV — wider for illiquid names."""
    if adv <= 0:
        return 50.0
    log_adv = np.log10(max(adv * price, 1))
    return float(min(max(1.0, 50.0 - 5.0 * log_adv), 500.0))


def estimate_market_impact_bps(trade_size: float, adv: float, daily_vol: float = 0.02) -> float:
    """Almgren-Chriss (2001) square-root impact: vol·√(size/ADV)·1e4 bps, capped at 500."""
    if adv <= 0 or trade_size <= 0:
        return 0.0
    return float(min(daily_vol * np.sqrt(trade_size / adv) * 10000, 500.0))


def compute_realistic_costs(trade_value: float = 0.0, adv: float = 0.0,
                            spread_bps: float | None = None, commission_bps: float = 1.0,
                            returns=None, short_fraction: float = 0.0,
                            borrow_rate_bps: float = BORROW_RATE_LIQUID_BPS,
                            leverage: float = 1.0,
                            annual_periods: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Full cost breakdown: per-trade bps + annual borrow/margin drag + net Sharpe.

    Two modes combine: per-trade (trade_value + adv → spread/impact bps) and
    returns-series (annual drag from borrow + margin, applied to give net Sharpe).
    """
    if spread_bps is None:
        spread_bps = estimate_spread_bps(adv) if adv > 0 else 0.0
    impact_bps = estimate_market_impact_bps(trade_value, adv) if adv > 0 else 0.0
    total_bps = spread_bps / 2 + impact_bps + commission_bps
    borrow_drag = float(short_fraction) * (float(borrow_rate_bps) / 10_000.0)
    margin_drag = max(0.0, float(leverage) - 1.0) * MARGIN_RATE_ANNUAL
    total_drag = borrow_drag + margin_drag

    result: dict = {
        "spread_bps": float(spread_bps), "half_spread_bps": float(spread_bps / 2),
        "market_impact_bps": float(impact_bps), "commission_bps": float(commission_bps),
        "total_one_way_bps": float(total_bps), "total_round_trip_bps": float(total_bps * 2),
        "borrow_drag": float(borrow_drag), "financing_drag": float(margin_drag),
        "total_drag": float(total_drag), "short_fraction": float(short_fraction),
        "borrow_rate_bps": float(borrow_rate_bps), "leverage": float(leverage),
    }
    if returns is not None:
        r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
        r = r.astype(np.float64)
        r = r[np.isfinite(r)]
        if len(r) > 1:
            gross = _m.compute_sharpe(r, annual_periods)
            net = _m.compute_sharpe(r - total_drag / annual_periods, annual_periods)
            result.update({"gross_sharpe": float(gross), "net_sharpe": float(net),
                           "vol_annual": float(np.std(r, ddof=1) * np.sqrt(annual_periods)),
                           "sharpe_drag": float(gross - net)})
    return result


def compute_financing_drag(short_fraction: float = 0.0,
                           borrow_rate_bps: float = BORROW_RATE_LIQUID_BPS,
                           leverage: float = 1.0) -> dict:
    """Annual financing drag = short borrow + margin interest (drag-only convenience)."""
    borrow_drag = short_fraction * borrow_rate_bps / 10_000.0
    margin_drag = max(0.0, leverage - 1.0) * MARGIN_RATE_ANNUAL
    return {"borrow_drag": float(borrow_drag), "financing_drag": float(margin_drag),
            "total_drag": float(borrow_drag + margin_drag)}


def capacity_ceiling(avg_trade_value: float, adv_median: float,
                     max_participation: float = 0.01, max_impact_bps: float = 20.0) -> dict:
    """Capacity ceiling from a participation cap, scaled down if impact exceeds the limit."""
    max_trade = adv_median * max_participation
    if avg_trade_value <= 0:
        return {"capacity_usd": 0, "max_positions": 0}
    n_positions = max(1, int(max_trade / avg_trade_value))
    impact = estimate_market_impact_bps(max_trade, adv_median)
    if impact > max_impact_bps and max_impact_bps > 0:
        n_positions = max(1, int(n_positions * (max_impact_bps / impact) ** 2))
    return {"capacity_usd": float(n_positions * avg_trade_value), "max_positions": n_positions,
            "impact_at_capacity_bps": float(impact)}


def compute_cost_sensitivity_table(sharpe: float, annual_return: float, n_trades: int,
                                   avg_holding_days: float,
                                   cost_levels_bps: list[float] | None = None,
                                   annual_periods: int = TRADING_DAYS_PER_YEAR) -> list[dict]:
    """Sharpe vs cost-assumption table (round-trip drag at the strategy's turnover)."""
    if cost_levels_bps is None:
        cost_levels_bps = [0, 2, 5, 10, 15, 20, 30, 50]
    annual_turnover = annual_periods / max(avg_holding_days, 1)
    vol_annual = annual_return / sharpe if sharpe > 0 and annual_return > 0 else None
    results = []
    for bps in cost_levels_bps:
        cost_drag = bps / 10000 * annual_turnover * 2
        net_return = annual_return - cost_drag
        net_sharpe: float | None
        red: float | None
        if vol_annual is not None and vol_annual > 0:
            net_sharpe = net_return / vol_annual
            red = float(1 - net_sharpe / sharpe) * 100 if sharpe != 0 else None
        else:
            net_sharpe = red = None
        results.append({"cost_bps": bps, "cost_drag_annual": float(cost_drag),
                        "net_return": float(net_return), "net_sharpe": net_sharpe,
                        "sharpe_reduction_pct": red})
    return results


def compute_borrow_cost(positions: pd.DataFrame, htb_tier_map: dict[str, str],
                        dates: pd.DatetimeIndex | None = None) -> dict:
    """Tier-specific annual borrow cost for short positions (rows=dates, cols=tickers)."""
    if dates is not None:
        positions = positions.reindex(index=dates)
    tickers = list(positions.columns)
    short_notional = positions.clip(upper=0.0).abs()
    per_ticker: dict[str, float] = {}
    daily_cost = pd.Series(0.0, index=positions.index, dtype=np.float64)
    for ticker in tickers:
        rate = BORROW_TIERS.get(htb_tier_map.get(ticker, "general"), BORROW_TIERS["general"])
        ticker_daily = short_notional[ticker] * (rate / 252.0)
        daily_cost += ticker_daily
        per_ticker[ticker] = float(ticker_daily.sum() * 252.0 / max(len(positions), 1))
    total_annual = float(daily_cost.sum() * 252.0 / max(len(positions), 1))
    total_short = float(short_notional.values.sum())
    weighted_rate = 0.0
    if total_short > 0:
        for ticker in tickers:
            rate = BORROW_TIERS.get(htb_tier_map.get(ticker, "general"), BORROW_TIERS["general"])
            weighted_rate += (float(short_notional[ticker].sum()) / total_short) * rate
    return {"total_annual_cost": total_annual, "per_ticker": per_ticker,
            "daily_cost_series": daily_cost, "weighted_rate": float(weighted_rate),
            "total_short_notional": total_short}


def recall_risk(short_holding_days: int, htb_tier: str) -> float:
    """P(share recall before exit) under an exponential hazard calibrated by HTB tier."""
    hazard = {"easy": 0.001, "general": 0.003, "hard": 0.015, "specials": 0.05}.get(htb_tier, 0.003)
    prob = 1.0 - float(np.exp(-hazard * max(int(short_holding_days), 0)))
    return min(max(prob, 0.0), 1.0)


def compute_capital_scaling(returns, trades_df=None, adv_median: float = 1e7,
                            capital_levels: list[float] | None = None,
                            annual_periods: int | None = None) -> dict:
    """Model Sharpe degradation as capital scales via Almgren-Chriss temp+perm impact."""
    if capital_levels is None:
        capital_levels = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9]
    if annual_periods is None:
        is_dt = isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex)
        annual_periods = _m.periods_per_year_of(returns.index) if is_dt else 252
    ppy = max(int(annual_periods), 1)
    base_sharpe = _m.compute_sharpe(returns)
    r = (returns.to_numpy() if isinstance(returns, pd.Series)
         else np.asarray(returns, dtype=np.float64))
    per_period_vol = float(np.std(r.astype(np.float64), ddof=1))
    temp_coef, perm_coef = 0.314, 0.10
    results = []
    for cap in capital_levels:
        participation = min(cap / adv_median if adv_median > 0 else 1.0, 1.0)
        temp_bps = per_period_vol * temp_coef * np.sqrt(participation) * 10000
        perm_bps = perm_coef * participation * 10000
        total_bps = temp_bps + perm_bps
        impact_drag = total_bps / 10000 * ppy
        adj = max(0.0, base_sharpe - impact_drag / max(per_period_vol * np.sqrt(ppy), 0.01))
        results.append({"capital": cap, "sharpe": float(adj), "impact_bps": float(total_bps),
                        "temp_impact_bps": float(temp_bps), "perm_impact_bps": float(perm_bps)})
    opt = int(np.argmin([abs(x["sharpe"] - base_sharpe * 0.8) for x in results]))
    return {"scaling_curve": results, "base_sharpe": float(base_sharpe),
            "optimal_capital": float(results[opt]["capital"])}


def estimate_aum_capacity(returns, adv_per_symbol: float = 1e7, n_positions: int = 10,
                          daily_vol: float = 0.02, target_market_impact_bps: float = 10.0,
                          aum_levels: list[float] | None = None) -> dict:
    """AUM capacity by solving for the AUM where per-trade impact crosses 10/20/50 bps."""
    r = (returns.to_numpy() if isinstance(returns, pd.Series)
         else np.asarray(returns, dtype=np.float64))
    r = r.astype(np.float64)
    r = r[np.isfinite(r)]
    base_sharpe = float(_m.compute_sharpe(r)) if len(r) > 1 else 0.0
    per_period_vol = float(np.std(r, ddof=1)) if len(r) > 1 else daily_vol
    if aum_levels is None:
        aum_levels = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9]

    def _impact(aum: float) -> float:
        return estimate_market_impact_bps(aum / max(n_positions, 1), adv_per_symbol, daily_vol)

    def _sharpe_at(aum: float, ann: int = 252) -> float:
        drag = _impact(aum) / 10_000.0 * ann
        return float(max(base_sharpe - drag / max(per_period_vol * np.sqrt(ann), 0.001), 0.0))

    sharpe_curve = [{"aum": a, "sharpe": _sharpe_at(a), "impact_bps": float(_impact(a))}
                    for a in aum_levels]

    def _find(threshold: float) -> float:
        lo, hi = 1e3, 1e12
        for _ in range(60):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if _impact(mid) < threshold else (lo, mid)
        return float((lo + hi) / 2.0)

    return {"capacity_low": _find(10.0), "capacity_central": _find(20.0),
            "capacity_high": _find(50.0), "capacity_low_impact_bps": 10.0,
            "capacity_central_impact_bps": 20.0, "capacity_high_impact_bps": 50.0,
            "sharpe_vs_aum": sharpe_curve, "base_sharpe": base_sharpe,
            "adv_per_symbol": adv_per_symbol, "n_positions": n_positions}


def compute_strategy_capacity(n_position_slots: int, universe: str = "equity_us",
                              max_participation_pct: float = 0.10,
                              adv_per_slot: float | None = None) -> dict:
    """Participation-limit capacity: slots × median-ADV$ × max-participation."""
    slots = max(int(n_position_slots), 1)
    adv = float(adv_per_slot) if adv_per_slot is not None else float(
        _UNIV_ADV_PER_SLOT.get(universe, _UNIV_ADV_PER_SLOT["equity_us"]))
    pct = float(max_participation_pct)
    return {"capacity_usd": float(slots) * adv * pct, "n_position_slots": slots,
            "adv_per_slot": adv, "max_participation_pct": pct, "universe": universe,
            "formula": f"{slots} slots x ${adv / 1e6:.0f}M ADV x {pct * 100:.0f}% participation"}


def aggregate_portfolio_capacity(sub_capacities: list[float]) -> float:
    """Aggregate per-strategy capacities (sum for non-overlapping symbols; >= max individual)."""
    valid = [float(c) for c in sub_capacities if c > 0]
    if not valid:
        return 0.0
    return max(sum(valid), max(valid))
