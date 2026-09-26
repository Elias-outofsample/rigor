# RSI(2) Mean Reversion — S&P 500

- **Slug:** `rsi2_mr`
- **Category:** mean_reversion

## Thesis
Short-horizon mean reversion (Connors): deeply oversold large-caps in an uptrend
tend to bounce. Trade only *with* the long-term trend (price > SMA-200), enter
on an RSI(2) washout, and exit fast on the first sign of reversion.

## Signal (daily)
On point-in-time S&P 500 members, a name is **eligible** when:
`RSI(2) < 10` AND `close > SMA(200)` AND `raw close > $1`. Of the eligible names,
hold the **top 5 by NATR(14)** (most volatile). **Exit** a position when `close >
the prior day's high`, or after **20 bars**. Long-only, equal weight across the
open positions.

## Universe & data
Survivorship-free S&P 500 PIT members (same data as Momentum_S7). EODHD
total-return-adjusted prices for RSI/SMA/NATR; raw close only for the $1 gate.

## Parameters (`config.json` → `extra`)
`rsi_period=2`, `rsi_threshold=10`, `sma_period=200`, `natr_period=14`,
`max_positions=5`, `min_price=1.0`, `time_stop_bars=20`. Cost 5 bps.

## Deviations from the original
Start **2010** (original ran from 1998) to bound the universe to the cached set;
EODHD prices; **close-to-close** target-weight execution (original used next-open
with per-share commission + slippage); single `commission_bps`.

## Run
```
python -m rigor run strategies/mean_reversion/rsi2_mr --as-of 2026-06-01 --offline
```
