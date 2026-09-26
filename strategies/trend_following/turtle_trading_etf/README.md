# Turtle Trading — Multi-ETF (Main)

- **Slug:** `turtle_trading_etf` · **Version:** `v1`
- **Category:** trend_following
- **Baseline:** atomic: the `v0` baseline is this production version (see *Baseline* below)
- **Verdict:** **REJECT** — an honest clean-data result; see "Faithfulness & the data-gap finding".

## Thesis
The classic Dennis/Eckhardt Turtle trend-following system (TurtleTrading_S16,
"Multi-Instrument") applied across five uncorrelated ETF proxies — SPY, QQQ (equity),
GLD (gold), TLT, IEF (bonds) — to test the diversification benefit of multi-market
trend following.

## Signal
Two breakout systems run together, N = ATR(20):
- **System 1:** enter on a 20-day high, exit on a 10-day low — with the Turtle filter
  (skip the next S1 entry if the *last* S1 trade on that name won, then take the one
  after).
- **System 2:** enter on a 55-day high, exit on a 20-day low — every signal taken.

Sizing is **1% equity risk per "unit"** (`shares = 0.01 × equity / N`). **Pyramiding:**
add a unit each time price advances ½N past the last add, up to 4 units per
instrument; the stop is **2N below the last add**. Correlation-group caps (equity
SPY/QQQ, bonds TLT/IEF, commodities GLD): max 6 units/group, 12 total. Signals at the
close, fills at the next open (+5 bps slippage), era-based costs. Long-only.

## Result
Committed run (multi-ETF, as intended): **Sharpe 0.20, CAGR 1.7%, MaxDD −29.7%**, validation verdict **REJECT**. The section below explains why.

## Faithfulness & the data-gap finding
The port logic is **faithful** — restricted to the two instruments the original
actually traded (SPY + QQQ) it reproduces the original closely: this port **0.51 /
MaxDD −16%** vs the original **0.47 / −18%**.

The original's headline, though, was a **data artifact**: its yfinance pull effectively
never traded GLD / TLT / IEF (its trade log shows QQQ 119, SPY 100, **TLT 1, GLD 0,
IEF 0** over 2005–2026 — impossible with real data, e.g. through the 2005–2011 gold
bull). With **clean EODHD data**, those three ETFs break out hundreds of times, and
trading them as the multi-ETF thesis *intends* drags the strategy to **Sharpe 0.22 /
MaxDD −30%** (REJECT): the bond/gold breakouts whipsaw and the diversification does not
help. This is the honest result the framework surfaces — the equity-only Turtle works
(0.51), the multi-ETF version does not.

## Universe & data
Fixed ETF basket (no index membership → no survivorship concern): SPY, QQQ, GLD, TLT,
IEF. EODHD **split-adjusted** prices for the channels/ATR (a trend system should react
to price, not dividend, moves). Era-based costs (15 bps in 2002 → 3 bps post-2019).

## Baseline
Atomic — the source has a single Multi-Instrument config (no separate optimization),
so the `v0` baseline is identical to this production version.

## Deviations from the original TurtleTrading_S16
EODHD split-adjusted prices (original used yfinance, which was missing the bond/gold
breakouts — see above); otherwise channels, unit sizing, pyramiding, the S1 filter,
correlation caps, and next-open execution match. Event-driven, simulated directly.

## Run
```
python -m rigor run strategies/trend_following/turtle_trading_etf --as-of 2026-06-01
```
