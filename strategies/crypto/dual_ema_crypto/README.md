# Dual-EMA Trend (BTC + ETH)

- **Slug:** `dual_ema_crypto`
- **Category:** crypto · **Version:** v1 · long-only

## Thesis
Crypto trends persist. Hold a coin while its fast EMA is above its slow EMA *and*
price is above its long SMA (macro uptrend); step out when either flips. Equal
weight across BTC and ETH.

## Universe & data
`BTC-USD.CC`, `ETH-USD.CC` (EODHD crypto), from 2017-01-01. Daily.

## Parameters
Promoted optimum `ema_fast=12, ema_slow=28, sma_trend=150, enable_short=0` (CPCV
ROBUST); baseline = engine defaults `8/21/200, enable_short=1`. Per-coin target
weight = pos/N; `commission_bps=15` (0.10% commission + 5 bps slippage).

## Results & faithfulness
**Faithful: corr +0.951 vs the TradingDesk original**, matching Sharpe (mine 1.26 vs
1.19 on the overlap). PROMOTE (ROBUST 85/100).

Note: the Rigor engine annualises on the inferred ~252-day calendar rather than the
365-day crypto convention, so the headline Sharpe (~1.12 full) is ~0.83× the source's
365-day figure; corr is unaffected. (corr vs User's prior port = 0.78 — his port
used a slightly different crypto window; this rebuild tracks the *original* more
closely.)

```
python -m rigor run strategies/crypto/dual_ema_crypto --as-of 2026-06-01
```
