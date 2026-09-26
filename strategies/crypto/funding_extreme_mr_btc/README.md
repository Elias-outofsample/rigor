# Funding-Extreme Mean-Reversion (BTC)

> Crypto funding-rate mean-reversion, ported from an earlier research prototype.
> Lifecycle **idle** (research). Slug `funding_extreme_mr_btc` · category `crypto` · as-of `2026-06-07`.

## Thesis

Fade crowded perpetual-futures positioning in BTC. When the perp funding rate is
extremely positive (z-score high), crowded longs are paying to be long → expect a
pullback; extremely negative funding → crowded shorts → expect a bounce. We trade the
reversion gated by a spot trend MA (only fade over-extensions).

## Data adaptation

The original (#096) trades **hourly perp** klines. The rigor data layer has 8-hourly
Binance funding history (`data.funding`) but **no perp price klines**, so this is a
**daily** adaptation traded on the EODHD **spot** price (`BTC-USD.CC`) — perp and
spot track within the basis, so funding extremes predict spot reversion too. Funding
is aggregated to a daily rate, z-scored, shifted one day (no look-ahead).

## Results (committed artifacts, as-of 2026-06-07)

| Metric | Value |
|---|---|
| Sharpe | -0.39 |
| CAGR | -10.23% |
| Max drawdown | -72.54% |
| Excess vs cash (annual) | -11.36% |
| Observations | 2349 |
| Validation verdict | **REJECT** (grade OVERFIT, score 0/100) |

Idle/research; reimplemented on EODHD daily spot + Binance funding.
