# Faber GTAA (5-asset)

- **Slug:** `faber_gtaa`
- **Category:** trend_following · **Version:** v1 · long-only · role: defensive

## Thesis
Faber (2007) GTAA: hold an asset-class sleeve only while it's above its 10-month
SMA, equal-weight across active sleeves, else cash. A simple, robust crash-avoider.

## Universe & data
SPY (US stocks), EFA (foreign), IEF (bonds), VNQ (real estate), GSG (commodities).
Total-return adjusted EODHD via the DataLoader, from 2010-01-01. Monthly trend
decision, held across daily bars → daily net returns (momentum_s7 pattern).

## Parameters
`sma_months=10` (engine default; no promoted override). `commission_bps=5`.

## Results & faithfulness
**Faithful — but compared at the right granularity.** The TradingDesk original
artifact is a **monthly** return series (one value per month stamped on month-ends),
so a naive *daily*-vs-daily corr is meaningless (≈0.06). On a like-for-like
**monthly** basis the reproduction is **corr +0.833** with matching full-period
Sharpe (daily-annualised mine 0.62 vs original 0.60). corr **+1.000 vs User's prior
EODHD port** confirms an exact rebuild. The residual is daily-MTM-vs-pure-monthly
compounding plus EODHD-vs-original ETF data. PROMOTE (ROBUST 82/100).

```
python -m rigor run strategies/trend_following/faber_gtaa --as-of 2026-06-01
```
