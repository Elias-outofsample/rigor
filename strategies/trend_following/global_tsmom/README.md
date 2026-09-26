# Global TSMOM — Time-Series Momentum (Main)

- **Slug:** `global_tsmom` · **Version:** `v1`
- **Category:** trend_following
- **Baseline:** atomic: the `v0` baseline is this production version (see *Baseline* below)

## Thesis
Time-series (trend-following) momentum across **22 global ETFs** spanning 8 asset
classes — US/intl/EM equity, government/corporate/high-yield/intl/EM bonds,
gold/silver/broad commodities, and US/intl REITs. Each market that has trended up over
the past year tends to keep trending. (Moskowitz–Ooi–Pedersen 2012; Hurst–Ooi–Pedersen
2017.) This is the "Global ETFs" variant of CommodityTSMOM_S21.

## Signal
Per ETF, independently: **go long if the trailing 12-month (252d) return > 0**, else
hold cash. Monthly rebalance (month-end). Long-only.

## Sizing
**Inverse-volatility** weights (1 / 20-day annualised realised vol, vol floored at 3%),
**scaled to a 10% annual portfolio-vol target** (diagonal: `port_vol = √Σ(wᵢ·volᵢ)²`),
each name capped at **15%** of NAV and total **gross leverage capped at 2.0×**. Daily
returns are winsorised at ±10%; the unallocated sleeve earns nothing (cash). Era-based
rebalance costs (5 bps pre-2010 → 2 bps post-2020) on turnover.

## Faithfulness & verdict
Reproduces the original almost exactly (all-days): original **0.77 / CAGR 11.9% / MaxDD
−35.5%**, this port **0.78 / 11.8% / −35.8%** (**PROMOTE**). A diversified,
risk-managed trend follower with a large but recoverable drawdown (the 2008/2020 trend
reversals).

## Universe & data
Fixed basket of 22 liquid global ETFs (no index membership → no survivorship concern):
SPY, QQQ, IWM, IWD, IWF, EFA, EWJ, EEM, TLT, IEF, SHY, TIP, LQD, HYG, BWX, GLD, SLV, DBC,
VNQ, RWX, EMB, DBA. EODHD total-return-adjusted prices.

## Baseline
Atomic — the source "Global ETFs" has a single config (no separate optimization), so the
`v0` baseline is identical to this production version.

## Deviations from the original CommodityTSMOM_S21
EODHD total-return-adjusted ETF prices (original used yfinance). Signal, vol-scaling,
caps, winsorisation, and monthly fixed-weight return accounting match the canonical
config. (S21's separate "Commodities" variant runs on commodity *futures* and is not
portable — EODHD has no futures data.)

## Run
```
python -m rigor run strategies/trend_following/global_tsmom --as-of 2026-06-01
```
