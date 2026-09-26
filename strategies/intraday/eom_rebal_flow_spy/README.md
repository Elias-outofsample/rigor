# eom_rebal_flow_spy

- **Slug:** `eom_rebal_flow_spy` · **Category:** intraday · **Role:** alpha (structural forced-flow, calendar-conditional)
- **Asset class:** US equity index (trades **SPY**; uses **TLT** as a cross-asset price signal)
- **Headline:** SPY Sharpe **+0.57** (IS **+0.63** / OOS **+0.49**), CAGR +3.5%, MDD −8.6%, WR 52%, N=740 trades over 2,992 sessions (2014–2026). Verdict **PROMOTE / ROBUST**.

## Thesis

Several $trillion in 60/40, target-date and pension funds rebalance toward fixed weights at each
calendar **month-end**. The rebalance trade is **mechanically directional** in the month's
equity-vs-bond performance: when equities **outperformed** bonds during the month, these funds are
overweight equities and **sell equities / buy bonds** into the month-end close (downward pressure on
SPY); when equities **underperformed**, they **buy equities** (upward pressure). The tradeable,
price-only signal is therefore:

```
spread = (SPY month-to-date return) − (TLT month-to-date return)   [measured the day before the window]
direction = −sign(spread)        # equities outperformed → SHORT SPY ; underperformed → LONG SPY
hold = open→close on each of the last 5 trading days of the calendar month ; flat otherwise
```

- **Source:** Etula, Rinne, Suominen & Vaittinen (2020), *Dash for Cash: Monthly Market Impact of
  Institutional Liquidity Needs* — predictable month-end price pressure from institutional rebalancing,
  with intraday concentration and a sign set by the equity-bond performance gap.
- **Provenance:** this is the **conditional (signed equity-bond spread) filter** that the committed
  `eoq_last5d_pension_rebal_short_spy` docstring explicitly names as its un-built improvement over a
  blind unconditional short.

## Why conditional, not a calendar long/short (the key result)

On long 1m data the **unconditional** month-end short *loses* (full-series Sharpe **−1.10**); only the
**conditional** rebalance-direction wins (**+0.57**). So this is not a generic month-end seasonal — the
**direction-conditioning is the alpha**, exactly as the rebalancing-flow mechanism predicts.

| month-end (last 5d) variant | full-series Sharpe |
|---|---|
| unconditional short (eoq-style, on SPY) | −1.10 |
| **conditional −sign(SPY−TLT spread)** | **+0.57** (IS +0.63 / OOS +0.49) |
| spread-gated \|spread\|>3% | +0.78 (fewer obs) |

## Universe & Data
- Trades **SPY** 1-minute RTH; signal from **SPY + TLT** daily closes (derived from 1m). 2014-01 → 2026-05.
- Single traded instrument (SPY); TLT enters only as the cross-asset spread signal — same single-asset/
  cross-signal pattern as the catalog's `leadlag_*` book and the TLT-filter suggested in `eoq_…`.
- Round-trip cost 1.5 bps (SPY highly liquid).

## Parameters
| param | value | note |
|---|---|---|
| `n_last` | 5 | last 5 trading days of each calendar month |
| `bond` | TLT | bond leg for the equity-bond spread |
| direction | −sign(MTD SPY−TLT spread) | conditional rebalance direction (signed, no gate — simplest) |
| `cost_bps` | 1.5 | round-trip |

## Results
### In-Sample — 2014-01 → 2020-12 — Sharpe **+0.63**, CAGR +3.9%, MDD −6.5%, N=415, WR 54%
### Out-of-Sample — 2021-01 → 2026-05 — Sharpe **+0.49**, CAGR +3.1%, MDD −8.6%, N=325, WR 49%
### Full — Sharpe **+0.57**, CAGR +3.5%, MDD −8.6%, N=740, WR 52%
**Verdict: IS ≈ OOS (+0.63 vs +0.49) → robust, validated out-of-sample.** Positive in ~9 of 13 years
(2018 +2.2, 2020 +3.5, 2022 +2.0, 2023 +2.1, 2025 +2.7 SR; negative 2015/2019/2021). Not single-year-dominated.

## ⚠️ Honest caveats
- WR ~52% (modest hit rate; positive expectancy from larger moves on big-rebalance months).
- Low frequency (~5 days/month × 12 = ~60 trades/yr) → judge on the multi-year aggregate.
- Cross-asset signal: relies on the SPY–TLT relationship persisting as the balanced-fund rebalance proxy.

## Distinct from catalog
- `eoq_last5d_pension_rebal_short_spy`: **quarter-end, UNCONDITIONAL short** (committed SR 0.158). This is
  **month-end (12×/yr), CONDITIONAL signed-spread direction** — the unconditional short loses on long data;
  conditioning is what works. The two trade different events (all month-ends vs 4 quarter-ends) and different
  directions (conditional vs always-short).
- `bond_monthend_*` (unconditional long bonds month-end) and `turn_of_month_*` (unconditional long equity TOM)
  are unconditional calendar seasonals on a single asset; this is the conditional equity-bond rebalance-flow.
