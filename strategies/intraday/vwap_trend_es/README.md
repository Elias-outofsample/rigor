# VWAP Trend (ES) (`vwap_trend_es`)

## Headline result
Databento ES 1-minute bars resampled to 5m, RTH 09:30–16:00 ET, 2010-06-07→2026-06-01, net 1.0bp per side.

| metric | value |
|---|---|
| Sharpe | -2.20 |
| CAGR | -23.50% |
| MaxDD | -98.70% |
| Sortino | -2.91 |
| Sessions | 4044 |
| Verdict | **REJECT** (OVERFIT, 26/100) |

In-sample (to 2019-12-31) Sharpe -2.65; out-of-sample (2020-01-01 on) Sharpe -1.68.

## Thesis
A — Zarattini & Aziz, *VWAP Trend* (SSRN 4631351)

Long while price trades above the session VWAP, short while below; the position flips on every VWAP cross. The paper reports a risk-adjusted return near 2.1 on QQQ/TQQQ over 2018–2023 — one asset family, six years, so nothing about it has been tested out of sample until now.

The paper does **not** specify a stop. One is added here (`stop_atr_mult` × the prior-day ATR) because without it the per-session risk is unbounded and the result cannot be sized.

## Universe / data
ES continuous front-month (CME e-mini S&P 500), Databento `GLBX.MDP3` `ohlcv-1m` committed in-repo at
`data_share/databento/ES_ohlcv1m.parquet`, resampled to 5-minute bars, RTH 09:30–16:00 ET only,
coverage 2010-06→2026-06. Served by `rigor.data.build_loader` → `LocalFuturesLoader`.
ES → FTMO **US500**. Flat overnight ⇒ roll-immune; roll sessions are skipped anyway.

## Rule
All logic lives in `rigor.strategies.vwap_papers_engine` (mechanism `vwap_trend`), a **bit-for-bit
port** of the reference implementation — verified to `max|diff| = 0` on daily points P&L across
15 parameter configurations (`tests/test_vwap_papers_engine.py`).

Entries fill at the CLOSE of the triggering bar; stops and targets are evaluated from the NEXT
bar onward, never on the entry bar. Every input is causal: the daily ATR is Wilder-smoothed over
*daily* bars and shifted one session, prior-session high/low are the previous session's extremes,
the session VWAP and its deviation-sigma are cumulative within the session, and ADX is
Wilder-smoothed over past bars only.

Session points P&L is converted to a return on notional by dividing by that session's opening
price (one contract, the same fixed-notional convention as the other futures strategies in the
book). Each entry is charged a round trip of 2 × `commission_bps`.

## Parameters
| param | value | note |
|---|---|---|
| `stop_atr_mult` | 1.0 | stop = 1.0 × prior-day ATR (grid 0.5/1.0/1.5) |
| `single_entry_per_day` | false | flip on every cross (grid false/true) |
| `commission_bps` | 1.0 | per side; a round trip is charged per entry |
| `interval` | 5m | 1-minute bars resampled; see Robustness for the resolution sweep |

## Robustness
**Transaction cost is the whole story.** On ES the same signal is worth **+0.72 gross**,
but at 6.85 entries per session it needs an unrealistically thin round trip to survive: +0.72 at
0bp, **−0.80 at 0.5bp**, **−2.20 at 1bp**, **−4.69 at 2bp** per side. The VWAP-cross signal is not
noise — it is real and sub-transaction-cost at this turnover.

**Bar resolution (ES).** −0.80 at 15m, −2.20 at 5m, −6.34 at 1m: monotone in trading frequency,
which is the cost-drag signature, not an edge that appears at some special resolution.

## Honest read
Does not replicate on futures. The signal has a small real gross edge (+0.72 on ES) but the mechanism trades ~7 times a session and pays it all away; the paper's QQQ/TQQQ result does not survive an instrument change or a realistic cost model. Not deployable. Kept for reproducibility and as the cost-drag reference case.

## How to run
```bash
python -m rigor run strategies/intraday/vwap_trend_es --as-of 2026-06-01 --offline
```
