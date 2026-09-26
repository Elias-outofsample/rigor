# VWAP Regime Router, causal (ES) (`vwap_regime_route_causal_es`)

## Headline result
Databento ES 1-minute bars resampled to 5m, RTH 09:30–16:00 ET, 2010-06-07→2026-06-01, net 1.0bp per side.

| metric | value |
|---|---|
| Sharpe | -1.03 |
| CAGR | -6.84% |
| MaxDD | -68.65% |
| Sortino | -1.39 |
| Sessions | 4044 |
| Verdict | **REJECT** (OVERFIT, 18/100) |

In-sample (to 2019-12-31) Sharpe -1.26; out-of-sample (2020-01-01 on) Sharpe -0.79.

## Thesis
D — Lee, volatility-normalised VWAP-deviation regime (SSRN 6438039), routing A and C — look-ahead removed

Identical to the as-published router except that the trend branch only begins trading AFTER the classification bar, so no P&L is routed on information that did not yet exist.

This is the honest version of mechanism D and the one to read.

## Universe / data
ES continuous front-month (CME e-mini S&P 500), Databento `GLBX.MDP3` `ohlcv-1m` committed in-repo at
`data_share/databento/ES_ohlcv1m.parquet`, resampled to 5-minute bars, RTH 09:30–16:00 ET only,
coverage 2010-06→2026-06. Served by `rigor.data.build_loader` → `LocalFuturesLoader`.
ES → FTMO **US500**. Flat overnight ⇒ roll-immune; roll sessions are skipped anyway.

## Rule
All logic lives in `rigor.strategies.vwap_papers_engine` (mechanism `regime_routed`), a **bit-for-bit
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
| `regime_dev_threshold` | 1.5 | trend if mean |dev/σ| ≥ 1.5 (grid 1.0/1.5/2.0) |
| `regime_n_bars` | 12 | classification window in bars (grid 12/30) |
| `regime_causal` | true | trend branch starts only after the label exists |
| `commission_bps` | 1.0 | per side; a round trip is charged per entry |
| `interval` | 5m | 1-minute bars resampled; see Robustness for the resolution sweep |

## Robustness
This is the as-published router with the leak closed, and it removes the
result: the 8-instrument mean goes from **+0.82** (as published) to **−0.61** (causal). For
ES specifically, +0.246 → -1.026. A strictly-lagged label (yesterday's regime, known at the
open) gives -1.503, and a *randomly shuffled* label gives -1.461 — statistically the same
thing, which is the tell that the classifier carries no information.

## Honest read
The honest version of the Lee router, and it has no edge: routing between two mechanisms that are individually negative does not produce a positive one, and the regime label itself is indistinguishable from noise. Not deployable. Clean negative result.

## How to run
```bash
python -m rigor run strategies/intraday/vwap_regime_route_causal_es --as-of 2026-06-01 --offline
```
