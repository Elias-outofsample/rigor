# VWAP Regime Router, as-published (ES) (`vwap_regime_route_es`)

## Headline result
Databento ES 1-minute bars resampled to 5m, RTH 09:30–16:00 ET, 2010-06-07→2026-06-01, net 1.0bp per side.

| metric | value |
|---|---|
| Sharpe | 0.25 |
| CAGR | 1.67% |
| MaxDD | -24.19% |
| Sortino | 0.37 |
| Sessions | 4044 |
| Verdict | **REJECT** (OVERFIT, 26/100) |

In-sample (to 2019-12-31) Sharpe +0.06; out-of-sample (2020-01-01 on) Sharpe +0.46.

## Thesis
D — Lee, volatility-normalised VWAP-deviation regime (SSRN 6438039), routing A and C

Not a strategy but a filter. Classify each day trend vs reversion from the mean absolute VWAP deviation, normalised by intraday volatility, over the first `regime_n_bars` bars; route trend days to mechanism A and reversion days to mechanism C.

**Published wiring — carries a look-ahead.** The label is only known after `regime_n_bars` bars, but the A branch starts trading at bar 1, so a whole session's P&L is routed on a label that did not exist when that session's first trades were taken. See the Robustness section: this leak is the entire result.

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
| `regime_causal` | false | **as-published wiring — leaks** |
| `commission_bps` | 1.0 | per side; a round trip is charged per entry |
| `interval` | 5m | 1-minute bars resampled; see Robustness for the resolution sweep |

## Robustness
**The published result is the look-ahead, and here is the proof.** Holding the routing
mechanics completely fixed and changing ONLY the information timing of the regime label:

| label used | ES Sharpe | 8-instrument mean |
|---|---|---|
| today's label, routes the full session (as published) | +0.246 | **+0.82** |
| **yesterday's** label (strictly known at the open) | -1.503 | **−0.81** |
| today's label, trend branch starts after the label exists | -1.026 | **−0.61** |
| **randomly shuffled** label | -1.461 | **−0.80** |

Yesterday's real label (−0.81) and a *randomly shuffled* label (−0.80) perform identically. A
regime classifier that carried genuine information would beat a coin flip; this one does not. The
entire +0.82 comes from routing a session's first-hour P&L on a label computed from that same
first hour.

**Bar resolution (ES).** The as-published number swings +2.18 (15m) → +0.25 (5m) → −3.18 (1m) —
it is not a stable quantity. The lagged-label version is negative at every resolution
(−1.15 / −1.50 / −4.12).

## Honest read
**Do not read the headline number as alpha.** It is a look-ahead artifact, demonstrated by the fact that a randomly shuffled regime label performs the same as the real lagged one. Shipped only as the documented counter-example; read `vwap_regime_route_causal_es` instead. Not deployable.

## How to run
```bash
python -m rigor run strategies/intraday/vwap_regime_route_es --as-of 2026-06-01 --offline
```
