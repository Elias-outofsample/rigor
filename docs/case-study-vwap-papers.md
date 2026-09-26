# The VWAP-papers family — four academic intraday mechanisms on CME futures

Independent test of four intraday papers on Databento 1-minute futures bars (licensed data,
not included in this repository), declined across eight instruments (48 strategies). Four
representative ES folders ship in [`strategies/intraday/`](../strategies/intraday/) with their
committed artifacts; the engine and its bit-for-bit reference test are in the package.

All logic lives in one shared engine, [`rigor/strategies/vwap_papers_engine.py`](../src/rigor/strategies/vwap_papers_engine.py);
each strategy folder is a thin shim over it. The engine is a numpy rewrite of a row-wise
reference implementation and is pinned to it **bit-for-bit** by
[`tests/test_vwap_papers_engine.py`](../tests/test_vwap_papers_engine.py) — the reference
is embedded in the test file and compared at `max|diff| = 0` across 15 parameter configurations.

## The mechanisms

| id | paper | mechanism |
|---|---|---|
| **A** | Zarattini & Aziz, *VWAP Trend* (SSRN 4631351) | long above session VWAP, short below, flip on every cross |
| **B** | Bhatti (SSRN 6454659) | fade a prior-session extreme when far from VWAP and ADX is falling from a peak |
| **C** | Grant, Wolf & Yu (SSRN 689282, *JBF* 2005) | fade the first 30 minutes when the move exceeds 0.4 × ATR |
| **D** | Lee (SSRN 6438039) | volatility-normalised VWAP-deviation regime label, routing A on trend days and C on reversion days |

A carries an added ATR stop (the paper specifies none, so per-session risk would be unbounded).
B's thresholds are a reasonable reading — the paper gives none and, by its own admission, was
never backtested by anyone, including its author.

## Result — full-sample net Sharpe, 1bp per side

Databento 1-minute bars resampled to 5m, RTH only, flat overnight, 2010-06 → 2026-06
(RTY from 2017-07, BTC 2017-12, ETH 2021-02).

| mechanism | ES | NQ | RTY | CL | GC | SI | BTC | ETH | mean |
|---|---|---|---|---|---|---|---|---|---|
| A `vwap_trend` | −2.20 | −1.27 | −1.93 | −0.09 | −3.15 | −1.06 | −0.69 | −0.48 | **−1.36** |
| A `vwap_trend_single` | −0.93 | −0.41 | −0.35 | −0.54 | −1.18 | −0.12 | −0.04 | −0.26 | **−0.48** |
| C `opening_fade` | −0.18 | −0.28 | −0.10 | −0.00 | −0.53 | +0.09 | +0.57 | +0.19 | **−0.03** |
| B `adx_vwap_rev` | −0.91 | −0.96 | −0.28 | −0.33 | −0.71 | −0.30 | −1.20 | −0.67 | **−0.67** |
| D `vwap_regime_route` (as published) | +0.25 | +0.98 | +0.84 | +0.52 | −0.30 | +1.24 | +1.47 | +1.58 | **+0.82** |
| D `vwap_regime_route_causal` | −1.03 | −0.80 | −0.97 | +0.06 | −1.53 | −0.29 | −0.36 | +0.07 | **−0.61** |

**All 48 carry a `REJECT` verdict** from the validation battery. None is deployable; every folder
is `status: idle`.

## The one result that looks good is a look-ahead

Mechanism D is the only row that looks like alpha, and it is an artifact. The regime label is
computed from the first `regime_n_bars` (12) bars of a session, but the A branch it routes to
starts trading at bar 1 — so a whole session's P&L is selected using information from inside
that same session.

Holding the routing mechanics **completely fixed** and changing only the label's information
timing:

| label used | ES | 8-instrument mean |
|---|---|---|
| today's label, routes the full session (as published) | +0.246 | **+0.82** |
| **yesterday's** label (strictly known at the open) | −1.503 | **−0.81** |
| today's label, trend branch starts after the label exists | −1.026 | **−0.61** |
| **randomly shuffled** label | −1.461 | **−0.80** |

Yesterday's real label (−0.81) and a **randomly shuffled** label (−0.80) perform identically. A
classifier carrying genuine information would beat a coin flip; this one does not. The entire
+0.82 is the leak.

The as-published number is also unstable in bar resolution — +2.18 at 15m, +0.25 at 5m, −3.18 at
1m on ES — while the lagged-label version is negative at every resolution (−1.15 / −1.50 / −4.12).

## Mechanism A is real but sub-transaction-cost

The most useful positive finding. On ES the VWAP-cross signal is worth **+0.72 gross**, but it
trades 6.85 times per session:

| bps per side | ES net Sharpe |
|---|---|
| 0.0 | **+0.72** |
| 0.5 | −0.80 |
| 1.0 | −2.20 |
| 2.0 | −4.69 |

Net Sharpe is also monotone in trading frequency (−0.80 at 15m, −2.20 at 5m, −6.34 at 1m) — the
cost-drag signature, not an edge that lives at some particular resolution. Capping the session at
one entry recovers most of the loss (mean −1.36 → −0.48), which confirms the damage is turnover
rather than signal direction.

So Zarattini & Aziz's mechanism is not noise; it is an edge too thin to pay for its own execution
on futures at this turnover. The paper's reported ~2.1 on QQQ/TQQQ over 2018–2023 does not survive
an instrument change or a realistic cost model.

## Mechanisms B and C

**C (Grant, Wolf & Yu)** lands at a mean of −0.03 across eight instruments — indistinguishable
from zero, exactly as the original authors predicted when they warned that significance "drops
a lot once transaction costs are applied". The BTC reading (+0.57) is a short sample and does not
clear its own deflated-Sharpe gate.

**B (Bhatti)** is negative on all eight instruments (mean −0.67). This is the first empirical test
the mechanism has had; the paper left backtesting "for future work". Recording it as a clean
negative result is the point.

## Optimisation study — IS search, one-shot OOS test

The numbers above are the papers' own parameterisations, run once. This section asks the
separate question: **does tuning rescue any of it?**

**Protocol.** Sessions are split chronologically **65 / 35** per instrument (ES: IS
2010-06-07 → 2020-11-02, 2628 sessions; OOS 2020-11-03 → 2026-06-01, 1416). The grid is
swept on IS **only**; the winner is locked and OOS is evaluated **once**. Two selectors
are reported — `best_is` (max in-sample Sharpe, the naive one) and `robust` (IS is halved
again; a cell must be positive in *both* halves, then maximise the worse half). The
in-sample winner's Sharpe is deflated for the trial count actually run.

**Entries were given real work.** The papers have no entry-quality logic — mechanism A
flips on every VWAP cross. Five causal knobs were added to the engine and swept, each
defaulting to a no-op so the untuned strategy stays bit-identical to the paper:
`entry_band_atr` (hysteresis band around VWAP), `stop_atr_mult`, `min_bars_cooldown`,
`max_entries_per_day`, `slope_bars` (VWAP-slope confirmation); plus `max_threshold_atr_mult`
and `stop_cap_atr` on mechanism C and an exposed `stop_atr_mult` on B.

| mechanism | trials | IS default | IS best | OOS default | **OOS best** | OOS robust | mean DSR | OOS > 0 |
|---|---|---|---|---|---|---|---|---|
| A `vwap_trend` | 720 | −1.57 | +0.45 | −0.99 | **−0.19** | +0.12 | 0.094 | 3/8 |
| C `opening_fade` | 324 | +0.15 | +0.53 | −0.33 | **−0.23** | −0.38 | 0.072 | 2/8 |
| B `adx_vwap_rev` | 108 | −0.65 | −0.31 | −0.75 | **−0.74** | none survived | 0.003 | 0/8 |
| D router, **as published (leaks)** | 162 | +0.67 | +2.90 | +1.20 | **+3.39** | +3.39 | 0.998 | 8/8 |
| D router, causal (leak closed) | 162 | −0.88 | +0.55 | −0.61 | **−0.10** | −0.22 | 0.187 | 4/8 |

*(means over the eight instruments)*

**Optimisation manufactures in-sample Sharpe and none of it transfers.** Every honest
mechanism goes from a negative in-sample default to a positive in-sample optimum — A from
−1.57 to +0.45, the causal router from −0.88 to +0.55 — and every one lands at roughly
zero or below out-of-sample, with deflated Sharpe ≤ 0.19. Mechanism B cannot be made
positive in-sample *at all* across 108 cells, which is how dead it is.

**Optimising a leak amplifies it.** The as-published router is the only row that survives,
and it goes from +1.20 untuned to **+3.39** tuned out-of-sample — the search finds the
settings that exploit the look-ahead hardest. Its OOS *exceeds* its IS on five of eight
instruments, which is the signature of a structural artifact rather than a fitted edge: a
real edge decays out-of-sample, a leak is equally available in both windows.

**A second leak was found and closed here.** The original causal router deferred only the
*trend* branch past the classification bar. The *fade* branch still entered at the end of
its 30-minute window — bar 6 on 5-minute bars — while the label consumed bars 0–11. Gating
both branches (`min_entry_bar`) moved the tuned causal router's mean OOS from **+0.68 to
−0.10**. Pinned by `test_causal_router_defers_BOTH_branches_past_the_label`.

**The one durable finding.** `entry_band_atr` was selected **> 0 on all eight instruments**
(0.05–0.35 ATR) — the hysteresis band is a genuine structural fix for the mechanism's real
failure mode, and it is what lifts A's in-sample mean from −1.57 to +0.45. The other four
knobs scatter with no cross-instrument pattern, which is what fitting noise looks like. The
band is a real improvement to the entries; it is still not enough to clear costs
out-of-sample.

The 16 tuned parameterisations are committed as `vwap_trend_tuned_*` and
`vwap_regime_route_causal_tuned_*`. Their report cards cover **only the OOS window**
(`eval_start` trims the reported returns while features stay warmed up on the full series),
so each committed artifact *is* the out-of-sample test. All 16 carry a `REJECT` verdict.

## Method notes

- **Causality.** Daily ATR is Wilder-smoothed over daily bars and shifted one session; prior-session
  high/low are the previous session's extremes; session VWAP and the deviation-sigma are cumulative
  within the session; ADX is Wilder-smoothed over past bars only. Entries fill at the CLOSE of the
  triggering bar; stops and targets are evaluated from the NEXT bar. Pinned by causality tests that
  truncate the series and assert surviving bars' features are unchanged.
- **Accounting.** Session points P&L ÷ that session's opening price = return on notional (one
  contract, the same fixed-notional convention as the other futures strategies in the book). Each
  entry is charged a round trip of 2 × `commission_bps`. Roll sessions stand aside.
- **Parameterisation.** Every knob the source specifies is declared explicitly in `config.extra`.
  This is load-bearing: `StrategyBase.default_params()` falls back to the *first grid cell* for any
  axis the config leaves undeclared, which silently backtests a different parameterisation than the
  one the source describes.
- **One normalisation artifact.** `vwap_trend_single_cl` reports an undefined CAGR: on 2020-04-21,
  the session after WTI's negative settlement, CL opened at \$1.75 and dividing points P&L by that
  near-zero price produces a return below −100%. The dollar loss is real; the percentage is the
  artifact. Left in rather than clipped — 1 session of 4111.

## Reproduce

```bash
python -m rigor run strategies/intraday/vwap_regime_route_es --as-of 2026-06-01 --offline
```

Each folder ships `returns.csv`, `summary.json`, `report.html` (the light report card) and
`thesis.pdf` under `artifacts/`, so the whole study is browsable without the data present.
