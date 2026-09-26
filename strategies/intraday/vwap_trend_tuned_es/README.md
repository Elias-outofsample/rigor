# VWAP Trend, IS-tuned (OOS window) (ES) (`vwap_trend_tuned_es`)

## Headline result — this is the OUT-OF-SAMPLE window only
Parameters were chosen by sweeping **720 grid cells on 2010-06-07 → 2020-11-02**
(in-sample, 2628 sessions). The window below was never used for selection and is
touched once. Databento ES 1-minute bars resampled to 5m, RTH, net 1.0bp per side.

| metric | value |
|---|---|
| Sharpe | -0.15 |
| CAGR | -1.78% |
| MaxDD | -25.40% |
| Sortino | -0.21 |
| OOS sessions | 1416 (2020-11-03 → 2026-06-01) |
| Verdict | **REJECT** (OVERFIT, 0/100) |

| | in-sample | out-of-sample |
|---|---|---|
| paper's own parameters | -2.41 | -1.84 |
| **IS-optimised (this folder)** | **+0.15** | **-0.15** |
| robustness-selected alternative | +0.15 | +0.37 |

**Deflated Sharpe of the in-sample winner: 0.004** — corrected for the 720
trials actually run. Anything near zero means the in-sample peak is trial luck, not skill.

> **Two different deflated Sharpes appear in this folder, on purpose.** The number
> above deflates the *in-sample winner* for the trials that produced it — the
> standard Bailey/López de Prado usage, and the one that answers "was the search
> result skill?". The `DSR P(skill)` on the report card deflates *this window's own*
> (out-of-sample) Sharpe by the same trial count, which is deliberately conservative
> since the out-of-sample window was never searched. Neither clears its gate here.

## Thesis
A — Zarattini & Aziz, *VWAP Trend* (SSRN 4631351)

Long above the session VWAP, short below, flipping on every cross. The paper reports ~2.1 on QQQ/TQQQ 2018-2023 and specifies no stop; one is added here so per-session risk is bounded.

## What was optimised
The paper's rule has no entry-quality logic at all: it flips on every VWAP cross. Four
causal knobs were added to the shared engine and swept — they default to no-ops, so the
untuned strategy remains bit-identical to the paper.

| knob | what it does | selected here |
|---|---|---|
| `entry_band_atr` | hysteresis band around VWAP (in prior-day ATR). A cross inside the band is not a signal at all, so a wobble on the line costs nothing | **0.1** |
| `stop_atr_mult` | stop distance in prior-day ATR | 1.5 |
| `min_bars_cooldown` | bars to wait after a close before re-entering | 0 |
| `max_entries_per_day` | cap on entries per session (0 = uncapped) | 0 |
| `slope_bars` | require the session VWAP itself to be sloping with the trade | 3 |

`entry_band_atr` is the one that matters, and it was selected **> 0 on all eight
instruments** — the band is a genuine structural fix for the mechanism's real failure
mode (crosses cluster where price sits on VWAP, and each one pays a round trip). It
lifts the mean in-sample Sharpe of mechanism A from **−1.57 to +0.45**. The other four
knobs scatter with no pattern across instruments, which is what fitting noise looks like.

## Method
Selection used **only** the in-sample window. The out-of-sample window was evaluated
once, after the parameters were locked. Features (ATR, ADX, prior-session extremes) are
still built on the **full** series — `eval_start` trims the reported *returns*, not the
data — so the first out-of-sample sessions carry a proper warmup rather than 14 sessions
of NaN.

Entries fill at the CLOSE of the triggering bar; stops and targets are evaluated from the
NEXT bar. All logic is in `rigor.strategies.vwap_papers_engine`, pinned bit-for-bit to the
reference implementation by `tests/test_vwap_papers_engine.py`.

## Honest read
The entry band is a real improvement to the mechanism and it is not enough. On ES the optimised parameterisation is -0.15 out-of-sample (deflated Sharpe 0.004): the in-sample gain did not transfer at all. Not deployable. Committed as evidence of the search, not as a candidate.

## How to run
```bash
python -m rigor run strategies/intraday/vwap_trend_tuned_es --as-of 2026-06-01 --offline
```
