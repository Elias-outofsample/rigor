# Bollinger Squeeze Breakout (`bollinger_squeeze`)

**Category:** breakout · **Direction:** long/short · **Status:** idle (research)
Single-instrument (QQQ) daily breakout. Reimplemented on EODHD daily.

## Thesis
Bollinger **bandwidth** ((upper − lower)/MA) measures realised dispersion; when it
compresses to (near) its lowest level of the last `sq_win` sessions the bands are in a
**squeeze** — coiled, low-energy — which historically precedes a directional expansion.
The squeeze *arms* the strategy; the subsequent close *through* a band fires the trade.

## Mechanism
- **Bands:** MA(`n`) ± `k`·SD(`n`); `bandwidth = (up − dn)/MA`.
- **Squeeze (arm):** `bandwidth <= 1.05 * bandwidth.rolling(sq_win).min()`.
- **Long:** while armed and flat, close > upper band → +1 (disarm).
- **Short:** while armed and flat, close < lower band → −1 (disarm).
- **Exit:** long flattens when close < MA; short flattens when close > MA (mid-band exit).
- **Causal:** the arm reads the prior bar (t-1); the engine lags the position one more bar
  (`simulate_weights`), so there is no look-ahead. Position in {-1, 0, +1} is the weight.

## Committed metrics (offline, as-of 2026-06-07, n=5389, 2005-01-04 → 2026-06-05)
| Sharpe | CAGR | MaxDD | Sortino | Vol | Win rate | Verdict |
|---|---|---|---|---|---|---|
| −0.19 | −1.62% | −34.14% | −0.25 | 7.34% | 48.5% | REJECT (OVERFIT) |

Reimplemented on EODHD daily; **idle / research** (exempt from the promotion gate).
The committed `summary.json` reconciles with `returns.csv` (`rigor audit` OK). The signal
as specified does not carry on QQQ over this window; kept as a faithful research baseline.

## Reproduce
    python -m rigor run strategies/breakout/bollinger_squeeze --as-of 2026-06-07 --offline
