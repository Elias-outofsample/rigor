# Credit Carry Rotation (S34)

Monthly rotation across credit / duration ETFs — **HYG / IEF / EMB / SHY** (cash BIL) — driven
by a credit-carry price-ratio signal plus a FRED credit-stress overlay. Port of an earlier research
version's `Carry_S34` **Credit** sub-strategy. (The S34 *main* blends 60% futures carry/trend + 40% credit;
the futures leg has no EODHD data, so this ports the portable credit-carry leg — the first
strategy unlocked by the FRED connector.)

## Signal

Each month-end, on total-return-adjusted prices:
- `ratio = HYG / IEF` vs its **9-month** moving average.
- **Macro overlay:** if FRED high-yield OAS (`BAMLH0A0HYM2`) > 8%, or the yield curve (`T10Y3M`)
  inverts past the threshold (disabled at −99 here), de-risk to mostly SHY.
- Otherwise by the ratio: **carry-on** (ratio > MA+1%) → HYG 0.55 / EMB 0.15 / IEF 0.30;
  **carry-off** (ratio < MA−1%) → IEF 0.60 / SHY 0.20 / HYG 0.20; **neutral** → HYG 0.45 / IEF
  0.25 / SHY 0.15 / EMB 0.15.

Monthly target weights applied to daily returns; 5 bps turnover cost.

## Result (this port) — faithful

| metric | this port | original |
|---|---|---|
| Sharpe | 0.76 | 0.52 |
| CAGR | 3.75% | 2.49% |
| MaxDD | −19.9% | — |

**Return correlation to the original is 0.995** — the allocation logic is reproduced essentially
exactly. The higher Sharpe/CAGR is the EODHD-vs-yfinance **total-return adjustment on the bond
ETFs** (monthly coupon handling differs by vendor), a magnitude-only difference on an identical
return shape. Verdict: **PROMOTE**.

## Deviations

- EODHD total-return-adjusted prices (original used yfinance auto-adjust).
- FRED OAS / T10Y3M via the framework's seeded cache.
- Only the credit-carry leg of S34 is ported (the 60% futures leg has no EODHD data).

## Baseline (v0)

`strategies/Baseline/carry/credit_carry_rotation/` runs the identical rotation **frictionless**.
