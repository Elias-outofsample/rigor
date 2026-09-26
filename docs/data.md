# Data

Every number `rigor` produces flows through one deterministic layer, `rigor.data`: a
single primary price source, an as-of-bounded cache, deterministic corporate-action
adjustment, and point-in-time universes. **No market data ships with this repository**
— the committed strategy artifacts are *results*, not inputs.

## Providers

| Source | What | Auth |
|---|---|---|
| **EODHD** | Daily EOD prices (split/dividend adjusted), 5-minute intraday, fundamentals, index constituents | API key in `.env` (`EODHD_API_KEY`) |
| **FRED** (St. Louis Fed) | Macro series: risk-free rate (`DGS3MO`), `VIXCLS`, rates, credit spreads | optional free key (`FRED_API_KEY`); falls back to the keyless CSV endpoint |
| **Binance** public API | Crypto perpetual funding rates, klines, aggressor-signed trades | none |
| **Databento** (local files) | 1-minute CME futures bars for the intraday futures studies | files you supply under `data/databento/` or `$RIGOR_FUTURES_DATA_DIR` |
| **Point-in-time membership** (frozen event logs) | Index add/remove history for survivorship-free universes | see below |

Copy `.env.example` to `.env` and fill in your keys; `python scripts/verify_eodhd.py`
checks the key and plan. `.env` is gitignored.

## As-of dates and reproducibility

Every pull is bounded by an **as-of date** and the cache is keyed by it, so two machines
that pin the same as-of date get the same inputs regardless of when they ran:

```bash
rigor run strategies/trend_following/sma_trend --as-of 2026-06-01
```

- Live data is the default. Vendors occasionally restate history, so live runs are
  close but not byte-identical across time — pin `--as-of` for anything you commit.
- For exact reproduction, freeze the cache into a release and run offline:
  `rigor data freeze --as-of 2026-06-01` → `rigor data verify` → `rigor run … --offline`.
- The test suite proves the offline path is deterministic on a committed synthetic
  snapshot (`tests/test_reproducibility.py`).

## Point-in-time universes

Backtesting today's index members over the past is survivorship bias. `rigor` replays
a frozen add/remove **event log** per index and maps it onto price symbols:

```
<reference dir>/<index>/<index>_membership.json      # {"events": [{"date", "ticker", "action"}]}
```

- **CAC 40, DAX 40, EURO STOXX 50** event logs ship with the package; they were compiled
  from public sources (exchange announcements, press releases, public encyclopedias).
- **S&P 500 and Nasdaq-100** histories come from a licensed vendor and are **not
  shipped**. Put your own files under `$RIGOR_REFERENCE_DIR/sp500/` and
  `$RIGOR_REFERENCE_DIR/nasdaq100/` (same format). Strategies that need them fail with
  an explicit message until you do; the test suite uses a synthetic event log.
- Delisted tickers are mapped carefully: when a symbol is later reused by another
  company, the mapping is only accepted if the price history ends at the real
  delisting date (see [design-decisions.md](design-decisions.md), section 3).

## Cache

- One shared cache under `data_cache/` (gitignored), keyed by symbol and as-of date.
- Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM*`, `LPT*`) are
  escaped in cache filenames — a raw `CON.json` is a console device and hangs forever.
- TLS goes through the OS trust store (`truststore`), so corporate SSL-inspection
  proxies work without disabling verification.

## Futures

EODHD has no futures. `FuturesDataSource` reads continuous, back-adjusted contracts
from files you export (Norgate, CSI, Interactive Brokers…) — see
[futures-data.md](futures-data.md). The intraday VWAP study
([case-study-vwap-papers.md](case-study-vwap-papers.md)) ran on Databento 1-minute
bars supplied the same way.
