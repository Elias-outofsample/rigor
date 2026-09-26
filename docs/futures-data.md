# Futures data — the pluggable file-based on-ramp

The framework ships the *plumbing* for continuous / back-adjusted futures OHLCV,
but **no futures price data is bundled**: export your own history (Norgate, CSI,
Interactive Brokers…) into the layout below and point the source at it.

## Why a separate source

The default vendor (**EODHD**) has **no futures of any kind** (verified 2026-06-05:
every continuous-futures ticker — `ES`/`NQ`/`CL`/`GC`/`ZN`/`6E`/`.COMM` — returns
HTTP 404). Strategies designed for futures therefore either run on liquid **ETF
proxies** (GLD, USO…) or need another source.

`rigor.data.FuturesDataSource` is the on-ramp: a concrete `DataSource` (same
protocol as the EODHD client) that reads continuous futures OHLCV from a **local
directory you populate**. It is **additive and optional** — the default EODHD
path is untouched, and an absent/empty directory makes the source inert (every
symbol returns no bars), so wiring it up can never break a run.

## File schema

**One file per continuous contract**, named for the root symbol (`CL.parquet`
or `CL.csv`). **Parquet is preferred** (typed, compact, fast); **CSV** is
supported for portability. If both exist for a root, parquet wins.

> **Parquet requires a parquet engine** — `pip install pyarrow` (bundled in the
> `[dev]` extra). **CSV needs no extra dependency**, so it is the engine-free
> default if you'd rather not install `pyarrow`.

Each file is a tidy, **date-ascending** table. Required and optional columns:

| Column | Required | Meaning |
|---|---|---|
| `date` | ✅ | Trading date (`YYYY-MM-DD`, or anything `pandas.to_datetime` parses) |
| `open` | ✅ | Back-adjusted open |
| `high` | ✅ | Back-adjusted high |
| `low` | ✅ | Back-adjusted low |
| `close` | ✅ | **Back-adjusted close** (the roll-adjusted continuous series) |
| `volume` | optional | Contract volume (`0`/blank tolerated) |
| `open_interest` | optional | Open interest (kept on the frame, not contract-checked) |

**Continuous, back-adjusted series only.** The file's `close` is treated as the
*adjusted* close: the source emits `adjusted_close == close`, and reports no
splits/dividends, so `DataLoader.prices` passes the roll-adjusted series through
unchanged. Do **not** supply individual delivery months here — supply one
stitched continuous series per root (Panama/back-adjusted or ratio-adjusted, per
your house convention).

### Header aliases (so vendor exports need no editing)

Common vendor spellings are auto-mapped onto the canonical schema, so a raw
Norgate/CSI/IB export usually loads as-is:

```
Date/DATE            -> date
Open/High/Low/Close  -> open/high/low/close
Settle/settle        -> close           (CSI/Norgate settlement price)
Volume/Vol/vol       -> volume
OpenInterest/OI/oi   -> open_interest
```

### Example (`CL.csv`)

```csv
date,open,high,low,close,volume,open_interest
2020-01-02,61.20,61.90,60.50,61.18,520000,2100000
2020-01-03,63.00,64.10,62.50,63.05,610000,2105000
```

The committed test fixtures under
[`tests/fixtures/futures/`](../tests/fixtures/futures/)
(`GC.parquet`, `CL.parquet`, `NG.csv`) are tiny, valid examples of both formats,
including the vendor-style `Settle`/`Vol`/`OI` headers.

## Directory layout

```
<your-futures-dir>/
├── GC.parquet        # Gold (COMEX), continuous back-adjusted
├── CL.parquet        # WTI Crude (NYMEX)
├── NG.csv            # Henry Hub Natural Gas (NYMEX)
├── ZC.parquet        # Corn (CBOT)
└── …                 # one file per continuous contract
```

The path is yours to choose (kept **out of the repo** — futures data is the
licensed property of whoever supplies it). A natural home alongside the existing as-of cache is
`data_cache/futures/`, but any directory works.

## Symbol convention

Continuous-futures symbols carry the framework's **`.COMM` pseudo-exchange
suffix** (e.g. `CL.COMM`), so they never collide with an EODHD equity code. The
file is named for the **root only** (`CL.parquet`); `CL` and `CL.COMM` both
resolve to it.

## Usage

```python
from rigor.data import DataConfig, DataLoader, FuturesDataSource

src = FuturesDataSource("data_cache/futures")          # your exported files
dl  = DataLoader(DataConfig.from_env(as_of="2026-06-01"), source=src)

px = dl.prices("CL.COMM")        # continuous WTI crude, back-adjusted OHLCV
print(src.available_symbols())   # which contracts are actually present
```

`FuturesDataSource` satisfies the full `DataSource` protocol, so it drops
straight into `DataLoader` in place of the EODHD client. It is **point-in-time
safe** (bars after the as-of date are dropped, like the EODHD client) and
**contract-checked** (each file is validated once on load with
`validate_ohlcv`, so a corrupt continuous series fails loudly with ticker + date
+ rule). Methods that have no meaning for futures
(`splits`/`dividends`/`earnings`/`fundamentals`/`funding`/`fred`/`intraday`/
`historical_constituents`) return empty/neutral defaults.

## Commodity-futures universe

The continuous contracts the commodity-momentum strategies target
are exposed as a constant + helper:

```python
from rigor.data import commodity_futures_universe

commodity_futures_universe()            # ['GC.COMM', 'SI.COMM', 'CL.COMM', …]
commodity_futures_universe(with_meta=True)
# [{'symbol': 'GC.COMM', 'root': 'GC', 'description': 'Gold (COMEX)',
#   'etf_proxy': 'GLD'}, …]
```

| Symbol | Root | Contract | ETF proxy |
|---|---|---|---|
| `GC.COMM` | GC | Gold (COMEX) | GLD |
| `SI.COMM` | SI | Silver (COMEX) | SLV |
| `PL.COMM` | PL | Platinum (NYMEX) | PPLT |
| `PA.COMM` | PA | Palladium (NYMEX) | PALL |
| `HG.COMM` | HG | Copper (COMEX) | CPER |
| `CL.COMM` | CL | WTI Crude Oil (NYMEX) | USO |
| `BZ.COMM` | BZ | Brent Crude Oil (ICE) | BNO |
| `NG.COMM` | NG | Henry Hub Natural Gas (NYMEX) | UNG |
| `ZC.COMM` | ZC | Corn (CBOT) | CORN |
| `ZW.COMM` | ZW | Wheat (CBOT) | WEAT |
| `ZS.COMM` | ZS | Soybeans (CBOT) | SOYB |
| `SB.COMM` | SB | Sugar No. 11 (ICE) | CANE |

`ETF_PROXY_TO_FUTURES` maps each ETF proxy to its continuous-futures symbol, so a
commodity strategy that loads `GLD`/`USO`/… today can be re-pointed at this
source once real futures files exist. **This is the on-ramp to the ~8 set-aside
futures strategies** — they go live as soon as continuous data is supplied
in this format.

## Exporting from a vendor

The source consumes the schema above; produce it once per continuous contract.

- **Norgate Data** (`norgatedata`): request a **continuous** series
  (`%-adjusted` / back-adjusted), `pandas-dataframe` format, then write
  `date, open, high, low, close, volume` (rename `Open`/`High`/… — the aliases
  handle the capitalised headers either way) to `<ROOT>.parquet`.
- **CSI (UA/Unfair Advantage)**: export the back-adjusted continuous contract to
  CSV. Its `Settle`/`Vol`/`OI` headers map straight onto `close`/`volume`/
  `open_interest`; just ensure a `Date` column and one file per root.
- **Interactive Brokers** (`ib_insync` / TWS API): request `CONTFUT` historical
  bars (`BID_ASK`/`TRADES`, `1 day`), then write the same columns. IB returns
  unadjusted continuous prices — apply your house roll/back-adjustment before
  writing so `close` is the back-adjusted series.

## Honest scope

No futures vendor is bundled or invented. The connector is **ready**; **real
futures data is yours to provide.** Drop your continuous files into a
directory in this schema and the ~8 set-aside futures strategies have their data
feed.
