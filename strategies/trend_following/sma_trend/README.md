# SMA Trend (SPY)

- **Slug:** `sma_trend`
- **Category:** trend_following

## Thesis
_Why should this edge exist? What inefficiency or risk premium does it harvest?_

## Universe & data
_Symbols / index, date range, any filters. Data is single-source EODHD via the
DataLoader (point-in-time, deterministically adjusted)._

## Parameters
_List the `param_grid` axes and what each controls._

## Results
_Generated artifacts live in `artifacts/` (committed). Regenerate with:_

```
python -m rigor run strategies/trend_following/sma_trend --as-of <YYYY-MM-DD>
```
