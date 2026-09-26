"""Templates the scaffolder writes for a new strategy."""

from __future__ import annotations

import json

STRATEGY_PY = '''\
"""{name} ({slug}) — {category}.

Implements the Rigor strategy contract:
  build_cache  -> data/indicators via the DataLoader (single-source EODHD, as-of)
  run_backtest -> signal -> target weights -> simulate_weights (standard accounting)
  param_grid   -> the parameter sweep (REQUIRED, multi-value; optimize_gate-enforced)

The module-level ``build(config, data)`` factory is what the runner calls.
"""

from __future__ import annotations

import pandas as pd

from rigor.engine import BacktestResult, simulate_weights
from rigor.strategy import StrategyBase, StrategyConfig


class Strategy(StrategyBase):
    def __init__(self, config: StrategyConfig, data) -> None:
        super().__init__(config)
        self.data = data
        self.symbol = config.extra.get("symbol", "SPY")

    def build_cache(self) -> dict:
        px = self.data.prices(
            self.symbol, start=self.config.start_date, end=self.config.end_date
        )
        px = px.set_index(pd.DatetimeIndex(px["date"]))
        return {{"dates": px.index, "close": px["adj_close"].astype("float64")}}

    def param_grid(self) -> dict:
        # REQUIRED — the optimize_gate (blocking CI) rejects a strategy that has
        # no sweepable grid. Replace the placeholder with THIS strategy's real
        # tunable parameters: >=1 axis with >=2 values, type-consistent per axis,
        # and every key must be read from `params` in run_backtest below. Do NOT
        # return an empty or single-value grid. If the strategy genuinely has no
        # tunable surface (a fixed calendar pattern), delete this method and set
        # `optimize_exempt_reason` in config.json instead. See docs/quality-gates.md.
        return {{"sma_window": [100, 200]}}

    def run_backtest(self, cache: dict, params: dict) -> BacktestResult:
        # TODO: replace with this strategy's real signal -> target weights.
        #
        # IMPORTANT: return the full BacktestResult (i.e. `simulate_weights(...)`)
        # NOT just `simulate_weights(...).returns`.  The runner uses the full result
        # to write the per-trade position ledger (MAE/MFE) automatically.
        close = cache["close"]
        window = int(params["sma_window"])
        weight = (close > close.rolling(window).mean()).astype("float64")
        weights = pd.DataFrame({{self.symbol: weight}}, index=close.index)
        prices = pd.DataFrame({{self.symbol: close}}, index=close.index)
        return simulate_weights(
            weights, prices, commission_bps=self.config.commission_bps
        )


def build(config: StrategyConfig, data) -> Strategy:
    return Strategy(config, data)
'''

README_MD = '''\
# {name}

- **Slug:** `{slug}`
- **Category:** {category}

## Thesis
_Why should this edge exist? What inefficiency or risk premium does it harvest?_

## Universe & data
_Symbols / index, date range, any filters. Data is single-source EODHD via the
DataLoader (point-in-time, deterministically adjusted)._

## Parameters
_List the `param_grid` axes and what each controls. Every strategy must declare a
sweepable grid (>=1 axis, >=2 values) so it is optimisable in-house and by the
external optimiser — the `optimize_gate` enforces this. A strategy with no tunable
surface (e.g. a fixed calendar pattern) sets `optimize_exempt_reason` in config.json
instead, and says why here._

## Results
_Generated artifacts live in `artifacts/` (gitignored). Run:_

```
python -m rigor run {run_path} --as-of <YYYY-MM-DD>
```
'''


def config_json(name: str, slug: str, category: str, *, version: str = "v1") -> str:
    cfg = {
        "name": name,
        "slug": slug,
        "category": category,
        "start_date": "2005-01-01",
        "end_date": None,
        "initial_capital": 200000.0,
        "commission_bps": 10.0,
        "long_short": "long",
        "rebalance_freq": "daily",
        "role": "alpha",
        "status": "idle",
        "version": version,
        "extra": {"symbol": "SPY"},
    }
    return json.dumps(cfg, indent=2)
