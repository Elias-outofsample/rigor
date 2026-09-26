# Rigor Strategy Standard

Every strategy in the book has the **same shape**, the **same files in the same
order**, and produces the **same report card** — so anyone can open any
strategy and know exactly where everything is.

## Folder layout

Strategies are grouped by edge family, with a `snake_case` slug. There are two
homes — the **production** version and its raw **baseline** — both using the
identical folder shape:

```
strategies/
├── <category>/<slug>/          # PRODUCTION — the current, evolved version
│   ├── strategy.py             #   StrategyBase subclass + build(config, data) factory
│   ├── README.md               #   thesis, universe/data, parameters, results
│   ├── config.json             #   declarative StrategyConfig (name, dates, costs, version, …)
│   └── artifacts/              #   GENERATED — committed (results ship with the strategy)
│       ├── <slug>_returns.csv  #     daily net returns
│       ├── <slug>_summary.json #     canonical metrics
│       ├── <slug>_report.html  #     the report card
│       ├── <slug>_thesis.pdf   #     the thesis (generated on every run)
│       └── <slug>_positions.csv #    position ledger (per-trade incl. MAE/MFE)
│
└── Baseline/<category>/<slug>/  # BASELINE — the raw version, same shape, same slug
```

**Report-card / `summary.json` metrics.** Every run computes the same canonical
block (via `rigor.metrics`): Sharpe, CAGR, **Expected Value** (`ev_annual`, the
arithmetic expected return), **Excess Return** (`ev_excess_annual` = expected
return − the period risk-free, 3-month T-bill / FRED `DGS3MO`; falls back to
rf = 0 if the series is unavailable), Max Drawdown, Volatility, **Gaussian VaR**
(`var_normal_95`, the 95% delta-normal value-at-risk), Sortino, Calmar, Total
Return, Win Rate. The **position ledger** (`<slug>_positions.csv`) additionally
records each trade's **MAE** (Maximum Adverse Excursion — worst unrealised loss
held) and **MFE** (Maximum Favorable Excursion). For numeric *forecast* signals,
`rigor.analysis.signal_quality` offers forecast-error metrics (MAE/RMSE/MAPE).

The **slug** is the one identifier: it names the folder and prefixes every
artifact. It must be `snake_case`, start with a letter, and match the folder name.

**Categories** (canonical): `trend_following`, `momentum`, `mean_reversion`,
`breakout`, `carry`, `seasonal`, `volatility`, `stat_arb`, `macro`,
`event_driven`, `crypto`, `intraday`, `other`.

## Strategy vs. version vs. baseline

- A change to the **core signal** is a **different strategy** → a different slug →
  its own folder. Example: ranking by ATR instead of volatility is a *different*
  strategy (`momentum_atr` vs `momentum_vol`), each with its own baseline.
- Adding a **filter** or **optimization** is a new **version of the same
  strategy** — same folder, bump the `version` field (`v1` → `v2`). Git history
  holds the trail; the folder always holds the *current* version.
- Every strategy's **raw form — before any filter or optimization — lives under
  `strategies/Baseline/<category>/<slug>/`**, using the **same slug** as its evolved
  counterpart. Baselines carry version `v0` and are **excluded from the catalog**.
  Scaffold one with `rigor new <category> <slug> --baseline`. So the two homes are
  always: the **baseline → `strategies/Baseline/...`**, the **optimized/evolved
  version → `strategies/...`**.
- **Intraday** strategies use the `intraday/` category like any other edge family:
  the evolved version goes in `strategies/intraday/<slug>/` and its baseline in
  `strategies/Baseline/intraday/<slug>/`.

## Not-yet-ready strategies: keep them local with a `NEEDS.md`

A strategy goes in `strategies/` **only when it reproduces and passes the gates**. If
it is missing something — a data source we don't have, a framework feature, a fix
for a look-ahead, or a real edge it doesn't yet show — it is **not committed**. Keep it **local**, in your own gitignored staging area —
name the folder whatever you like and exclude it via your personal
`.git/info/exclude` or a global gitignore (it never goes in the shared
`.gitignore`, since everyone names theirs differently) — with a **`NEEDS.md`** in
its folder that states exactly what is blocking it.

This is a **book-wide rule**: every strategy kept local because it's missing
something carries a `NEEDS.md`, so anyone can see at a glance what each one is
waiting on, and nothing half-finished leaks into the catalog. Copy the template at
the template below. Delete the `NEEDS.md` and promote the strategy to
`strategies/` (with its baseline under `strategies/Baseline/`) once the blocker is
resolved and it reproduces.

## The ledger rule: return the full BacktestResult

**Required output:** `<slug>_positions.csv` — per-trade entry/exit with
**MAE** (Maximum Adverse Excursion) and **MFE** (Maximum Favorable Excursion).

The runner writes this automatically, but only when `run_backtest` returns the
full `BacktestResult` (not just `.returns`):

```python
# CORRECT — runner writes positions ledger automatically
def run_backtest(self, cache: dict, params: dict) -> BacktestResult:
    ...
    return simulate_weights(weights, prices, commission_bps=self.config.commission_bps)

# WRONG — loses the ledger; positions CSV will be absent or degenerate
def run_backtest(self, cache: dict, params: dict) -> pd.Series:
    ...
    return simulate_weights(weights, prices, ...).returns   # don't do this
```

`simulate_weights` returns a `BacktestResult` that carries `weights` and
`asset_returns`. The runner uses those to write the per-trade ledger. When you
return only `.returns`, that context is lost and the ledger cannot be
reconstructed. **Audit will WARN (`MISSING_LEDGER`)** today and **ERROR after
the migration window** (see `docs/quality-gates.md`).

### Event-driven strategies: use TradeLog

For strategies with discrete entry/exit events (not a continuous weights
matrix), record trades directly into `rigor.engine.TradeLog`:

```python
from rigor.engine import TradeLog, result_from_returns

class Strategy(StrategyBase):
    def run_backtest(self, cache, params) -> BacktestResult:
        log = TradeLog()
        # ... signal logic ...
        log.record(entry_date, exit_date, symbol, entry_price, exit_price)
        returns = ...  # derive from log
        return result_from_returns(returns, index=dates, trade_log=log)
```

### Opting out: `ledger_exempt`

For strategies where per-trade MAE/MFE is semantically meaningless (e.g.
delta-neutral carry strategies that are always fully invested), set in
`config.json`:

```json
{
  "ledger_exempt": true
}
```

This suppresses `MISSING_LEDGER` and `DEGENERATE_LEDGER`. Use sparingly.

## The three source files

1. **`strategy.py`** — a `Strategy(StrategyBase)` subclass implementing
   `build_cache` (data via the `DataLoader`), `run_backtest` (signal → target
   weights → `simulate_weights`), and `param_grid`; plus a module-level
   `build(config, data)` factory the runner calls.
2. **`config.json`** — the declarative header: `name`, `slug`, `category`,
   `start_date`, `end_date`, `commission_bps`, `long_short`, `rebalance_freq`,
   `role`, **`status`** (`live` / `paper` / `idle` — deployment state, default
   `idle`), **`version`** (current version this folder holds; `v0` = baseline,
   `v1`+ = production), `ledger_exempt` (optional bool, default false), and
   `extra` (e.g. `{"symbol": "SPY"}`).
3. **`README.md`** — thesis, universe/data, parameters, how to run.

## The catalog

Every `rigor run` (and `rigor index`) refreshes **`catalog/INDEX.md`** — a
committed, shared table of every **production** strategy with its category,
`status`, `version`, headline metrics, and validation verdict. Baselines are
excluded (working strategies only). It's the quick "what do we have" view; a `git
pull` always shows the current book.

## Workflow (CLI)

```bash
# 1. scaffold a conforming folder
python -m rigor new mean_reversion buy_the_dip --name "Buy The Dip"

# 2. edit strategy.py + config.json, then check it conforms
python -m rigor validate strategies/mean_reversion/buy_the_dip

# 3. backtest -> writes the artifacts, pinned to a shared as-of date
python -m rigor run strategies/mean_reversion/buy_the_dip --as-of 2026-06-01
```

## Optimisation (opt-in)

> This section is the quick reference (how to run it + every flag). For the full
> explanation — the pipeline stage by stage, **how to read the results**, a worked
> example, and the architecture — see **[optimization.md](optimization.md)**.

Tuning is **opt-in**: a strategy joins the optimiser by declaring a multi-value
`param_grid()` (e.g. `{"lookback": [20, 50, 100], "stop_atr": [1.5, 2.0]}`). Strategies
without one just run as a single config — nothing changes.

```bash
python -m rigor optimize strategies/mean_reversion/buy_the_dip --as-of 2026-06-01
```

`rigor optimize` searches the grid (the data cache is built once and reused), ranks the
configurations, and reports the **best** one — graded by the *same* validation verdict any
single backtest gets (no parallel scoring), plus the search-specific anti-overfit machinery:

- **Deflated Sharpe + CPCV-PBO** — the winner's Sharpe deflated for the number of configs
  searched, and the probability the ranking is overfit (Combinatorially-Purged CV).
- **Data-snooping tests** (`rigor.validation.permutation`) — White Reality Check, Hansen SPA
  and Romano-Wolf StepM: does the best of the searched family beat the **untuned default**
  once you account for having searched the whole family, using the configs' actual
  correlation (a stationary bootstrap) rather than assuming independence?
- **Walk-forward optimisation** — re-selects the best params on each in-sample fold and
  measures them out-of-sample (`--wf-mode expanding|rolling`), so you see whether the
  *search itself* generalises.

Optional modes (all off by default):

- `--select robust` — instead of top Sharpe, keep only configs that clear robustness floors
  (Sharpe, PSR, no wipe-out), reject the grid outright if CPCV-PBO is too high, then maximise
  performance among the survivors. Returns `REJECTED` if nothing is robust — the honest answer.
- `--two-pass` — coarse→fine search: thin each axis to a coarse grid, find the winner's
  neighbourhood, then re-expand only that neighbourhood at full resolution. Visits a fraction
  of a large grid, so big sweeps stay tractable on modest hardware.
- `--ensemble N` — also report an equal-weight ensemble of `N` diverse top configs; if the
  ensemble Sharpe holds up near the single winner's, the edge is broad rather than a lucky point.

**The full research pipeline** (`--full`, or pick stages individually) adds the deeper
diagnostics ported from the prior research library — opt-in because they add compute:

- `--enrich` — stage-3 robustness scalars on the winner (ROC310 temporal decay, Spearman
  stability, K-ratio, drawdown-recovery, PSR, Ulcer Performance Index, rolling-Sharpe
  consistency, Sortino, recency).
- `--holdout` — a never-optimised final-20% slice; flags the winner if its holdout Sharpe
  collapses vs the out-of-sample window.
- `--falsify` — noise-injection *fragility* (how fast Sharpe decays under added noise) and,
  when a price-resimulation hook is available, a phase-randomised surrogate-price test.
- `--score` — the 5-pillar V4.2 composite selection score across the grid (risk-adjusted /
  drawdown / stability / consistency / tail, weighted geometric mean).
- `--regime` — the best config *within* each volatility regime, plus a consensus.
- `--cluster` — clusters the top configs in parameter space (how many distinct good regions).

`rigor optimize <path> --full` runs them all. The verdict you already trust still grades the
winner; these stages are reported alongside it, never replacing it.

**Compute backend** (`--backend auto|numpy|numba|cuda`). The bootstrap-heavy stages (data
snooping, falsification) accelerate on Numba (CPU JIT) or CUDA (GPU) when installed; `auto`
picks the fastest available and falls back to NumPy. **Results are identical across backends**
— the random draws are generated deterministically and only the arithmetic is accelerated, so
the backend is a speed knob, never a results knob. The accelerators are optional extras and
never required:

```bash
pip install -e "Framework[accel]"        # Numba (CPU) — ~1.5-2x on the bootstrap stages
pip install -e "Framework[accel-cuda]"   # + CuPy (NVIDIA GPU); match the wheel to your CUDA
```

It is **non-destructive**: it writes a local `artifacts/<slug>_optimization.json` report
(not committed) and never edits your `config.json`. If you decide to adopt the winning
params, update the config yourself and **bump the `version`** (it's a new version of the
same strategy — see "Strategy vs. version vs. baseline").

### Optional in the framework, mandatory on your machine if you want

Optimisation is **compute-heavy and never mandatory** — it is a separate `rigor optimize`
command, never part of `rigor run`, so nothing in the shared framework or CI requires it.
You run it when you choose to.

If *you* want it enforced **on your own machine** (e.g. you must run it before pushing a
strategy you tuned), install the optional, **local-only** git hook — it lives in
`.git/hooks/` and is never committed, so it affects only you and breaks nothing for anyone
else:

```bash
cp scripts/hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push   # enable
rm .git/hooks/pre-push                                                          # disable
```

The hook runs `rigor optimize` on any strategy you're pushing that has a `param_grid` and
blocks the push if it errors (set `RIGOR_OPTIMIZE_BLOCK_OVERFIT=1` to also block on a REJECT
verdict). Bypass it once with `git push --no-verify`. Clones without the hook are
unaffected — the framework stays optional for everyone by default.

## Data & reproducibility rules

- **Live data by default** — each machine fetches with its own key. Pin `--as-of`
  for any committed/shared result: it prevents look-ahead and keeps the catalog's
  metrics stable (live data drifts slightly as the vendor restates bars, so runs
  are close but not byte-identical — that's expected). For *exact* reproduction,
  freeze a snapshot and run `--offline` (opt-in; see [data.md](data.md)).
- `artifacts/` is generated **and committed** — each strategy ships its results
  (returns/summary/report/thesis) so they're visible on GitHub. Re-run `rigor run` to
  regenerate them from the same as-of.
- Data is single-source EODHD via the `DataLoader` (point-in-time, deterministic
  adjustment). Metrics come only from `rigor.metrics`. Never compute either inline.
