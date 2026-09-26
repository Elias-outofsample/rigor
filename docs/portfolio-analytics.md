# Portfolio analytics & strategy diagnostics

The Section-E additions: deeper risk attribution and allocation diagnostics in the
Portfolio Cockpit, and a deep-diagnostics pack on every strategy's report card.
All additive, offline, and reusing the existing `rigor.analysis` engine — no new
system dependencies, nothing that changes how a strategy runs.

## Portfolio analytics (`cockpit/portfolio_engine`)

| Function | Module | What it answers |
|---|---|---|
| `marginal_risk_contributions(returns_matrix, weights)` | `portfolio_engine.risk` | **Euler risk attribution** — each leg's marginal & total risk contribution and % of portfolio risk. Flags a leg carrying materially more risk than its weight (10% weight, 40% risk). |
| `risk_hhi` / `risk_concentration_hhi` | `portfolio_engine.risk` | Herfindahl concentration on **risk shares** vs capital weights. |
| `autocorrelation_profile(returns)` | `portfolio_engine.risk` | ACF + Ljung-Box; flags return-smoothing / stale-price **leakage** (lag-1 ACF > 0.2). |
| `risk_budget(cov, target_budget)` | `portfolio_engine.weight_methods` | **SLSQP risk-budget solver** — hit an arbitrary target risk split, not just equal-risk parity. Also exposed via `allocate_weights(..., method="risk_budget", target_budget=...)`. |
| `portfolio_pbo(returns_matrix)` | `portfolio_engine.portfolio_validation` | **PBO on the allocation itself** — Dirichlet alt-allocations through `rigor.validation.cpcv`; treats the weights as an overfittable choice (ROBUST / BORDERLINE / OVERFIT). |
| `cardinality_search(returns_matrix)` | `portfolio_engine.portfolio_validation` | **"How many strategies?"** — the knee where a subset reaches 95% of the best Sharpe. |
| `stress_vs_calm_correlation(returns_matrix)` | `portfolio_engine.stress` | **False-diversification detector** — cross-leg correlation in stress vs calm regimes; flags diversification that evaporates in a crash. |
| `export_as_strategy(portfolio, dest_dir)` | `portfolio_engine.portfolio_io` | Write a portfolio as a conforming, **re-composable** `strategies/` folder (members + weights replayed via `result_from_returns`). Guarded never to write into the repo `strategies/` tree implicitly. |

These are library functions with full unit tests; they're available to the FastAPI
service and to any script importing `portfolio_engine`.

## Strategy report diagnostics (`rigor.analysis.diagnostics`)

Every strategy's **report card** (`report.html`) and **`summary.json`** now carry a
**Deep Diagnostics** block, computed from the strategy's own returns (offline):

- **EVT tail** — generalised-Pareto tail fit + Kupiec VaR-95 backtest.
- **Risk battery** — VaR/CVaR, Omega, Ulcer/UPI, tail ratio, Rachev, CDaR, Sterling, Burke.
- **Crisis decomposition** — per-crisis-window Sharpe + a crash-Sharpe verdict
  (robust-to-crises vs vol-risk-premium vs convexity).
- **Capacity** — Almgren-Chriss AUM capacity (low/central/high).
- **Vol-regime performance** — Sharpe / time-share in low/mid/high realised-vol regimes.
- **Monte-Carlo / bootstrap Sharpe CI** — seeded bootstrap CI on the Sharpe + P(SR>0).
- **Return persistence / edge decay** — rolling-Sharpe slope + lag-1..5 autocorrelation
  (flags return smoothing / a decaying edge).
- **Single-month concentration** *(blind-spot gate)* — flags when one month drives an
  outsized share of all-time return (WARN >20%, FAIL >30%).
- **Performance decay** *(blind-spot gate)* — first-half vs second-half Sharpe + slope
  across sub-periods (WARN >20%, FAIL >50% decay).
- **Correlation breakdown** *(blind-spot gate, when a benchmark is given)* — rolling
  strategy-vs-benchmark correlation; flags ≥3 regime shifts (|Δρ|>0.3).
- **Risk-DNA** (when a benchmark is available) — vol/gamma/theta option-exposure profile.
- **Factor regression** (when a factor panel is supplied) — CAPM→FF6 alpha with HAC t-stats.

The three blind-spot gates are **advisory** (surfaced in every report; they do not change
the promotion verdict). The top KPI strip also now carries **Expected Value** —
`ev_annual = mean(returns) × periods_per_year`, the *arithmetic* expected annual return
(complements CAGR's *geometric* mean), stored in `summary.json` as `ev` / `ev_annual`.

```python
from rigor.analysis import diagnostics
diag = diagnostics.strategy_diagnostics(returns)          # returns-only blocks
diag = diagnostics.strategy_diagnostics(returns, benchmark=spy, factors=ff)  # + Risk-DNA + factor
html = diagnostics.diagnostics_to_html(diag)              # report fragment
row  = diagnostics.diagnostics_summary(diag)              # JSON-safe headline scalars
```

The whole step is wrapped in the report card so a diagnostics failure can never
break a report — the section is simply omitted.

## Trade / position ledger (`rigor.project.trades`)

`build_position_ledger(result)` reconstructs per-symbol position spells
(entry/exit/holding-days/side/return-contribution) from a `BacktestResult`'s target
weights, and `rigor run` writes `artifacts/<slug>_positions.csv` when weights are
present. Note: the standard strategy contract returns a *returns series* (weights
are computed internally then discarded), so a strategy opts into the ledger by
returning the full `BacktestResult` from `run_backtest` — `StrategyBase.backtest()`
now passes that through unchanged (existing return-a-Series strategies are byte-for-byte
unaffected). The ledger populates automatically for exported portfolios and any
weight-returning strategy.

## Framework hardening (shared utilities & robustness)

Small, reusable additions that prevent recurring bug classes and tighten methodology:

- **`rigor.utils`** — shared helpers: `safe_divide`, `clip_safe`, `max_consecutive`;
  `NumpyEncoder` (numpy/pandas → JSON without crashes; non-finite → null) + `coerce_float`
  + `format_pct`/`format_number`; `align_series` / `align_frame` / `align_mask`
  (inner-join + NaN-drop so paired `.to_numpy()` can never silently mismatch lengths).
- **`rigor.xs`** — first-class cross-sectional primitives for future factor/basket strategies:
  `xs_daily_basket`, `monthly_xs_quintile_ls`, `xs_information_coefficient` — all respect a
  point-in-time `membership_mask` (no survivorship bias) and reuse the engine's 1-bar lag.
- **Cache schema versioning** — `rigor.data.cache` writes a `<blob>.meta.json` sidecar with
  `CACHE_SCHEMA_VERSION`; a version *mismatch* invalidates the entry (re-fetch / offline-miss),
  while a *missing* sidecar (legacy cache) is treated as current — so an upgrade never nukes
  the multi-GB cache, but a future format change can't silently serve stale-shaped data.
- **Cumulative `n_trials`** — `rigor optimize` persists `artifacts/<slug>_n_trials_log.json` and
  deflates the DSR by the *cumulative* trial count across repeated optimisation campaigns on the
  same strategy (honest multiple-testing accounting). Opt out with `cumulative_trials=False`.
- **Optimisation evidence is now committed** — `artifacts/<slug>_optimization.json`
  (CPCV / PBO / sensitivity / walk-forward) is no longer gitignored, so the anti-overfit
  evidence for an optimised strategy is visible on GitHub.
