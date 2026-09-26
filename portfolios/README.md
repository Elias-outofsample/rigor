# Portfolios

Saved multi-strategy portfolios built with the **Portfolio Cockpit** (`cockpit/`).
Committed and visible on GitHub, so a portfolio can be inspected — or rebuilt — from
any clone.

This is to portfolios what `strategies/` is to strategies: a reproducible home. Each saved portfolio is one folder:

```
portfolios/<name>/
├── portfolio.json     the definition: included strategies (by slug), weights,
│                      allocation mode, constraints, as-of date
└── artifacts/         generated — the combined portfolio's results
    ├── <name>_returns.csv     daily portfolio returns
    ├── <name>_summary.json    headline metrics (Sharpe, CAGR, MaxDD, beta, ...)
    └── <name>_report.html     the tearsheet
```

A portfolio references its strategies **by slug** (e.g. `sma_trend`), so it
always rebuilds from the current `strategies/` book. Re-open it in the app, or
regenerate its artifacts, any time.

Save a portfolio from the app's **Compare/Backtest** tab → it writes a folder here.
See **[docs/portfolio-cockpit.md](../docs/portfolio-cockpit.md)** for the app and the full standard.
