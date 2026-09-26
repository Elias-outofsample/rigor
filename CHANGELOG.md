# Changelog

## 1.0.0 — public release

- Package and CLI renamed `rigor`; repository reorganised around `src/`, `strategies/`,
  `catalog/`, `cockpit/` and `docs/`.
- `rigor demo`: offline end-to-end run on a zero-drift synthetic market — backtest,
  validation battery, 25-configuration optimisation, report card and thesis.
- Example book of sixteen strategies with committed artifacts; thesis PDFs regenerated
  from the committed returns.
- S&P 500 / Nasdaq-100 point-in-time membership made configurable
  (`RIGOR_REFERENCE_DIR`) since it is licensed data; the test suite uses a synthetic
  event log and a synthetic snapshot for the offline reproducibility test.
- Coverage floor raised to 75% (measured 79%, PDF generator excluded).

## 0.x — private development

Engine, metrics, report card, thesis generator, validation battery and verdict,
optimisation pipeline (deflated Sharpe, CPCV-PBO, data snooping, walk-forward, holdout,
falsification, regimes, clustering), analysis toolkit, quality gates, model registry,
Portfolio Cockpit.
