# Notice

`rigor` is an original framework, but a few parts were adapted from an earlier in-house
research library rather than written from scratch:

- the `StrategyBase` / `StrategyConfig` contract (the `build_cache` / `run_backtest` /
  `param_grid` pattern) and some metric conventions;
- several cross-strategy structural checks in `rigor.project.audit` and the calibration
  constants of `rigor.analysis.torture`;
- the API shape of the Portfolio Cockpit service (`cockpit/cockpit_api`), re-implemented
  over `rigor`'s own engine;
- the HTML report's visual theme.

The PDF thesis generator (`rigor.thesis`) was vendored from an earlier standalone tool.

Academic methods are credited where they are implemented, notably: Bailey & López de Prado
(probabilistic and deflated Sharpe ratio, backtest overfitting / CSCV), López de Prado
(purged and embargoed cross-validation, CPCV), Harvey, Liu & Zhu (multiple-testing
t-statistics), White (Reality Check), Hansen (SPA test), Romano & Wolf (StepM), and the four
papers of the VWAP case study (Zarattini & Aziz; Bhatti; Grant, Wolf & Yu; Lee).
