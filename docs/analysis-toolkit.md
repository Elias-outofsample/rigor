# Analysis toolkit (`rigor.analysis`)

`rigor.analysis` is a **deliberate, opt-in analyst toolkit**. It is *purely
additive*: nothing in the core data/engine/metrics/validation path imports these
modules, so a strategy that never calls them runs exactly the same. They exist to
be reached for **on purpose** during a deep-dive on a candidate — "where does the
alpha come from?", "will it survive?", "what breaks it first?" — not to run on
every backtest.

Because they have no production call-site, they are easy to mistake for dead
code. They are not: **every module below is exercised by a unit test** (see the
"Test" column), and this page is the index so they stay discoverable rather than
silently rotting. The opt-in design is intentional — see
[`design-decisions.md`](design-decisions.md).

## How to call any of them

All are plain functions/classes — import the module and call. None mutate global
state; all are offline and deterministic unless they explicitly fetch factor data.

```python
from rigor.analysis import capture, survival, attribution   # etc.

caps = capture.compute_capture_ratios(returns, benchmark)
surv = survival.compute_survival(...)
```

Heavier modules need the optional extras (`pip install -e ".[analysis]"`:
statsmodels, arch, hmmlearn, pandas-datareader). Without them those functions
degrade gracefully or raise a clear install message; everything else still works.

---

## The opt-in analyst modules

These ~10 have **no production call-site by design** — the counter-audit flagged
them as "orphaned"; they are the analyst toolkit. Each is unit-tested.

| Module | What it computes | Entry point | Test |
|---|---|---|---|
| `alpha_robustness` | Separates idiosyncratic alpha (skill) from factor risk-premium: frequency-invariance, regime-conditional alpha, missing-factor PCA, crisis stress; emits a `GENUINE_ALPHA / RISK_PREMIUM / ...` verdict. | `alpha_vs_risk_premium_verdict(progressive_alpha, ...)` | `test_analysis_wave3.py` |
| `attribution` | Splits return between the *entry signal* and the *ranker/SetupScore* (Brinson contribution + Memmel Sharpe-difference tests) — catches a "mean-reversion" edge that is really a vol ranker. | `decompose_signal_vs_setup(...)` | `test_analysis_wave2.py` |
| `capture` | Morningstar up/down capture ratios plus a stress-tail capture with a `crash_convex` fragility flag — distinct from beta. | `compute_capture_ratios(returns, benchmark)` | `test_capture.py` |
| `cost` | Turns a gross backtest net: spread + Almgren-Chriss impact + commission + borrow/financing, and AUM capacity ceilings. | `compute_realistic_costs(...)` | `test_analysis_wave3.py` |
| `edge_classifier` | Maps a diagnostic vector (signal/factor/exposure/crisis) to a likely *edge source*, scored and ranked. | `classify_edge_source(diag, universe=...)` | `test_analysis_wave3.py` |
| `labeling` | Lopez de Prado triple-barrier labels, meta-labels, label-uniqueness sample weights and sequential bootstrap (corrects overlapping-label inflation). | `triple_barrier_labels(close, events, ...)` | `test_analysis_wave1.py` |
| `predictive_ability` | Forecast-skill tests: Diebold-Mariano, Giacomini-White, Hansen-Lunde-Nason model-confidence-set, Pesaran-Timmermann, stochastic dominance. | `diebold_mariano(loss_a, loss_b, ...)` | `test_analysis_wave2.py` |
| `scenario_response` | Conditions returns on interpretable regimes (trend/vol/drawdown buckets) and reports per-regime return/hit-rate/beta + a behavioural verdict (defensive vs directional). | `scenario_response(...)` | `test_scenario_response.py` |
| `structural` | Locates *when* an edge broke: Page CUSUM (mean), CUSUM-of-squares (variance), ICSS multiple breaks, rolling-Sharpe decay, Masters block-permutation + optimisation-bias tests. | `cusum_mean(returns, ...)` | `test_analysis_wave2.py` |
| `survival` | Forward-durability prior in `[0,1]` from crowding + life-cycle phase + edge-decay — answers "will this still work in five years?". | `compute_survival(...)` | `test_survival.py` |

## The rest of the layer (same opt-in contract)

These complete the toolkit; a few are also re-exported and used by the report/
thesis surface (`diagnostics`, `quality`, `verdict`, `kelly`), but the contract is
identical — call them deliberately.

| Module | What it computes | Test |
|---|---|---|
| `diagnostics` | Report-ready aggregate (`strategy_diagnostics`) over tail risk, risk battery, crisis decomposition, capacity, options-exposure, factor attribution; every block degrades gracefully. | `test_diagnostics.py`, `test_diagnostics_extra.py` |
| `distribution` | Return-distribution/stationarity battery: Jarque-Bera, runs test, Lo-MacKinlay variance ratio, Durbin-Watson, ARCH-LM, White hetero. | `test_distribution.py` |
| `entry_signal_ic` | Reconstructs candidate entry signals at each trade and measures which known anomaly (reversal/momentum/RSI2/IBS/...) the entries actually load on. | `test_entry_signal_ic.py` |
| `exposure` | Hidden option-like exposure (vol/gamma/theta) + per-crisis-window decomposition. | `test_analysis_wave2.py` |
| `factor` | Fama-French/Carhart factor regressions with Newey-West HAC t-stats and progressive (nested-model) alpha; VIF; sector-neutralisation. | `test_analysis_wave3.py` |
| `kelly` | Kelly position sizing in five flavours (full/half/fractional/continuous/multi-asset/shrinkage). | `test_kelly.py` |
| `lookahead` | Engine-agnostic look-ahead/causal-leakage detector via future-data perturbation. | `test_lookahead_detector.py` |
| `quality` | Compact four-level `WEAK-FRAGILE-MODERATE-ROBUST` badge with hard-cap logic (additive to `verdict`). | `test_quality.py` |
| `regime` | Leak-aware regime detection (rolling HMM, SMA/VIX rules) + per-regime performance + Bai-Perron breaks. | `test_analysis_wave3.py` |
| `risk` | Risk/drawdown ratios beyond `rigor.metrics`: VaR/CVaR, Omega, Ulcer/UPI, tail ratio, Rachev, CDaR, Sterling, Burke + drawdown-event extraction. | `test_risk_var.py` |
| `signal_quality` | Signal diagnostics (rank-IC/ICIR, Chatterjee xi, signal decay half-life, OU half-life, net-IC, monotonicity). | `test_signal_decay.py`, `test_xi_correlation.py` |
| `tail_risk` | EVT (GPD tail, Hill/Pickands), VaR backtests (Kupiec, Christoffersen), GJR-GARCH-FHS conditional CVaR. | `test_risk_var.py` |
| `torture` | Adversarial "what breaks this edge first?" score aggregating the robustness battery into ranked weaknesses. | `test_torture.py` |
| `verdict` | Unified 5-pillar deployment verdict (0-100) over overfit/stability/consistency/significance/viability. | `test_verdict_presets.py` |

> Coverage note: each module above is referenced by at least one unit test
> (verified by import-grep over `tests/`). No analyst module is
> currently lacking a test. If you add a module here, add its test in the same
> change and a row to this table.
