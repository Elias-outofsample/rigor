# Advanced analysis layer (`rigor.analysis`)

A **purely additive** package of governance / attribution / rigour tools that sit
on top of `rigor.metrics` and `rigor.validation`. Nothing in the core data → engine →
metrics → validation path imports it, so a strategy that doesn't use it is
completely unaffected — `import rigor` and every strategy keep working whether or
not you install the optional extras below.

```python
from rigor.analysis import risk, signal_quality, factor, tail_risk, regime, verdict
```

## Install

The core analysis needs **nothing beyond the base install** (numpy/scipy/pandas).
A few heavier capabilities are gated behind an optional extra:

```bash
pip install -e ".[analysis]"   # statsmodels, arch, hmmlearn, pandas-datareader
```

Without the extra, the affected functions **degrade gracefully** — they either
fall back to a transparent method or return a clear `error`/`fallback_used` flag.
They never raise an import error at module load, so CI and a bare
install stay green.

| Capability | Needs extra? | Without it |
|---|---|---|
| risk battery, labeling, look-ahead, signal quality, structural breaks, predictive ability, attribution, exposure, crisis, edge classifier, verdict, cost/capacity, EVT, VaR backtests, factor regression (+ HAC), Bayesian Sharpe, Romano-Wolf DSR | no | — works on base install |
| `tail_risk.gjr_garch_fhs_cvar` (GARCH-FHS ES) | `arch` | historical-CVaR fallback (`fallback_used=True`) |
| `regime.detect_regimes_hmm*` (HMM regimes) | `hmmlearn` | returns `{"error": "hmmlearn not installed"}`; use the rule-based regimes instead |
| `factor.load_french_factors` (Ken French download) | `pandas-datareader` + network | returns an empty frame — pass your own factor panel |

## What's in it (the 13 capabilities)

| Module | What it answers |
|---|---|
| `verdict` | **Unified 5-pillar verdict** [0-100] → ROBUST/MODERATE/FRAGILE/OVERFIT; missing inputs are *neutral* (dropped from num + denom), not auto-fails. |
| `factor` | **Factor attribution** — CAPM→FF3→FF5→FF6→FF9 regressions, Newey-West HAC t-stats (pure-numpy, no statsmodels needed), progressive alpha + absorbed-%, VIF, sector residualisation. |
| `alpha_robustness` | Frequency-invariance, regime-conditional alpha, missing-factor PCA, crisis stress, and the **GENUINE_ALPHA / RISK_PREMIUM / MOMENTUM_FACTOR / MIXED** verdict (Harvey-Liu-Zhu t threshold). |
| `edge_classifier` | **Where does the edge come from?** mean-reversion / momentum / short-vol / convexity / carry / microstructure / residual-alpha, with per-universe thresholds. |
| `cost` | **Realistic cost & capacity** — spread + Almgren-Chriss impact + borrow + margin → net Sharpe; participation-limit & impact-threshold AUM capacity. |
| `tail_risk` | **EVT** (GPD tail, Hill/Pickands/Moment), **VaR backtests** (Kupiec POF + Christoffersen CC), **GJR-GARCH-FHS** conditional Expected Shortfall. |
| `signal_quality` | Rank-IC, ICIR, decay half-life, OU half-life, variance ratio, **Fundamental Law** (IC·√breadth·TC), **forecast-error metrics** (`forecast_mae`/`forecast_rmse`/`forecast_mape` — for a numeric return forecast vs realised). |
| `risk` | VaR/CVaR — historical, **Gaussian delta-normal** (`gaussian_var`) and Cornish-Fisher — Omega, Ulcer/UPI, tail ratio, Rachev, CDaR, Sterling, Burke, drawdown events, time-under-water. |
| `labeling` | Triple-barrier labels, meta-labels, uniqueness sample weights, sequential bootstrap (AFML). |
| `lookahead` | Shift-the-signal **look-ahead leak detector**. |
| `structural` | CUSUM mean/variance breaks, edge decay, ICSS, Masters block-permutation & optimisation-bias. |
| `predictive_ability` | Diebold-Mariano, Giacomini-White, Model Confidence Set, Pesaran-Timmermann, stochastic dominance. |
| `attribution` | **Signal-vs-ranker** decomposition, Memmel correlated-Sharpe test, Brinson-Fachler. |
| `exposure` | Hidden vol/gamma/theta exposure, option-payoff classification, **crisis-window Sharpe** decomposition. |
| `regime` | Leak-free walk-forward **HMM regimes** + rule-based (SMA/VIX/realised-vol/6-bucket) fallbacks, per-regime performance, Bai-Perron breaks. |

`rigor.validation` also gained `bayesian` (Bayesian Sharpe posterior + deploy/wait/reject)
and `overfit.deflated_sharpe_romano_wolf` (empirical-E[max] DSR via stationary bootstrap).

## Tests

`tests/test_analysis_wave1.py … wave3.py` cover every module on synthetic
data and assert the optional-dependency paths fall back cleanly. Run them with the
rest of the suite: `python -m pytest`.
