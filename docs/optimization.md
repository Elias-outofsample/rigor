# Optimisation Guide

How to tune a strategy's parameters **without fooling yourself**. This is the deep
dive — for the quick "how to run it + every flag", see the *Optimisation* section
of [strategy-standard.md](strategy-standard.md).

The optimiser's job is not "find the highest Sharpe". The highest Sharpe in a grid
is the **maximum of many noisy trials**, so it looks good partly by luck. The whole
point of this pipeline is to tell you **how much of the winner is real** — and, just
as often, to tell you the tuning isn't worth adopting.

---

## 1. Opting in

A strategy joins the optimiser by declaring a multi-value `param_grid()`:

```python
def param_grid(self):
    return {"ema_fast": [8, 10, 12, 15], "ema_slow": [21, 26, 30, 40]}

def default_params(self):          # the config the strategy ships with
    return {"ema_fast": 8, "ema_slow": 21}
```

Two rules that matter:

1. **The parameters must be used in `run_backtest`, not baked into `build_cache`.**
   The cache is built **once** and reused for every config, so anything the sweep
   varies has to be applied when the signal is computed — otherwise it can't vary.
2. **The default must be a cell in the grid.** The optimiser benchmarks the winner
   against the untuned default; if the default isn't in the grid it warns and the
   comparison is unavailable.

Strategies without a `param_grid()` just run as a single config — nothing changes.
`rigor optimize` is a **separate command, never part of `rigor run` or CI**, so it is
never mandatory.

```bash
rigor optimize strategies/<category>/<slug> --as-of 2026-06-01          # base pipeline
rigor optimize strategies/<category>/<slug> --as-of 2026-06-01 --full   # every stage
```

It is **non-destructive**: it writes a gitignored `artifacts/<slug>_optimization.json`
and never edits `config.json`. If you adopt the winning params, you edit the config
yourself and **bump the `version`** (see *Strategy vs. version vs. baseline* in the
standard).

---

## 2. The pipeline, stage by stage

The base pipeline (always runs) plus the opt-in research stages (`--full`). Each
stage answers one question; read them together, not in isolation.

| Stage | Flag | The question it answers |
|---|---|---|
| **Search** | always | Which config has the best in-sample Sharpe? (`--two-pass` for big grids) |
| **Select** | `--select` | `sharpe` = top Sharpe; `robust` = clears robustness floors *then* maximises performance |
| **Verdict** | always | Rigor's normal grade on the winner (the same one any backtest gets) |
| **Deflated Sharpe** | always | Is the winner's Sharpe still positive after correcting for *N* trials? |
| **CPCV-PBO** | always | Probability the ranking is overfit (combinatorially-purged CV) |
| **Data snooping** | always | Does the winner beat the **default** once you account for the whole search? |
| **Walk-forward** | always | Does the *search itself* generalise out-of-sample? |
| **Enrichment** | `--enrich` | Robustness scalars: temporal decay, stability, PSR, UPI, Sortino, recency |
| **Holdout** | `--holdout` | Does the winner survive a never-optimised final-20% slice? |
| **Falsification** | `--falsify` | How fast does the edge die under added noise (fragility)? |
| **5-pillar score** | `--score` | A composite selection score across 5 balanced axes |
| **Regime** | `--regime` | Which config wins in each volatility regime, and is there a consensus? |
| **Clustering** | `--cluster` | How many *distinct* good parameter regions does the grid have? |
| **Ensemble** | `--ensemble N` | Does an average of N diverse top configs hold up near the single best? |

### The anti-overfit trio (why three?)

They measure different things and a real edge should pass all three:

- **Deflated Sharpe** corrects the *single* winner for the number of trials,
  assuming the trials are independent.
- **CPCV-PBO** asks whether the *ranking* generalises — does the in-sample best
  tend to land in the bottom half out-of-sample across many splits?
- **Data snooping** (White Reality Check / Hansen SPA / Romano-Wolf StepM) uses the
  configs' **actual correlation** (a stationary bootstrap), not the independence
  assumption, to ask whether the best-of-grid genuinely beats the untuned default.

---

## 3. How to read the report — a worked example

`rigor optimize strategies/Baseline/crypto/dual_ema_crypto --as-of 2026-06-01 --full`:

```
optimized dual_ema_crypto: 126/126 configs (one_pass search, numba backend)
  best   : sharpe=0.86  {ema_fast: 10, ema_slow: 21, sma_trend: 150}
  default: sharpe=0.81  (improvement +0.05)
  overfit: deflated_sharpe=0.709 over n_trials=126
  cpcv   : PBO=0.22 (ROBUST)
  snoop  : White p=0.649  Hansen-SPA p=0.602  StepM beats 0/64
  walk-fwd: IS=1.16 -> OOS=0.67 (degradation +0.49, 5/5 folds positive)
  holdout: IS=1.12 OOS=0.34 hold=0.33 (ok)
  falsify: noise fragility=0.29  suspect=False
  5-pillar: composite=4.80  {ema_fast: 8, ema_slow: 26, sma_trend: 150}
  regime : 3 regimes, consensus (1 votes)
  cluster: 5 parameter clusters
```

**The honest reading — two separate conclusions:**

1. **The strategy's edge is real.** Deflated Sharpe stays **0.71** after correcting
   for 126 trials; CPCV-PBO **0.22 = ROBUST**; walk-forward is **5/5 folds positive**;
   holdout (0.33) ≈ OOS (0.34), not degraded; noise fragility **0.29** is low (the
   edge decays gracefully).
2. **But tuning the parameters is *not* worth it.** The best config beats the untuned
   default by only **+0.05 Sharpe**, and the snooping tests say that gap is **noise**:
   White p=**0.65**, and Romano-Wolf StepM finds **0 of 64** configs that provably beat
   the default. → **Keep the baseline; don't adopt the tuned peak.**

That second conclusion is the one a naïve "pick the highest Sharpe" optimiser would
hide. The `+0.05` looks like free money until the snooping tests price in the fact
that you searched 126 configs to find it.

**Reading rules of thumb:**

| Signal | Good | Worry |
|---|---|---|
| Deflated Sharpe | > 0 (ideally ≳ 0.5) | ≤ 0 → the winner is trial-luck |
| CPCV-PBO | < 0.40 (ROBUST) | > 0.55 (FRAGILE/OVERFIT) |
| Snooping (White / Hansen p) | < 0.05 → tuning beats default | > 0.10 → tuning is noise |
| Walk-forward | most folds positive, small IS→OOS gap | OOS negative / huge degradation |
| Holdout | `degraded: false` | `degraded: true` |
| Falsification fragility | < 0.8 | ≥ 0.8 → brittle to tiny perturbations |
| Ensemble vs best | within ~0.05 of the single best | collapses → winner is a lucky point |

The full numeric detail (per-config ranking, CPCV CI, enrichment scalars, per-regime
winners, cluster centres, 5-pillar breakdown) is in the report JSON.

---

## 4. Search & selection modes

- **`--two-pass`** — for grids too big to run in full on modest hardware. Thins each
  axis to a coarse grid, finds the winner's neighbourhood, then re-expands *only* that
  neighbourhood at full resolution — visiting a fraction of the cells. (On the example
  above it evaluated 45 of 126 and found an equally-good winner.) Large grids are also
  down-sampled with a **Sobol low-discrepancy** sequence for even coverage.
- **`--select robust`** — instead of top Sharpe, keep only configs that clear the
  robustness floors (Sharpe ≥ 0.3, PSR ≥ 0.6, no wipe-out), reject the whole grid if
  CPCV-PBO is too high, then maximise `CAGR × log(1 + activity)` among the survivors.
  Returns **`REJECTED`** if nothing is robust — the honest answer that the grid has no
  durable edge, rather than shipping the prettiest overfit config.

---

## 5. Calibration profiles — one command, research → deployment-readiness

`--select robust` picks *which* config wins, but it always uses **one** set of floors.
Real work has two questions, and they want **different** floors:

- **Research:** "Is there *anything* here worth a closer look?" — you want *permissive*
  floors that surface candidates and let you read the anti-overfit numbers yourself.
- **Deployment:** "Is this winner ready to carry real risk?" — you want *strict* floors
  aligned with the promotion gate, so only a genuinely robust edge passes.

`--profile` runs the whole pipeline under a **named calibration profile** and ends with a
single deployment-readiness verdict. It is **one command** for
search → validate → select-under-the-profile → gate:

```bash
rigor optimize strategies/<category>/<slug> --profile discovery    # permissive (advisory)
rigor optimize strategies/<category>/<slug> --profile deployment   # strict (gates, exits non-zero on REJECT)
```

A profile is **just a typed bundle of thresholds** — no new algorithm. It reuses the
existing `robust` selection ladder (it *is* the floors the ladder enforces) and adds a few
deployment-readiness checks the ladder doesn't cover. The two shipped profiles:

| Floor / gate | `discovery` (permissive) | `deployment` (strict) | Where it acts |
|---|---|---|---|
| Sharpe floor | ≥ 0.30 | ≥ 0.80 | selection ladder (per config) |
| PSR floor | ≥ 0.60 | ≥ 0.90 | selection ladder (per config) |
| CAGR floor | > −0.05 | > 0.00 | selection ladder (per config) |
| Max-drawdown floor | > −0.999 | > −0.60 | selection ladder (per config) |
| CPCV-PBO ceiling | < 0.50 | < 0.40 | selection ladder (grid-level) |
| Deflated-Sharpe floor | ≥ 0.50 | ≥ 0.60 | deployment gate (winner) |
| Min active bars | ≥ 20 | ≥ 100 | deployment gate (winner) |
| Blocking verdicts | `REJECT` | `REJECT`, `CONDITIONAL` | deployment gate (winner) |

`discovery`'s floors are exactly the framework's existing `RobustFloors()` defaults, so
`--profile discovery` selects identically to `--select robust` — it just adds the advisory
gate on top. **`deployment` admits a strict subset of what `discovery` admits**: a
borderline winner (say Sharpe 0.69, PBO 0.13) is *selected and admitted* under `discovery`
but its grid is *rejected* under `deployment` because 0.69 < the 0.80 Sharpe floor.

### How `deployment` ties to promotion

The deployment gate does **not** invent its own verdict — it defers to the **one verdict**
the whole framework trusts (`rigor.validation.verdict`, the same `PROMOTE / CONDITIONAL /
REJECT` you see on every `rigor run`). The profile only chooses which of those verdicts are
*disqualifying for deployment*: `deployment` blocks both `REJECT` and `CONDITIONAL`, which
matches the **enforced promotion gate** (`rigor.project.promotion_gate`) that already blocks
any deployable (paper/live) strategy carrying a `REJECT`. The outcome vocabulary follows:

- `--profile deployment` → **`PROMOTE`** / **`REJECT`**, and the command **exits non-zero**
  on `REJECT` so a CI / pre-push hook can block shipping a not-ready winner.
- `--profile discovery` → advisory **`ADMIT`** / **`WITHHOLD`**, always exits 0.

The full per-check detail (which floor failed and by how much) is printed under `gate :`
and written to `artifacts/<slug>_pipeline_<profile>.json`. Omitting `--profile` is fully
backward-compatible — you get the classic per-stage report, unchanged.

---

## 6. Compute backend

The bootstrap-heavy stages (snooping, falsification) accelerate on **Numba** (CPU JIT)
or **CUDA** (GPU). `--backend auto` picks the fastest available and falls back to NumPy.

**Results are identical across backends.** The random draws are generated once,
deterministically, with NumPy; the backends only accelerate the arithmetic reduction
over those fixed draws. The backend is a *speed knob, never a results knob* — verified:
numpy and numba produce byte-identical p-values. The accelerators are optional extras,
never required:

```bash
pip install -e "Framework[accel]"        # Numba (CPU)  — ~1.5-2x on the bootstrap stages
pip install -e "Framework[accel-cuda]"   # + CuPy (NVIDIA GPU); match the wheel to your CUDA
```

Without a GPU the `cuda` choice simply falls back — that is expected, not an error.

---

## 7. Architecture

The optimiser is a thin orchestrator over small, single-purpose modules. Everything
statistical lives in `rigor.validation` (reused by the normal verdict too); the
`rigor.optimize` package is the search + the research stages.

```
rigor/optimize/
  core.py          orchestrator: search -> select -> verdict -> snooping -> stages
  pipeline.py      profiled one-command flow: search -> validate -> select -> deploy gate
  calibration.py   named threshold profiles (discovery / deployment)
  grid.py          enumerate / Sobol-sample / two-pass (decimate + refine_around)
  select.py        selection modes (sharpe, robust-then-performance)
  walkforward.py   expanding / rolling walk-forward optimisation
  ensemble.py      diverse top-K ensemble
  enrichment.py    stage-3 robustness scalars (ROC310, PSR, UPI, Sortino, ...)
  holdout.py       final-20% IS/OOS/holdout split
  falsification.py noise-injection fragility + phase-randomised surrogate prices
  scoring.py       5-pillar V4.2 composite selection score
  regime.py        per-regime best config + consensus
  clustering.py    k-means over top-config parameter space

rigor/validation/
  permutation.py   White RC / Hansen SPA / Romano-Wolf StepM + p-adjustments
  accel.py         compute backend (numpy/numba/cuda) + batched bootstrap + Sobol
  cpcv.py          combinatorial purged CV -> PBO (+ Kendall rank-corr, CI)
  overfit.py       deflated / probabilistic / haircut Sharpe, Newey-West
  robustness.py    walk-forward, per-year, bootstrap CI
  verdict.py       the single promotion verdict (shared with `rigor run`)
```

**Design principle:** the optimiser **reuses the one verdict** the whole framework
trusts — it never introduces a parallel scoring system. The research stages (5-pillar
score, enrichment, etc.) are reported *alongside* the verdict, never replacing it. The
calibration **profiles** follow the same principle: they tune *which floors* the existing
ladder and gate enforce, and the deployment outcome defers to the shared verdict — they
never introduce a second opinion.

---

## 8. Making it mandatory on *your* machine (optional)

The framework keeps optimisation optional for everyone. If you personally want it
enforced before you push a tuned strategy, install the local-only git hook (it lives
in `.git/hooks/`, is never committed, and affects only you):

```bash
cp scripts/hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push   # enable
rm .git/hooks/pre-push                                                          # disable
```

It runs `rigor optimize` on any pushed strategy that has a `param_grid` and blocks on
error (`RIGOR_OPTIMIZE_BLOCK_OVERFIT=1` also blocks on a REJECT verdict). Bypass once with
`git push --no-verify`.
