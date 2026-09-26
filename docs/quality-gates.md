# Quality gates

A set of **additive, non-breaking** checks that keep the shared catalog honest.
Nothing here changes how strategies run — they validate what's already committed
and give every contributor the same guardrails. Three are exposed as `rigor`
sub-commands; two run automatically with the test suite.

## `rigor audit` — artifact coherence (blocking in CI)

Re-derives each strategy's headline metrics **from its own `artifacts/<slug>_returns.csv`**
(using `rigor.metrics`) and compares them to the committed `artifacts/<slug>_summary.json`.
Catches "the committed summary doesn't match its own returns."

```bash
rigor audit                 # audit every production strategy
rigor audit --json          # machine-readable
rigor audit --include-baseline
```

Finding codes:

| Code | Severity | Meaning |
|---|---|---|
| `PERF_NE_CURVE` | ERROR | a recomputed metric (Sharpe/CAGR/MaxDD/vol) disagrees with `summary.json` beyond tolerance |
| `DEGENERATE_ARTIFACT` | ERROR | returns empty / all-NaN / constant, or the metrics block is missing |
| `MISSING_ARTIFACT` | ERROR | `returns.csv` or `summary.json` absent / unreadable |
| `MISSING_THESIS` | ERROR | the strategy has no `<slug>_thesis.pdf` — every strategy must ship its thesis (run `rigor run` without `--no-thesis`) |
| `MISSING_LEDGER` | **WARN → ERROR*** | `<slug>_positions.csv` is absent and the strategy is not `ledger_exempt`; run `rigor run` so the runner writes it, or set `ledger_exempt: true` in `config.json` for semantically-N/A strategies |
| `DEGENERATE_LEDGER` | **WARN → ERROR*** | positions CSV exists but `max_adverse_excursion` and `max_favorable_excursion` are entirely NaN — re-run with a `BacktestResult` that carries weight/price data (i.e. `return simulate_weights(...)`, not `.returns`) |
| `DUAL_SLUG_SHADOW` | WARN | the same slug appears in two strategy directories |
| `OK` | OK | returns and summary reconcile and the thesis is present |

\* **Migration window:** `MISSING_LEDGER` and `DEGENERATE_LEDGER` are WARN today so existing
strategies committed before the ledger requirement was introduced are not immediately broken.
Once every committed strategy has been re-run and its `_positions.csv` is present, these
findings will be promoted to ERROR. Do not rely on the WARN grace period for new strategies —
the scaffold template already returns the full `BacktestResult` by default.

Exit code is non-zero if any ERROR is present, so **CI fails on an incoherent
artifact**. (Tolerates both committed `returns.csv` layouts — named `date` column
and unnamed leading index.) The whole current catalog passes clean.

## `rigor check-thesis` — README headline-number gate (advisory)

Extracts the **Sharpe / CAGR / Max-DD** claims from each strategy's `README.md`
prose and compares them to the committed `summary.json`. Flags drift so a stale
README can't quietly contradict the data. Advisory by default (prints `WARN`,
never blocks); pass `--strict` to make WARN findings fail.

```bash
rigor check-thesis          # advisory: PASS / WARN, always exit 0
rigor check-thesis --strict # WARN counts as failure (exit 1)
```

It is deliberately lenient about two-column "this-port vs deployed-main" tables —
a claim that matches *either* column is accepted — and only checks the three
headline metrics, not every number in the prose.

## `rigor regression` — metric regression diff

Diffs each strategy's current `summary.json` against the version at a git ref
(default `origin/main`) and flags regressions in Sharpe, CAGR, Max-DD or verdict
score beyond tolerance. New strategies (no baseline at the ref) are allowed, not
failures. Degrades to a no-op if git/the ref is unavailable, so it never crashes.

```bash
rigor regression                    # vs origin/main
rigor regression --ref HEAD~1       # vs the previous commit
```

Use it locally before committing a strategy edit to confirm you didn't silently
degrade it.

## `rigor.project.promotion_gate` — promotion gate (blocking in CI, status-aware)

The validation battery already computes a promotion **verdict**
(`PROMOTE` / `CONDITIONAL` / `REJECT` — see `rigor.validation.verdict`) and
`rigor run` persists it into each strategy's `artifacts/<slug>_summary.json` under
`verdict.verdict`. Historically that verdict was *reported only*: nothing stopped
a `REJECT` strategy from being promoted to paper/live and shipped. This gate
turns the verdict into an **enforced** check.

```bash
python -m rigor.project.promotion_gate          # blocking: exit 1 on a deployable REJECT
python -m rigor.project.promotion_gate --json   # machine-readable, per-strategy
```

**What it enforces.** For every production strategy it resolves an *effective
lifecycle* (below) and reads the recorded verdict from the committed
`summary.json` — **no live data is fetched, the verdict is read from committed
artifacts.** It FAILS (exit 1) if and only if a strategy is **deployable** *and*
its recorded verdict is `REJECT`.

**Deployable vocabulary.** "Deployable" means lifecycle ∈ **{`paper`, `live`}**
(`DEPLOYABLE_LIFECYCLES`). These are the only states the gate enforces against;
`research` / `backtested` / `validated` / `idle` / `retired` are **EXEMPT**. The
effective lifecycle is resolved exactly like `rigor.project.governance`:

1. an explicit `governance.lifecycle` in `config.json`, if present; else
2. the legacy `status` field mapped through `LEGACY_STATUS_TO_LIFECYCLE`
   (`idle` → `backtested`, `paper` → `paper`, `live` → `live`); else
3. `backtested` (a safe, non-deployable default).

Because today's whole research catalog is `status: idle` (→ `backtested`,
**not** deployable), the gate **passes vacuously now** — but it bites the moment
a strategy is promoted to `paper`/`live`.

**Exemptions (the gate never fails on missing evidence).**

| Situation | Result |
|---|---|
| lifecycle not in {paper, live} (e.g. idle/research) | EXEMPT — passes even if `REJECT` |
| deployable but **no verdict recorded** (legacy artifact, or no `summary.json`) | EXEMPT — re-run to record a verdict |
| deployable + verdict `PROMOTE`/`CONDITIONAL` | PASS |
| **deployable + verdict `REJECT`** | **FAIL (exit 1)** |

Baseline strategies (`strategies/Baseline/`) are skipped — they are raw research
references, never independently deployed.

**Promoting a strategy now requires a non-REJECT verdict.** To move a strategy to
`paper`/`live` (via `status` or `governance.lifecycle`), its committed
`summary.json` must record a `PROMOTE` or `CONDITIONAL` verdict — re-run
`rigor run` to refresh it. A `REJECT` strategy cannot be promoted until its
overfit/significance problems are fixed.

There is an opt-in local equivalent on `rigor run`:

```bash
rigor run <path> --gate     # (alias: --strict) exit 1 if this run's verdict is REJECT
```

Default behaviour is unchanged; `--gate` only adds the non-zero exit so a
pre-commit hook can block shipping an overfit strategy at author time.

## `rigor.project.optimize_gate` — optimize gate (blocking in CI, ratcheting)

A strategy is only *optimisable* — in-house via `rigor optimize` and by the
external optimiser — if it declares which parameters are tunable and
over what ranges. That surface is `param_grid()`: a `dict[str, list]` whose
cartesian product is the search space. `StrategyBase.param_grid` defaults to `{}`
(a single run), so historically a strategy could be written, backtested and
committed with **no tunable surface at all** and every gate still passed — about
half the catalog drifted that way (mostly ported strategies reproducing one
committed configuration). This gate makes "declares a sweepable grid" an enforced
invariant.

```bash
python -m rigor.project.optimize_gate                  # blocking: exit 1 on a NEW violation
python -m rigor.project.optimize_gate --json           # machine-readable, per-strategy
python -m rigor.project.optimize_gate --strict         # also fail on grandfathered debt
python -m rigor.project.optimize_gate --update-baseline # re-snapshot debt after a backfill
```

**What it enforces — statically, no data.** It parses each `strategy.py` (AST)
and inspects the literal returned by `param_grid()` — **no import, no data fetch,
no cache** — so it runs in milliseconds against the whole catalog and never flakes
on a missing EODHD key. A strategy is:

| Classification | Meaning | Result |
|---|---|---|
| **MULTI** | literal `dict`, ≥1 axis with ≥2 type-consistent values | PASS |
| **DYNAMIC** | grid computed at runtime (non-literal) | UNVERIFIABLE — reported, not blocked |
| **NO_METHOD / EMPTY / SINGLE / INCONSISTENT** | no override / `{}` / no multi-value axis / mixed types | VIOLATION |

A VIOLATION **fails the build (exit 1)** unless the strategy is grandfathered or
exempt. Type consistency mirrors the external optimiser's contract: numbers may mix
`int`/`float`, otherwise a single axis must be one type. (A nested helper's
`return` inside `param_grid` is not mistaken for the grid.)

**Grandfathering → fully backfilled (baseline = 0).** The baseline
(`rigor/project/optimize_gate_baseline.json`) once held the pre-gate catalog as accepted
WARN-debt; that debt has now been **driven to zero** — every strategy is either gridded
(local grid, or one inherited from a shared `rigor.strategies.*` engine) or carries an
`optimize_exempt_reason`. CI therefore runs the gate with **`--strict`**, which fails on
*any* residual debt, so the catalog can never regress. Cache-bound knobs were made
sweepable without touching `build_cache` via a per-combo rebuild from an overridden copy
of `self`; the 121 hardcoded-knob exemptions are the manual-parameterisation backlog in
a generated backlog file.

**Shared-engine re-exports.** A thin `from rigor.strategies.<engine> import Strategy`
strategy inherits `param_grid` from the engine; the gate resolves the engine file and
classifies its grid, so these count as compliant without a local grid.

**Exemptions.** A strategy with a genuinely empty tunable surface (a fixed
calendar/seasonal pattern with no knobs) sets `optimize_exempt_reason` in its
`config.json` — a reviewed, explicit opt-out (an empty/whitespace reason does not
exempt), never a silent empty grid. New strategies are compliant by construction:
the `rigor new` scaffold emits a real multi-value `param_grid()` stub.

Baseline strategies (`strategies/Baseline/`) are skipped — raw research references
are never optimised independently.

## Property-based tests (`tests/test_properties.py`)

[Hypothesis](https://hypothesis.readthedocs.io/) tests that assert invariants of
the numeric primitives across randomly-generated inputs — e.g. max-drawdown ≤ 0,
ES ≤ VaR, Sharpe scale-invariance, PSR/DSR ∈ [0,1], verdict score ∈ [0,100].
They run with the normal suite (`pytest`); `hypothesis` is in the `dev` extra.

## Archetype verdict presets (`rigor.analysis.verdict`)

`compute_strategy_verdict(..., preset=...)` adjusts the 5-pillar gate thresholds
for a strategy archetype without changing gate names, weights, or the
neutral-gate rescaling. `preset="default"` (or omitting it) is byte-identical to
before. Presets: `momentum`, `mean_reversion`, `conservative`, `institutional`
(strictest), `exploratory` (most lenient). See `VERDICT_PRESETS`.

## CI

The `lint + unit tests` job runs, after pytest:
- `python -m rigor.project.audit` — **blocking** (fails the build on an incoherent artifact).
- `python -m rigor.project.promotion_gate` — **blocking** (fails the build if a deployable
  strategy carries a `REJECT` verdict; research/idle strategies are exempt so it passes today).
- `python -m rigor.project.optimize_gate --strict` — **blocking** (every strategy must be
  grid-or-exempt; the backfill baseline is now **empty**, so any strategy that is neither
  gridded nor exempt fails the build). The gate resolves shared-engine re-exports
  (`from rigor.strategies.<engine> import Strategy`) to the engine's inherited grid.
- `python -m rigor.project.thesis_gate` — **advisory** (prints README drift, never blocks).
