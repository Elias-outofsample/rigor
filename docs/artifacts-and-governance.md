# Artifacts, aggregation & governance

Additive tooling (Sections C & D) around the committed per-strategy artifacts
(`returns.csv` / `summary.json` / `report.html` / `thesis.pdf`). Nothing changes
how strategies run — these aggregate, enrich, and track what's already there. All
four are exposed as `rigor` sub-commands and reuse the canonical `find_repo_root()`
and `rigor.metrics` (no reinvented formulas, no duplicated repo-root walkers).

## `rigor book` — book-wide aggregator (C1)

Rolls every production strategy's committed results into one queryable table plus
lean per-strategy snapshots, so reports / the portfolio app render on any machine
with **no data license** — straight at the reproducibility / OneDrive-corruption pain.

```bash
rigor book                 # MASTER table + per-strategy decimated snapshots
rigor book --master-only   # just the MASTER table
```

- `catalog/MASTER_SNAPSHOT.parquet` — one row per strategy (name/category/
  status/version + Sharpe/CAGR/MaxDD/vol/Sortino/Calmar/n + verdict + as_of).
- `artifacts/<slug>_snapshot.json.gz` — headline metrics + an equity curve
  decimated to ~800 points (gzip, < 500 KB).

Both are **regenerable build outputs** (gitignored) — rebuild offline any time
from the committed `returns.csv` + `summary.json`. The aggregate is a convenience
cache, not a source of truth.

## `rigor derive` — richer per-strategy artifacts (C2)

Computes governance artifacts purely from each strategy's committed `returns.csv`
(fully offline, no key), reusing `rigor.validation` + `rigor.analysis`:

```bash
rigor derive               # all strategies
rigor derive --slug <slug> # one strategy
```

- `artifacts/<slug>_wf.json` — walk-forward folds + efficiency.
- `artifacts/<slug>_risk.json` — VaR/CVaR/Omega/Ulcer/… + EVT (GPD tail) + Kupiec VaR backtest.
- `artifacts/<slug>_peryear.json` — per-calendar-year return / Sharpe / MaxDD.

Regenerable build outputs (gitignored). They feed reports, the coherence auditor,
and per-event analysis without re-running the backtest.

## `rigor.project.audit` — the artifact coherence auditor

Recomputes each strategy's headline metrics from its committed `returns.csv` and
reports drift against `summary.json`, plus **structural** cross-strategy checks.
Severity model: **ERROR fails the build** (CI exit non-zero); **WARN is advisory**
(never blocks). Run `python -m rigor.project.audit` (`--json` for machine output).

Per-strategy checks (recompute / presence):

- `PERF_NE_CURVE` (ERROR) — recomputed Sharpe/CAGR/MaxDD/vol disagree with `summary.json`.
- `MISSING_ARTIFACT` (ERROR) — `returns.csv` / `summary.json` absent or unparseable.
- `MISSING_THESIS` (ERROR) — no `<slug>_thesis.pdf`.
- `DEGENERATE_ARTIFACT` (ERROR) — returns empty / all-NaN / constant (std == 0).
  When `summary.json` nonetheless claims a non-trivial Sharpe (|Sharpe| ≥ 0.10),
  the message flags the artifact as **internally incoherent** (a flat curve cannot
  earn a real Sharpe).
- `MISSING_LEDGER` / `DEGENERATE_LEDGER` (WARN) — per-trade positions CSV absent or
  all-NaN excursions; suppressible with `ledger_exempt: true` in `config.json`.

Structural cross-strategy checks (adapted from an earlier in-house coherence audit, re-implemented
in our Finding model; calibrated so the **current catalog passes at 0 ERROR**):

- `DUAL_SLUG_DUPLICATE` (ERROR) — two slugs with **numerically identical** returns
  whose `config.json` engine params are **also identical** (name/slug ignored): an
  unambiguous copy-paste / mis-slug — the same strategy committed twice.
- `DUAL_SLUG_NEAR_DUPLICATE` (WARN) — identical returns but configs declare a
  **different** engine parameter (e.g. `extra.variant`): a documented redundancy by
  design, not a defect. Review whether both slugs are needed.
- `RETURNS_CORRELATION` (WARN, **advisory only**) — two *distinct* strategies whose
  returns correlate above ~0.999. Deliberately never an ERROR: our momentum tilt/base
  variants are legitimately near-1.0 correlated by design, so an ERROR here would red
  CI on dozens of valid strategies.
- `DUAL_SLUG_SHADOW` (WARN) — the same slug name appears in multiple directories.

## `rigor govern` — controlled-vocabulary governance (D1)

Richer governance than the auto-`INDEX`: a controlled vocabulary for **lifecycle**
(`research→backtested→validated→paper→live→retired`), **asset_class**, **edge_type**,
and cross-links — held as an optional `governance` block inside `config.json` (the
single source of truth; not README front-matter, to avoid duplication).

```bash
rigor govern               # book-wide governance summary (advisory)
rigor govern --strict      # exit non-zero on vocabulary violations
rigor govern --backfill    # add a default governance block to configs that lack one
```

Works **without** explicit blocks: `status` maps to lifecycle and `category` maps
to asset_class via documented fallbacks, so the summary is populated from day one.
Add an explicit `governance` block per strategy when you want a non-default
lifecycle (e.g. `live`) or to record an `edge_type` / linked thesis. New strategies
can be enriched with `governance.enrich(...)`. `--backfill` reformats JSON, so it's
opt-in, not run across the book by default.

## `rigor registry` — model provenance ledger (D2)

An append-only ledger that gives every strategy version a fingerprint and lineage —
mapping the R1→R4 optimization round-arcs and the provenance we otherwise only have
in prose + git.

```bash
rigor registry record <strategy_dir>   # append a record (fingerprints + metrics)
rigor registry list [--slug <slug>]    # list records
rigor registry show <model_id>         # trace the parent lineage chain
```

Stored as a git-friendly **append-only JSONL** (`catalog/model_registry.jsonl`,
one object per line) — *not* a binary SQLite blob, which would churn git diffs.
Each record carries `model_id`, `params`, `metrics`, `dataset_hash`,
`code_fingerprint` (sha256 of strategy.py), `config_fingerprint`, `parent_model_id`
(lineage), `status`, `created_at`. `registry.to_sqlite()` builds an in-memory
SQLite view for ad-hoc SQL when you want it.

## Deliberately not adopted (and why)

- **Packed-membership int64 codec** — worth 120 MB → 5 MB on a *dense daily
  matrix*; our PIT membership is a **5.3 MB event log** (already sparse). Marginal.
- **`<ASSET>_<strategy>_<dates>` filename convention** — the book uses **one** as-of
  EODHD cache keyed by `ticker.exchange`, not per-strategy data copies.
- **Rigid 13-section `strategy.py`** — covered by `strategy-standard.md` + `rigor
  validate` + `rigor new`; a 75-strategy retrofit is churn for little gain.
- **ProjectTree git-DAG site** — cosmetic; `git`/`gh` already cover it.
