# Design decisions

Deliberate, documented tradeoffs in the `rigor` framework — the "why it's like
this on purpose" record, so an auditor (or a future maintainer) doesn't re-flag a
choice as an oversight. Each entry states the decision, the rationale, and how to
revisit it.

---

## 1. Thesis module is type-checked (no mypy blind spot)

**Decision.** `rigor.thesis` (the vendored PDF/thesis generator) is included in the
`mypy -p rigor` type surface like the rest of the package. The former
`[[tool.mypy.overrides]] module = "rigor.thesis.*" / ignore_errors = true` block has
been **removed**.

**Rationale.** Auditing the real cost found it small and trivial — 23 errors
across 16 files, ~22 of them implicit-`Optional` defaults (`x: str = None` →
`x: str | None = None`) plus two `-> dict` functions that return `None`, one
abstract-registry annotation, and one svglib stub-vs-runtime mismatch. None
touched logic, so typing the module (rather than hiding it) was the right call:
the whole package is now genuinely type-clean and there is no silenced corner
where regressions could accumulate.

- The two genuine type-system frictions are documented inline: a typed
  `_REGISTRY: dict[str, type[BasePaperType]]` in `paper_types/__init__.py`, and a
  narrow `# type: ignore[arg-type]` on `svg2rlg(BytesIO)` in `pdf/pdf_builder.py`
  (svglib accepts file-like objects at runtime; its signature only declares
  `str | PathLike`).
- `rigor/thesis` remains excluded from **ruff** (`extend-exclude`) and **coverage**
  (`omit`): it is vendored, so we don't restyle it, and it's exercised end-to-end
  by the smoke + `test_thesis_gate` tests rather than line-targeted unit tests.
  Type-checking is cheap and high-value there; lint/coverage churn is not.

**To revisit.** If a future vendored drop reintroduces many type errors and they
can't be fixed cheaply, prefer a *scoped* `# type: ignore` per line over
re-adding a blanket `ignore_errors` override — keep the blind spot as small as
possible.

---

## 2. `rigor.analysis` is an opt-in analyst toolkit (no production call-sites)

**Decision.** ~10 `rigor.analysis` modules (`alpha_robustness`, `attribution`,
`capture`, `cost`, `edge_classifier`, `labeling`, `predictive_ability`,
`scenario_response`, `structural`, `survival`, and the rest of the layer) have
**no production call-site on purpose**. They are not wired into the default
`rigor run` path.

**Rationale.** These answer deep-dive questions ("where does the alpha come
from?", "will it survive?", "what breaks it first?") that belong to a deliberate
analyst session, not to every backtest. Forcing them into the hot path would add
cost, heavy optional dependencies, and noise for strategies that don't need them.
The layer is purely additive: nothing in core imports it, so it can't regress a
strategy.

**Why it isn't rot.** Every module is covered by a unit test, and the full
catalogue (one line each: what it computes, that it's opt-in, how to call it) is
documented in [`analysis-toolkit.md`](analysis-toolkit.md), with the
`rigor.analysis` package docstring pointing there. Discoverability — not a
production call-site — is the thing that keeps opt-in code alive.

**To revisit.** If a module graduates into the default report/thesis surface,
wire it in and move its row out of the "opt-in" table in `analysis-toolkit.md`.

---

## 3. PIT membership — bankruptcy "Q" alias tail is incremental, not exhaustive

**Decision.** `rigor/data/pit_membership.py` maps Sharadar PIT membership tickers to
EODHD price codes. Coverage is active 100% / delisted ~95%; the unmapped ~5% is
the bankruptcy "Q" tail and is closed **incrementally** via a hand-verified
`ALIASES` map, **not** chased to 100%.

**Rationale.** When an S&P member files Chapter 11 its ticker gains a "Q" suffix
(`LEHMQ`, `WAMUQ`, ...). EODHD usually keys the delisted instrument under the
company's *pre-bankruptcy* primary symbol — which the `_candidates` heuristic can't
derive, and which other vendors frequently **reused** for an unrelated live
company. Blind root-stripping is actively dangerous: `DAL` (today's Delta is a
different listing from the 2005-bankruptcy `DALRQ`), `AMR`, and `DYN` all resolve
to live, unrelated instruments. So each alias must be verified by hand against the
real delisting date; automating it would silently inject the wrong instrument.

**Safety net added.** `TickerMapper.map` now emits an alias **only when its target
code is present in the live EODHD code set**. An alias to a missing/unlicensed or
mistyped code falls through to the heuristic and, failing that, to `unmapped` —
never fabricating a symbol. This makes newer candidate aliases self-validating and
preserves the existing verified aliases exactly (their codes are present).

**Aliases (current).** Verified-end-date: `LEHMQ→LEH`, `MTLQQ→GM_old`,
`RSHCQ→RSH`. Famous non-reused primaries, guard-gated: `WAMUQ→WAMU`,
`ENRNQ→ENE`, `NRTLQ→NT`, `EKDKQ→EK`.

**How to extend.** (1) Pull an unmapped "Q" ticker from `map_many(...)[1]`.
(2) Find its pre-bankruptcy *primary* ticker (not a digit/root strip of the
Q-symbol). (3) Confirm in EODHD that that code's price history **ends at the real
delisting date** (not a current company reusing the symbol). (4) Add it to
`ALIASES` with a dated comment. Beware the `DAL/AMR/DYN` reuse trap.

---

## 4. Committed `report.html` + `thesis.pdf` artifacts (~182 MB) are intentional

**Decision.** Per-strategy `*_report.html` and `*_thesis.pdf` (≈262 of each,
~182 MB total) are **tracked in git on purpose** and are **not** removed.

**Rationale.** These rendered artifacts are *anti-overfit evidence*: a reviewer
can open the report/thesis for any strategy directly on GitHub — equity curve,
walk-forward, regime behaviour, verdict — without cloning, installing, or holding
a data license. That public, browsable provenance is the point; treating them as
disposable build output would defeat it. They are fully regenerable (no
information is lost by committing them), so the cost is repo size only.

**Tradeoff.** ~182 MB of binary blobs inflate clone size and can't diff
meaningfully. Accepted: the audit/credibility value of one-click-visible evidence
outweighs repo weight, and the files change only when a strategy is intentionally
re-run.

**Regeneration.** Any artifact is reproducible from the committed inputs:

```bash
rigor run <path-to-strategy>          # re-emits returns/summary + report.html + thesis.pdf
rigor run <path-to-strategy> --no-thesis   # report only, skip the PDF
```

(Lean derived caches — `*_snapshot.json.gz`, `*_wf.json`, etc. — are by contrast
gitignored build outputs; see `docs/artifacts-and-governance.md`. The
distinction: rendered evidence is committed; recomputable caches are not.)
