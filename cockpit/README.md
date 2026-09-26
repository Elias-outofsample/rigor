# Portfolio Cockpit

A browser-based, single-screen **Portfolio Cockpit** for the `rigor` strategy book. The flow is
**search → candidate list → open one**: pick strategies in the left rail, set the
criteria, then **★ Search** grid-searches the best focused *k-of-N* portfolios that
meet them and returns a **ranked, sortable** list; click any candidate to open its
full dossier (verdict, gates, a toggleable equity chart, weights, attribution,
per-leg-vs-portfolio metrics, correlation, tail dependence, regimes, stress, factors).

Options include a **common-window** backtest (score each book only where its legs
overlap), a **minimum-start-year** universe filter (rail + search), optional **CPCV**
out-of-sample scoring, and per-strategy weight caps / cardinality bounds.

The FastAPI service (`cockpit_api/`) is a thin layer over the headless engine
(`portfolio_engine/`) that ships in the same package — every number comes from
`portfolio_engine.cockpit` (`build_portfolio` / `search_portfolios`), which builds on
the `rigor` framework (single-source metrics / validation). A PySide6 desktop app
preceded this; it was retired when the cockpit moved to the browser.

> **Provenance.** The API *shape* follows an earlier in-house cockpit service; it is
> re-implemented over `rigor`'s own engine and `strategies/` book, with a small
> dependency-free frontend.

---

## Architecture

```
cockpit/
├── portfolio_engine/       # the headless engine (no GUI) — on rigor.metrics / rigor.validation
│   ├── cockpit.py          #   build_portfolio / build_from_weights / search_portfolios
│   ├── candidates.py  weight_methods.py  allocator.py  selector.py  backtester.py
│   ├── search.py  analysis.py  portfolio_validation.py  stress.py  risk.py
│   └── data.py  portfolio_io.py
├── cockpit_api/            # FastAPI backend
│   ├── main.py             #   app + CORS + static mount + /api/saved-report
│   ├── settings.py         #   env-overridable config (strategy root, CORS)
│   ├── schemas.py          #   Pydantic request/response models
│   ├── engine.py           #   adapter over portfolio_engine + JSON serialisation
│   └── routers/            #   strategies.py · portfolio.py
├── frontend/               # vanilla HTML/CSS/JS (no build step, no npm)
│   ├── index.html  styles.css  app.js
└── tests/                  # engine tests + the FastAPI TestClient API tests (offline)
```

The package ships **two importable modules**: `portfolio_engine` (the headless engine,
also usable on its own) and `cockpit_api` (the FastAPI service over it). No PySide6 —
the desktop app was retired when the cockpit moved to the browser.

---

## Run it

```bash
# from the repo root
pip install -e Framework             # the rigor package the engine builds on
pip install -e "cockpit[dev]"   # the FastAPI service + portfolio_engine + deps

# launch (from cockpit/) and open http://localhost:8000
uvicorn cockpit_api.main:app --reload --port 8000
```

Discovery uses the repo's `strategies/` book by default; override with
`COCKPIT_STRATEGY_ROOT`. A live benchmark/factor fetch needs the EODHD key in
`.env` (same as the rest of the repo); without it the cockpit still builds and
validates, and simply shows "factor data unavailable offline".

---

## API

The primary flow is **search → candidate list → open one**: pick strategies, set
criteria, grid-search every portfolio that meets them, then click a candidate to see
its full dossier.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | status + number of discovered strategies |
| GET  | `/api/strategies` | the book: slug, category, Sharpe, CAGR, MaxDD, PBO, verdict |
| GET  | `/api/portfolio/methods` | available weighting methods (for the direct `/build`) |
| POST | `/api/portfolio/search` | grid-search every portfolio meeting the criteria; return the ranked candidates |
| POST | `/api/portfolio/detail` | open one chosen candidate (its weights) → full dossier |
| POST | `/api/portfolio/build` | direct single allocation by a chosen method → dossier (kept for parity) |
| POST | `/api/portfolio/save` | persist a chosen candidate to `portfolios/<name>/` (definition + artifacts + report) |
| GET  | `/api/portfolio/saved` | slugs of saved portfolios |
| GET  | `/api/saved-report/{name}` | a saved portfolio's HTML tearsheet |

`POST /api/portfolio/search` body → returns `{ candidates: [...], n, min_sharpe }`,
each candidate carrying its `weights` (keyed by slug) + headline metrics:

```json
{
  "slugs": ["std_long_spx", "ibs_long_only", "mr_ibs_nasdaq"],
  "max_weight": 0,
  "use_cpcv": false,
  "n_samples": 3000,
  "constraints": { "min_sharpe": 1.0, "max_dd": 30 }
}
```

`POST /api/portfolio/detail` then opens one: `{ "slugs": [...], "weights":
{ "std_long_spx": 0.5, ... }, "constraints": {...} }`.

Percentages are entered as shown in the UI (`max_dd: 30` = 30%); the adapter
converts to fractions before calling the engine. `min_sharpe` / `min_cagr` /
`max_dd` filter the search; the full strict gates are evaluated per candidate in
`/detail`.

---

## Tests & lint

```bash
ruff check cockpit_api tests
pytest                       # offline — synthetic book, no network
```

CI runs both in a path-isolated job that triggers only on `cockpit/**` changes
(see `.github/workflows/portfolioweb-ci.yml`).
