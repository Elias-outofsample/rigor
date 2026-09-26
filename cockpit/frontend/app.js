"use strict";
// Web Portfolio Cockpit — vanilla JS, no build step. Talks to the /api backend,
// renders the strategy rail, fires ★ Build, and draws the result cards (incl. a
// dependency-free SVG equity chart). Mirrors the desktop cockpit's layout.

const VERDICT_COLORS = { PROMOTE: "#2ea043", CONDITIONAL: "#e3b341", REJECT: "#f85149" };
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid !== null && kid !== undefined)
    n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
};
// SVG elements must be created in the SVG namespace (createElement won't render them).
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs))
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  return n;
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const fmtPct = (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");
const fmtNum = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const setStatus = (msg) => { $("#status").textContent = msg || ""; };

let STRATEGIES = [];
const selected = new Set();

// --------------------------------------------------------------------------- //
// Strategy rail                                                               //
// --------------------------------------------------------------------------- //
function renderRail() {
  const needle = $("#filter").value.trim().toLowerCase();
  const minYear = Number($("#minYear").value) || 0;   // keep strategies since ≤ this year
  const byCat = {};
  for (const s of STRATEGIES) (byCat[s.category || "uncategorised"] ??= []).push(s);
  const box = $("#strategies");
  box.textContent = "";
  let shown = 0;
  for (const cat of Object.keys(byCat).sort()) {
    const members = byCat[cat]
      .filter((s) => s.slug.toLowerCase().includes(needle))
      .filter((s) => !minYear || (s.start_year && s.start_year <= minYear))
      .sort((a, b) => (b.sharpe ?? -9) - (a.sharpe ?? -9));
    if (!members.length) continue;
    const group = el("div", { class: "cat" });
    group.append(el("div", { class: "cat-head" }, `${cat}  (${members.length})`));
    for (const s of members) {
      shown++;
      const cb = el("input", { type: "checkbox" });
      cb.checked = selected.has(s.slug);
      cb.addEventListener("change", () => {
        cb.checked ? selected.add(s.slug) : selected.delete(s.slug);
        updateCount();
      });
      const badge = el("span", { class: "badge" }, fmtNum(s.sharpe));
      badge.style.color = VERDICT_COLORS[(s.verdict || "").toUpperCase()] || "#8b949e";
      badge.title = `${s.verdict} · since ${s.start_year ?? "?"} · CAGR ${fmtPct(s.cagr)} · MaxDD ${fmtPct(s.max_dd)}`;
      const yr = el("span", { class: "year muted" }, s.start_year ? `’${String(s.start_year).slice(2)}` : "");
      group.append(el("div", { class: "row" }, el("label", {}, cb, " " + s.slug), yr, badge));
    }
    box.append(group);
  }
  if (!box.children.length) box.textContent = "No strategies match.";
  updateCount(shown);
}

function updateCount(shown) {
  if (shown === undefined) shown = $("#strategies").querySelectorAll(".row").length;
  $("#count").textContent = `${selected.size} selected · ${shown} shown`;
  $("#search").disabled = selected.size < 2;
}

function checkShown(on) {
  for (const row of $("#strategies").querySelectorAll(".row")) {
    const cb = row.querySelector("input");
    const slug = row.querySelector("label").textContent.trim();
    cb.checked = on;
    on ? selected.add(slug) : selected.delete(slug);
  }
  updateCount();
}

// --------------------------------------------------------------------------- //
// Search → candidate list → open one                                          //
// --------------------------------------------------------------------------- //
function criteria() {
  const c = {};
  for (const inp of document.querySelectorAll("[data-con]")) c[inp.dataset.con] = Number(inp.value) || 0;
  return c;
}

async function search() {
  $("#search").disabled = true;
  setStatus(`Searching portfolios of ${selected.size} strategies…`);
  try {
    const req = {
      slugs: [...selected],
      max_weight: Number($("#maxWeight").value),
      use_cpcv: $("#useCpcv").checked,
      n_samples: Number($("#nSamples").value) || 5000,
      min_strats: Number($("#cardMin").value) || 0,
      max_strats: Number($("#cardMax").value) || 0,
      common_window: $("#commonWindow").checked,
      min_start_year: Number($("#minYear").value) || 0,
      constraints: criteria(),
    };
    const data = await post("/api/portfolio/search", req);
    renderSearch(req, data);
    setStatus(`${data.n} portfolios meet the criteria.`);
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    $("#search").disabled = selected.size < 2;
  }
}

let CURRENT = null; // { slugs, weights, constraints } of the opened candidate

async function openCandidate(req, cand, rowEl) {
  document.querySelectorAll("#cand-table tr.sel").forEach((t) => t.classList.remove("sel"));
  if (rowEl) rowEl.classList.add("sel");
  SEARCH.openRank = cand.rank;   // keep the selection highlighted across re-sorts
  setStatus("Opening portfolio…");
  try {
    CURRENT = { slugs: req.slugs, weights: cand.weights, constraints: req.constraints,
      common_window: !!req.common_window };
    renderDetail(await post("/api/portfolio/detail", CURRENT));
    $("#save").disabled = $("#export").disabled = false;
    setStatus("Portfolio opened.");
    $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    setStatus("Error: " + e.message);
  }
}

function topWeights(weights) {
  return Object.entries(weights).filter(([, w]) => w > 1e-6).sort((a, b) => b[1] - a[1])
    .slice(0, 4).map(([k, w]) => `${k} ${Math.round(w * 100)}%`).join(", ");
}

// Candidate-table state + columns (each sortable column carries the field key and
// the direction that means "best first").
let SEARCH = { req: null, candidates: [], sortKey: "composite", sortDir: "desc", openRank: null };
const _COLS = [
  { label: "#", key: "rank", best: "asc", fmt: (c) => String(c.rank + 1) },
  { label: "Composite", key: "composite", best: "desc", fmt: (c) => fmtNum(c.composite, 3) },
  { label: "Sharpe", key: "sharpe", best: "desc", fmt: (c) => fmtNum(c.sharpe) },
  { label: "OOS Sharpe", key: "sharpe_oos", best: "desc", fmt: (c) => fmtNum(c.sharpe_oos) },
  { label: "CAGR", key: "cagr", best: "desc", fmt: (c) => fmtPct(c.cagr) },
  { label: "MaxDD", key: "max_dd", best: "desc", fmt: (c) => fmtPct(c.max_dd) }, // desc = shallowest
  { label: "DSR", key: "dsr", best: "desc", fmt: (c) => fmtPct(c.dsr, 0) },
  { label: "Legs", key: "n_strats", best: "asc", fmt: (c) => String(c.n_strats) },
  { label: "Obs", key: "n_obs", best: "desc", fmt: (c) => (c.n_obs ? String(c.n_obs) : "—") },
  { label: "Top weights", key: null, fmt: (c) => topWeights(c.weights) },
];

function renderSearch(req, data) {
  const out = $("#results");
  out.textContent = "";
  SEARCH = { req, candidates: data.candidates || [], sortKey: "composite", sortDir: "desc", openRank: null };
  const header = el("div", {}, `${data.n} portfolio${data.n === 1 ? "" : "s"} meet the criteria` +
    (data.min_sharpe ? ` (Sharpe ≥ ${data.min_sharpe})` : "") +
    (data.common_window ? " · common-window (overlap only)" : "") +
    " — click a column to sort, click a row to open its dossier.");
  out.append(card("Candidate portfolios", "sortable — ranked by composite anti-overfit score by default",
    header, el("div", { id: "cand-wrap" })));
  out.append(el("div", { id: "detail" }, el("p", { class: "placeholder" }, SEARCH.candidates.length
    ? "Click a portfolio above to see its full dossier."
    : "No portfolio met the criteria — loosen them (e.g. lower min Sharpe) or add strategies.")));
  renderCandidateTable();
}

function sortCandidates(key) {
  if (!key) return;
  if (SEARCH.sortKey === key) {
    SEARCH.sortDir = SEARCH.sortDir === "desc" ? "asc" : "desc";
  } else {
    SEARCH.sortKey = key;
    SEARCH.sortDir = (_COLS.find((c) => c.key === key) || {}).best || "desc";
  }
  renderCandidateTable();
}

function renderCandidateTable() {
  const { candidates, sortKey, sortDir } = SEARCH;
  const sorted = [...candidates].sort((a, b) => {
    const d = (a[sortKey] ?? 0) - (b[sortKey] ?? 0);
    return sortDir === "asc" ? d : -d;
  });
  const head = el("tr", {}, ..._COLS.map((col) => {
    const arrow = col.key === sortKey ? (sortDir === "asc" ? " ▲" : " ▼") : "";
    const th = el("th", {}, col.label + arrow);
    if (col.key) { th.className = "sortable"; th.addEventListener("click", () => sortCandidates(col.key)); }
    return th;
  }));
  const body = sorted.map((c) => {
    const tr = el("tr", {}, ..._COLS.map((col) => el("td", {}, col.fmt(c))));
    if (c.rank === SEARCH.openRank) tr.classList.add("sel");
    tr.addEventListener("click", () => openCandidate(SEARCH.req, c, tr));
    return tr;
  });
  const table = el("table", { id: "cand-table" }, el("thead", {}, head), el("tbody", {}, ...body));
  const wrap = $("#cand-wrap");
  wrap.textContent = "";
  wrap.append(table);
}

async function savePortfolio(openReport) {
  if (!CURRENT) return;
  const name = prompt("Portfolio name:");
  if (!name) return;
  try {
    const out = await post("/api/portfolio/save", { ...CURRENT, name });
    setStatus(`Saved → portfolios/${out.saved}`);
    if (openReport && out.report) window.open(`/api/saved-report/${out.saved}`, "_blank");
  } catch (e) {
    setStatus("Save failed: " + e.message);
  }
}

// --------------------------------------------------------------------------- //
// Result cards                                                                //
// --------------------------------------------------------------------------- //
function card(title, subtitle, ...body) {
  const head = el("div", { class: "card-head", onclick: () => c.classList.toggle("collapsed") },
    el("span", { class: "toggle" }, "–"), el("h3", {}, title),
    subtitle ? el("span", { class: "muted" }, subtitle) : null);
  const c = el("div", { class: "card" }, head, el("div", { class: "card-body" }, ...body));
  return c;
}

function table(headers, rows) {
  const thead = el("tr", {}, ...headers.map((h) => el("th", {}, h)));
  const trs = rows.map((r) => el("tr", {}, ...r.map((v) => el("td", {}, v))));
  return el("table", {}, el("thead", {}, thead), el("tbody", {}, ...trs));
}

function renderDetail(r) {
  const out = $("#detail");
  out.textContent = "";
  const v = r.verdict || {}, m = r.metrics || {};
  const vColor = VERDICT_COLORS[(v.verdict || "").toUpperCase()] || "#c9d1d9";

  const headline = el("div", { class: "verdict" }, `${v.verdict} · grade ${v.grade ?? "?"} · score ${Math.round(v.score ?? 0)}`);
  headline.style.color = vColor;
  const kpis = el("div", { class: "kpis muted" },
    `Sharpe ${fmtNum(m.sharpe)} · CAGR ${fmtPct(m.cagr)} · MaxDD ${fmtPct(m.max_drawdown)} · ` +
    `Sortino ${fmtNum(m.sortino)} · Vol ${fmtPct(m.volatility)} · Calmar ${fmtNum(m.calmar)} · ` +
    `PBO ${fmtPct(r.pbo?.pbo, 0)} (${r.pbo?.verdict ?? "—"}) · perm p=${fmtNum(r.permutation?.p_value, 3)} · ` +
    `divers ${fmtNum(r.diversification)}`);
  out.append(card("Verdict & headline", "out-of-sample, CPCV/PBO + permutation", headline, kpis));

  out.append(gatesCard(r.gates || []));
  out.append(card("Equity — portfolio · SPY · legs", "growth of $1, log — click a legend item to toggle",
    equityChart(r.equity)));

  out.append(card("Allocation weights", "",
    table(["Strategy", "Weight"], (r.weights || []).map((w) => [w.name, fmtPct(w.weight)]))));

  out.append(card("Attribution", "return vs Euler risk per leg",
    table(["Strategy", "Weight", "Return %", "Risk %", "Sharpe"],
      (r.attribution || []).map((a) => [a.name, fmtPct(a.weight), fmtPct(a.ret_contrib_pct, 0),
        fmtPct(a.risk_contrib_pct, 0), fmtNum(a.sharpe)]))));

  out.append(metricsCard(r));

  if (r.correlation?.names?.length) out.append(card("Correlation matrix", "allocated strategies", corrTable(r.correlation)));

  if (r.tail && r.tail.overall_corr !== undefined)
    out.append(card("Tail dependence vs SPY", "worst-decile clustering",
      el("div", {}, `Equity beta ${fmtNum(r.tail.beta)} · Overall ρ ${fmtNum(r.tail.overall_corr)} · Tail ρ ${fmtNum(r.tail.tail_corr)}`)));

  out.append(card("Regime-conditional performance", "SPX trend × volatility",
    table(["Regime", "Sharpe", "CAGR", "MaxDD", "Win", "% time"],
      Object.entries(r.regime || {}).map(([k, d]) =>
        [k, fmtNum(d.sharpe), fmtPct(d.cagr), fmtPct(d.max_dd), fmtPct(d.win_rate, 0), fmtPct(d.pct_time, 0)]))));

  out.append(card("Historical stress / crisis decomposition", "fixed weights per window",
    table(["Scenario", "Total ret", "Sharpe", "MaxDD", "CVaR95"],
      (r.stress || []).map((s) => [s.name, fmtPct(s.total_return), fmtNum(s.sharpe), fmtPct(s.max_dd), fmtPct(s.cvar_95, 2)]))));

  out.append(card("Factor exposure", "ETF-proxy OLS, Newey-West t", factorLine(r.factor)));
}

function gatesCard(gates) {
  if (!gates.length) return card("Constraint gates", "", el("div", { class: "muted" }, "No constraints set — book is unconstrained."));
  const passed = gates.filter((g) => g.ok).length;
  const summary = el("div", {}, `${passed}/${gates.length} constraints satisfied`);
  summary.style.color = passed === gates.length ? "#2ea043" : "#f85149";
  const lines = gates.map((g) =>
    el("div", { class: g.ok ? "gate-ok" : "gate-bad" }, `${g.ok ? "✓" : "✗"}  ${g.label}   (actual ${g.actual})`));
  return card("Constraint gates", "strict", summary, ...lines);
}

function factorLine(fr) {
  if (!fr || !fr.factors) return el("div", { class: "muted" }, (fr && fr.note) || "—");
  const bits = [`α ${fmtPct(fr.alpha_ann)} (t=${fmtNum(fr.alpha_t, 1)})`, `R² ${fmtPct(fr.r_squared, 0)}`];
  for (const [k, v] of Object.entries(fr.factors)) bits.push(`${k} ${v.beta >= 0 ? "+" : ""}${fmtNum(v.beta)} (t=${fmtNum(v.t_stat, 1)})`);
  return el("div", {}, bits.join("  ·  "));
}

// Individual-strategy metrics + a bold portfolio row (over the common window).
function metricsCard(r) {
  const m = r.metrics || {};
  const head = el("tr", {}, ...["Strategy", "Weight", "CAGR", "Sharpe", "Vol", "MaxDD"].map((h) => el("th", {}, h)));
  const rows = (r.components || []).map((c) => el("tr", {},
    el("td", {}, c.name), el("td", {}, fmtPct(c.weight)), el("td", {}, fmtPct(c.cagr)),
    el("td", {}, fmtNum(c.sharpe)), el("td", {}, fmtPct(c.vol)), el("td", {}, fmtPct(c.max_dd))));
  const portRow = el("tr", { class: "port-row" },
    el("td", {}, "▸ Portfolio"), el("td", {}, "100%"), el("td", {}, fmtPct(m.cagr)),
    el("td", {}, fmtNum(m.sharpe)), el("td", {}, fmtPct(m.volatility)), el("td", {}, fmtPct(m.max_drawdown)));
  const t = el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows, portRow));
  return card("Strategy vs portfolio metrics", "each leg vs the whole book, over the common window", t);
}

function corrTable(c) {
  const head = el("tr", {}, el("th", {}, ""), ...c.names.map((n) => el("th", {}, n)));
  const rows = c.matrix.map((row, i) => {
    const tds = row.map((val) => {
      const td = el("td", {}, val === null ? "—" : Number(val).toFixed(2));
      if (val !== null) {
        const r = Math.round(180 * Math.max(val, 0)), g = Math.round(150 * Math.max(-val, 0));
        td.style.background = `rgb(${40 + r},${40 + g},50)`;
      }
      return td;
    });
    return el("tr", {}, el("th", {}, c.names[i]), ...tds);
  });
  return el("table", { class: "corr" }, el("thead", {}, head), el("tbody", {}, ...rows));
}

// Equity chart with a toggleable legend: Portfolio + SPY on by default, each leg off.
const _LEG_PALETTE = ["#2ea043", "#db6d28", "#a371f7", "#e3b341", "#f85149",
  "#1abc9c", "#e67e22", "#3498db", "#9b59b6", "#16a085"];

function equityChart(equity) {
  const series = [];
  if (equity?.portfolio?.length) series.push({ name: "Portfolio", pts: equity.portfolio, color: "#1f6feb", on: true });
  if (equity?.benchmark?.length) series.push({ name: "SPY", pts: equity.benchmark, color: "#8b949e", on: true });
  (equity?.strategies || []).forEach((s, i) => {
    if (s.points?.length) series.push({ name: s.name, pts: s.points, color: _LEG_PALETTE[i % _LEG_PALETTE.length], on: false });
  });
  if (!series.length) return el("div", { class: "muted" }, "no data");
  const host = el("div");
  const legend = el("div", { class: "legend" });
  const draw = () => { host.textContent = ""; host.append(equitySvgMulti(series.filter((s) => s.on))); };
  for (const s of series) {
    const tag = el("span", { class: "legend-item" + (s.on ? "" : " off") }, "● " + s.name);
    tag.style.color = s.on ? s.color : "#6e7681";
    tag.addEventListener("click", () => {
      s.on = !s.on;
      tag.classList.toggle("off");
      tag.style.color = s.on ? s.color : "#6e7681";
      draw();
    });
    legend.append(tag);
  }
  draw();
  return el("div", {}, host, legend);
}

// Draw multiple equity series on a shared, date-based x-axis (log y). Lines with
// later start dates begin partway across — so you can see when each leg came online.
function equitySvgMulti(series) {
  const W = 720, H = 260, pad = 34;
  if (!series.length) return el("div", { class: "muted" }, "no series selected");
  let lo = Infinity, hi = -Infinity, tmin = Infinity, tmax = -Infinity;
  for (const s of series) for (const p of s.pts) {
    const t = Date.parse(p.t);
    lo = Math.min(lo, p.v); hi = Math.max(hi, p.v); tmin = Math.min(tmin, t); tmax = Math.max(tmax, t);
  }
  const ly = (v) => Math.log(Math.max(v, 1e-6));
  const yLo = ly(lo), yHi = ly(hi);
  const X = (t) => pad + ((t - tmin) / ((tmax - tmin) || 1)) * (W - 2 * pad);
  const Y = (v) => H - pad - ((ly(v) - yLo) / ((yHi - yLo) || 1)) * (H - 2 * pad);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
  svg.append(svgEl("line", { class: "axis", x1: pad, y1: H - pad, x2: W - pad, y2: H - pad }));
  for (const s of series) {
    const d = s.pts.map((p, i) => `${i === 0 ? "M" : "L"}${X(Date.parse(p.t)).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ");
    svg.append(svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.4 }));
  }
  return svg;
}

// --------------------------------------------------------------------------- //
// Wire up                                                                     //
// --------------------------------------------------------------------------- //
async function init() {
  $("#filter").addEventListener("input", renderRail);
  $("#applyYear").addEventListener("click", renderRail);
  $("#minYear").addEventListener("keydown", (e) => { if (e.key === "Enter") renderRail(); });
  $("#checkShown").addEventListener("click", () => checkShown(true));
  $("#clear").addEventListener("click", () => checkShown(false));
  $("#search").addEventListener("click", search);
  $("#save").addEventListener("click", () => savePortfolio(false));
  $("#export").addEventListener("click", () => savePortfolio(true));
  $("#saved").addEventListener("click", async () => {
    const out = await api("/api/portfolio/saved");
    alert(out.portfolios.length ? out.portfolios.join("\n") : "No saved portfolios yet.");
  });
  $("#maxWeight").addEventListener("input", (e) =>
    $("#maxWeightVal").textContent = e.target.value === "0" ? "none" : e.target.value + "%");

  try {
    STRATEGIES = await api("/api/strategies");
    renderRail();
    setStatus(`${STRATEGIES.length} strategies in the book — select ≥ 2, set criteria, then Search.`);
  } catch (e) {
    $("#strategies").textContent = "Failed to load: " + e.message;
  }
}

init();
