"""Standardized strategy report card.

Every Rigor strategy emits the SAME card from its BacktestResult, so three people
reviewing three strategies read the same layout, the same metrics, computed the
same way (via rigor.metrics). Self-contained single-file HTML (Chart.js from CDN)
plus a machine-readable JSON summary.

Design follows an institutional light theme (adapted from an earlier in-house tool), but the code
is clean-room and dependency-light (numpy/pandas only) and DETERMINISTIC — no
wall-clock timestamps embedded, so the same result yields the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as _metrics
from .engine import BacktestResult

_CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"


# --------------------------------------------------------------------------
# Payload (deterministic series the card renders)
# --------------------------------------------------------------------------
def build_payload(
    result: BacktestResult, name: str, *, as_of: str | None = None,
    validation: dict | None = None,
) -> dict:
    r = result.returns
    idx = r.index
    has_dates = isinstance(idx, pd.DatetimeIndex)
    labels = [d.strftime("%Y-%m-%d") for d in idx] if has_dates else [str(i) for i in idx]
    ppy = _metrics.periods_per_year_of(idx) if has_dates else _metrics.TRADING_DAYS

    equity = result.equity
    underwater = (equity / equity.cummax() - 1.0)

    win = min(ppy, max(20, len(r) // 5)) if len(r) else 20
    roll = (r.rolling(win).mean() / r.rolling(win).std(ddof=1) * np.sqrt(ppy))
    roll = roll.replace([np.inf, -np.inf], np.nan)

    annual_years, annual_vals = [], []
    monthly_table: dict[str, list] = {}
    if has_dates:
        ann = r.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
        annual_years = [d.year for d in ann.index]
        annual_vals = [round(v, 6) for v in ann.to_numpy()]
        m = r.resample("ME").apply(lambda x: float((1 + x).prod() - 1))
        for ts, val in m.items():
            row = monthly_table.setdefault(str(ts.year), [None] * 12)
            row[ts.month - 1] = round(float(val), 6)

    hist_counts, hist_edges = np.histogram(r.to_numpy(), bins=40)
    centers = ((hist_edges[:-1] + hist_edges[1:]) / 2.0)

    # Deep diagnostics (EVT tail, risk battery, crisis decomposition, capacity) from
    # rigor.analysis. Bulletproof: any failure just omits the section — it must NEVER
    # break a report card (the whole catalog re-renders through this path).
    diag_html, diag_summary = "", None
    try:
        from .analysis.diagnostics import (
            diagnostics_summary,
            diagnostics_to_html,
            strategy_diagnostics,
        )
        _diag = strategy_diagnostics(r)
        diag_html = diagnostics_to_html(_diag)
        diag_summary = diagnostics_summary(_diag)
    except Exception:  # noqa: BLE001 — diagnostics are additive; never fatal
        pass

    return {
        "name": name,
        "as_of": as_of,
        "diagnostics_html": diag_html,
        "diagnostics_summary": diag_summary,
        "start": labels[0] if labels else None,
        "end": labels[-1] if labels else None,
        "ppy": int(ppy),
        "metrics": result.metrics,
        "equity": {"labels": labels, "values": [round(v, 6) for v in equity.to_numpy()]},
        "drawdown": {"labels": labels, "values": [round(v, 6) for v in underwater.to_numpy()]},
        "rolling_sharpe": {
            "labels": labels,
            "values": [None if not np.isfinite(v) else round(v, 4) for v in roll.to_numpy()],
        },
        "annual": {"years": annual_years, "values": annual_vals},
        "monthly": {"table": monthly_table},
        "distribution": {
            "centers": [round(float(c), 5) for c in centers],
            "counts": [int(c) for c in hist_counts],
        },
        "validation": validation,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:#fafafa;color:#0f172a;font-size:14px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px}
.header{padding:24px 0;border-bottom:1px solid #d1d5db;margin-bottom:28px}
.header h1{font-size:22px;font-weight:600;letter-spacing:-.01em}
.header .sub{color:#475569;font-size:12px;margin-top:6px;font-family:'JetBrains Mono',ui-monospace,monospace}
.section{margin-bottom:40px}
.section h2{font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;padding-bottom:10px;margin-bottom:16px;border-bottom:1px solid #d1d5db}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.kpi{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px}
.kpi .label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;font-weight:600}
.kpi .value{font-size:22px;font-weight:600;margin-top:6px;font-family:'JetBrains Mono',ui-monospace,monospace}
.pos{color:#047857}.neg{color:#b91c1c}
.chart-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px}
table.mret{border-collapse:collapse;width:100%;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px}
table.mret th,table.mret td{padding:5px 7px;text-align:right;border:1px solid #eef0f3}
table.mret th{color:#475569;font-weight:600;background:#f8fafc}
.verdict{display:flex;align-items:center;gap:16px;margin-bottom:16px}
.badge{font-size:15px;font-weight:700;letter-spacing:.04em;padding:8px 16px;border-radius:8px;color:#fff}
.badge.promote{background:#047857}.badge.conditional{background:#b45309}.badge.reject{background:#b91c1c}
.vmeta{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:12px;color:#475569}
table.gates{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
table.gates th,table.gates td{padding:6px 10px;text-align:left;border-bottom:1px solid #eef0f3}
table.gates th{color:#475569;font-weight:600}
table.gates td.s{text-align:center;font-weight:700;font-family:'JetBrains Mono',ui-monospace,monospace}
.gpass{color:#047857}.gfail{color:#b91c1c}.gna{color:#94a3b8}
.aogrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.aocard{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px}
.aocard .l{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;font-weight:600}
.aocard .v{font-size:16px;font-weight:600;margin-top:4px;font-family:'JetBrains Mono',ui-monospace,monospace}
"""

_JS = """
const D = window.Rigor;
const fmtPct = v => v==null ? '' : (v*100).toFixed(2)+'%';
function line(id,labels,values,color,fill){
  new Chart(document.getElementById(id),{type:'line',
    data:{labels:labels,datasets:[{data:values,borderColor:color,backgroundColor:fill||'transparent',
      fill:!!fill,borderWidth:1.4,pointRadius:0,tension:0}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{maxTicksLimit:10,font:{size:10}},grid:{display:false}},
              y:{ticks:{font:{size:10}},grid:{color:'#f1f5f9'}}}}});
}
function bars(id,labels,values){
  new Chart(document.getElementById(id),{type:'bar',
    data:{labels:labels,datasets:[{data:values,backgroundColor:values.map(v=>v>=0?'#047857':'#b91c1c')}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{font:{size:10}},grid:{display:false}},y:{ticks:{font:{size:10}},grid:{color:'#f1f5f9'}}}}});
}
line('eq',D.equity.labels,D.equity.values,'#1e3a8a');
line('dd',D.drawdown.labels,D.drawdown.values,'#b91c1c','rgba(185,28,28,0.08)');
line('rs',D.rolling_sharpe.labels,D.rolling_sharpe.values,'#0891b2');
bars('annual',D.annual.years,D.annual.values);
bars('dist',D.distribution.centers,D.distribution.counts);
"""


def _kpi(label: str, value: str, cls: str = "") -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>'


def _monthly_table(table: dict) -> str:
    if not table:
        return "<p style='color:#94a3b8'>No dated returns.</p>"
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    head = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Year</th></tr>"
    rows = []
    for year in sorted(table):
        vals = table[year]
        cells = []
        prod = 1.0
        for v in vals:
            if v is None:
                cells.append("<td></td>")
            else:
                prod *= 1 + v
                cls = "pos" if v >= 0 else "neg"
                cells.append(f'<td class="{cls}">{v*100:.1f}</td>')
        ytd = prod - 1.0
        ycls = "pos" if ytd >= 0 else "neg"
        rows.append(f"<tr><th>{year}</th>{''.join(cells)}<td class='{ycls}'>{ytd*100:.1f}</td></tr>")
    return f"<table class='mret'>{head}{''.join(rows)}</table>"


def _validation_section(v: dict | None) -> str:
    if not v:
        return ""
    vd = v["verdict"]
    o = v.get("overfit", {})
    nw = o.get("newey_west", {})
    cls = {"PROMOTE": "promote", "CONDITIONAL": "conditional", "REJECT": "reject"}.get(
        vd["verdict"], "reject"
    )

    def _ao(label, val):
        return f'<div class="aocard"><div class="l">{label}</div><div class="v">{val}</div></div>'

    ao = "".join([
        _ao("PSR", f"{o.get('psr', 0):.3f}"),
        _ao("DSR P(skill)", f"{o.get('dsr', 0):.3f}"),
        _ao("Harvey t", f"{o.get('harvey_t', 0):.2f}"),
        _ao("Haircut", f"{o.get('haircut_pct', 0) * 100:.0f}%"),
        _ao("NW Sharpe", f"{nw.get('nw_sharpe', 0):.2f}"),
        _ao("Trials", str(o.get("n_trials", 1))),
    ])

    rows = []
    for g in vd["gates"]:
        if g["neutral"]:
            mark, mc = "n/a", "gna"
        elif g["passed"]:
            mark, mc = "PASS", "gpass"
        else:
            mark, mc = "FAIL", "gfail"
        val = g["value"]
        vals = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
        name = g["name"] + (" *" if g["critical"] else "")
        rows.append(f"<tr><td class='s {mc}'>{mark}</td><td>{name}</td><td>{vals}</td></tr>")
    gates = ("<table class='gates'><tr><th></th><th>Gate</th><th>Value</th></tr>"
             + "".join(rows) + "</table>")
    return (
        '<div class="section"><h2>Validation &amp; Verdict</h2>'
        f'<div class="verdict"><span class="badge {cls}">{vd["verdict"]}</span>'
        f'<span class="vmeta">grade {vd["grade"]} &nbsp;&middot;&nbsp; score {vd["score"]}/100'
        f' &nbsp;&middot;&nbsp; * = critical gate</span></div>'
        f'<div class="aogrid">{ao}</div>{gates}</div>'
    )


def render_html(payload: dict) -> str:
    m = payload["metrics"]
    sub = f"{payload['start']} → {payload['end']}  ·  {m['n_obs']} bars"
    if payload.get("as_of"):
        sub += f"  ·  data as-of {payload['as_of']}"

    def signed(v):
        return "pos" if v >= 0 else "neg"

    _ev_annual = m.get("ev_annual")
    _ev_str = f"{_ev_annual * 100:.2f}%" if _ev_annual is not None else "—"
    _ev_cls = (signed(_ev_annual) if _ev_annual is not None else "")

    # Expected excess return (E[R] − risk-free) and Gaussian (delta-normal) VaR.
    # Guarded with .get so reports render from older summaries that predate these keys.
    _exc = m.get("ev_excess_annual")
    _exc_str = f"{_exc * 100:.2f}%" if _exc is not None else "—"
    _exc_cls = (signed(_exc) if _exc is not None else "")

    _vn = m.get("var_normal_95")
    _vn_str = f"{_vn * 100:.2f}%" if _vn is not None else "—"

    kpis = "".join([
        _kpi("Sharpe", f"{m['sharpe']:.2f}", signed(m["sharpe"])),
        _kpi("CAGR", f"{m['cagr']*100:.2f}%", signed(m["cagr"])),
        _kpi("Expected Value", _ev_str, _ev_cls),
        _kpi("Excess Return", _exc_str, _exc_cls),
        _kpi("Max Drawdown", f"{m['max_drawdown']*100:.2f}%", "neg"),
        _kpi("Volatility", f"{m['volatility']*100:.2f}%"),
        _kpi("VaR 95% (Normal)", _vn_str, "neg"),
        _kpi("Sortino", f"{m['sortino']:.2f}", signed(m["sortino"])),
        _kpi("Calmar", f"{m['calmar']:.2f}", signed(m["calmar"])),
        _kpi("Total Return", f"{m['total_return']*100:.1f}%", signed(m["total_return"])),
        _kpi("Win Rate", f"{m['win_rate']*100:.1f}%"),
    ])

    def chart_section(title, cid, height=260):
        return (f'<div class="section"><h2>{title}</h2>'
                f'<div class="chart-card"><div style="height:{height}px">'
                f'<canvas id="{cid}"></canvas></div></div></div>')

    body = f"""
<div class="wrap">
  <div class="header"><h1>{payload['name']}</h1><div class="sub">{sub}</div></div>
  <div class="section"><h2>Key Metrics</h2><div class="kpis">{kpis}</div></div>
  {_validation_section(payload.get('validation'))}
  {chart_section('Equity Curve', 'eq', 300)}
  {chart_section('Drawdown', 'dd', 200)}
  {chart_section('Rolling Sharpe', 'rs', 200)}
  {chart_section('Annual Returns (%)', 'annual', 220)}
  <div class="section"><h2>Monthly Returns (%)</h2><div class="chart-card">{_monthly_table(payload['monthly']['table'])}</div></div>
  {chart_section('Return Distribution', 'dist', 220)}
  {payload.get('diagnostics_html', '')}
</div>
"""
    # The diagnostics HTML is rendered above; keep it out of the embedded JS payload
    # (which only feeds the Chart.js charts) to avoid duplicating a few KB.
    payload_js = {k: v for k, v in payload.items() if k != "diagnostics_html"}
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{payload['name']} — Rigor Report Card</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap' rel='stylesheet'>"
        f"<style>{_CSS}</style></head><body>{body}"
        f"<script src='{_CHARTJS}'></script>"
        f"<script>window.Rigor={json.dumps(payload_js, separators=(',', ':'))};</script>"
        f"<script>{_JS}</script></body></html>"
    )


def write_report_card(
    result: BacktestResult, out_dir: str | Path, slug: str, *,
    name: str | None = None, as_of: str | None = None, validation: dict | None = None,
) -> dict[str, Path]:
    """Write ``<slug>_report.html`` + ``<slug>_summary.json``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(result, name or slug, as_of=as_of, validation=validation)
    html_path = out / f"{slug}_report.html"
    json_path = out / f"{slug}_summary.json"
    html_path.write_text(render_html(payload), encoding="utf-8")
    summary = {"name": payload["name"], "as_of": as_of, "start": payload["start"],
               "end": payload["end"], "metrics": payload["metrics"]}
    if validation:
        summary["verdict"] = validation["verdict"]
    if payload.get("diagnostics_summary"):
        summary["diagnostics"] = payload["diagnostics_summary"]
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"html": html_path, "json": json_path}
