"""Save / load portfolios to the committed ``portfolios/`` folder.

A saved portfolio is ``portfolios/<name>/`` with ``portfolio.json`` (the
definition — strategies by slug, weights, allocation mode, metrics) plus
``artifacts/`` (the combined returns, summary, and an HTML tearsheet). It
references strategies **by slug**, so it always rebuilds from the current
``strategies/`` book. This is to portfolios what the strategy artifacts are to
strategies: committed, visible on GitHub, reproducible.
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from .backtester import BacktestResult
from .candidates import StrategyCandidate

__all__ = ["slugify", "save_portfolio", "load_portfolio", "list_saved", "export_as_strategy"]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "portfolio"


def save_portfolio(
    name: str, selected: list[StrategyCandidate], weights: dict[str, float],
    *, mode: str, result: BacktestResult, portfolio_root: str | Path,
    as_of: str | None = None,
) -> Path:
    """Write ``portfolios/<slug>/`` (portfolio.json + artifacts). Returns the folder."""
    slug = slugify(name)
    folder = Path(portfolio_root) / slug
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    name_to_slug = {c.name: c.slug for c in selected}
    strategies = [{"slug": name_to_slug.get(n, n), "weight": round(float(w), 6)}
                  for n, w in weights.items() if abs(w) > 1e-9]
    definition = {
        "name": name, "slug": slug, "created": date.today().isoformat(),
        "as_of": as_of, "allocation_mode": mode,
        "strategies": sorted(strategies, key=lambda s: -s["weight"]),
        "metrics": {k: round(float(v), 6) for k, v in result.metrics.items()},
        "beta": {k: round(float(v), 6) for k, v in (result.beta or {}).items()
                 if isinstance(v, (int, float))},
    }
    (folder / "portfolio.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    result.portfolio_returns.rename("returns").to_frame().to_csv(
        artifacts / f"{slug}_returns.csv", index_label="date")
    (artifacts / f"{slug}_summary.json").write_text(
        json.dumps({"name": name, "metrics": definition["metrics"],
                    "beta": definition["beta"], "weights": strategies}, indent=2),
        encoding="utf-8")
    (artifacts / f"{slug}_report.html").write_text(
        _tearsheet_html(definition, result), encoding="utf-8")
    return folder


def load_portfolio(name_or_slug: str, portfolio_root: str | Path) -> dict:
    """Load a saved portfolio's definition (``portfolio.json``)."""
    folder = Path(portfolio_root) / slugify(name_or_slug)
    return json.loads((folder / "portfolio.json").read_text(encoding="utf-8"))


def list_saved(portfolio_root: str | Path) -> list[str]:
    """Slugs of every saved portfolio under ``portfolio_root``."""
    root = Path(portfolio_root)
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/portfolio.json"))


# ---------------------------------------------------------------------------
# Export a saved portfolio as a conforming Rigor strategy folder
# ---------------------------------------------------------------------------

def _make_strategy_py(name: str) -> str:
    """Return the source text for a generated portfolio strategy.py.

    We build it via concatenation rather than ``str.format`` because the
    generated code itself contains curly-braces (f-strings, dict literals)
    that would collide with ``format``'s expansion.
    """
    # Use a raw-string body for the generated module so we never have to
    # worry about brace escaping in the Python code sections.
    header = (
        '"""Auto-generated portfolio strategy — '
        + name
        + ".\n\n"
        "This module was produced by ``portfolio_io.export_as_strategy`` from a saved\n"
        "portfolio definition.  It exposes a ``build(config, data)`` factory that\n"
        "reconstructs the portfolio by loading each member strategy's committed\n"
        "``artifacts/<slug>_returns.csv``, aligning on common dates, applying the\n"
        "stored weights, and returning an ``rigor.engine.BacktestResult``.\n\n"
        "DO NOT edit signal logic here — this is a composition layer.  "
        "To update the\nportfolio, re-run the Portfolio Builder and re-export.\n"
        '"""\n'
    )
    body = (
        "from __future__ import annotations\n"
        "\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from rigor.engine import result_from_returns\n"
        "from rigor.strategy import StrategyBase, StrategyConfig\n"
        "\n"
        "\n"
        "def _load_member_returns(repo_root: Path, member_slug: str) -> pd.Series:\n"
        '    """Locate and load a member strategy\'s returns CSV.\n'
        "\n"
        "    Searches ``strategies/<category>/<slug>/artifacts/<slug>_returns.csv`` for all\n"
        "    categories under ``strategies/``.  Raises ``FileNotFoundError`` with a helpful\n"
        "    message if the file cannot be found so the user knows what is missing.\n"
        '    """\n'
        "    pattern = f\"strategies/*/*/artifacts/{member_slug}_returns.csv\"\n"
        "    for candidate in sorted(repo_root.glob(pattern)):\n"
        "        try:\n"
        "            df = pd.read_csv(candidate)\n"
        "        except Exception as exc:\n"
        "            raise FileNotFoundError(\n"
        "                f\"Could not read returns for {member_slug!r} at {candidate}: {exc}\"\n"
        "            ) from exc\n"
        "        idx = pd.to_datetime(df.iloc[:, 0], errors=\"coerce\")\n"
        "        vals = pd.to_numeric(df.iloc[:, 1], errors=\"coerce\")\n"
        "        return pd.Series(vals.to_numpy(), index=idx, name=member_slug).dropna()\n"
        "    raise FileNotFoundError(\n"
        "        f\"Returns file for strategy {member_slug!r} not found under \"\n"
        "        f\"{repo_root / 'strategies'}.  \"\n"
        "        \"Run `rigor run` on that strategy first to generate its artifacts.\"\n"
        "    )\n"
        "\n"
        "\n"
        "class _PortfolioStrategy(StrategyBase):\n"
        '    """Lightweight StrategyBase subclass that replays a weighted combination of\n'
        '    pre-computed member-strategy returns."""\n'
        "\n"
        "    def __init__(self, config: StrategyConfig, data: Any) -> None:\n"
        "        super().__init__(config)\n"
        "        self._data = data  # unused for composition; kept for interface conformance\n"
        "        e = config.extra\n"
        "        self._members: list[dict] = e[\"members\"]  # [{slug, weight}, ...]\n"
        "        self._repo_root = Path(\n"
        "            e.get(\"repo_root\") or Path(__file__).resolve().parents[5]\n"
        "        )\n"
        "\n"
        "    def build_cache(self) -> dict:\n"
        "        # Composition needs no price cache; members are loaded in run_backtest.\n"
        "        return {}\n"
        "\n"
        "    def run_backtest(self, cache: Any, params: Any) -> pd.Series:\n"
        "        series: list[pd.Series] = []\n"
        "        weights: list[float] = []\n"
        "        for m in self._members:\n"
        "            slug = m[\"slug\"]\n"
        "            weight = float(m[\"weight\"])\n"
        "            s = _load_member_returns(self._repo_root, slug)\n"
        "            series.append(s)\n"
        "            weights.append(weight)\n"
        "        if not series:\n"
        "            return pd.Series(dtype=\"float64\", name=\"returns\")\n"
        "        combined = pd.concat(series, axis=1).dropna()\n"
        "        total_w = sum(weights)\n"
        "        norm_w = (\n"
        "            [w / total_w for w in weights]\n"
        "            if total_w > 0\n"
        "            else [1.0 / len(weights)] * len(weights)\n"
        "        )\n"
        "        port = (combined * norm_w).sum(axis=1)\n"
        "        port.name = \"returns\"\n"
        "        return port\n"
        "\n"
        "    def backtest(self, params: dict | None = None) -> \"BacktestResult\":  # type: ignore[name-defined]\n"
        "        rets = self.run_backtest({}, params or {})\n"
        "        return result_from_returns(rets)\n"
        "\n"
        "\n"
        "def build(config: StrategyConfig, data: Any) -> _PortfolioStrategy:\n"
        "    return _PortfolioStrategy(config, data)\n"
    )
    return header + body


def _safe_slug(name: str) -> str:
    """Produce a valid Rigor slug (``[a-z][a-z0-9_]*``) from an arbitrary name.

    Lowercases, replaces any run of non-alphanumeric characters with ``_``,
    strips leading/trailing underscores, and prepends ``p_`` when the result
    starts with a digit (so it always matches ``[a-z][a-z0-9_]*``).
    """
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    s = s or "portfolio"
    if s[0].isdigit():
        s = "p_" + s
    return s


def export_as_strategy(
    portfolio: dict,
    dest_dir: Path | str,
    *,
    slug: str | None = None,
    category: str = "portfolio",
    overwrite: bool = False,
) -> Path:
    """Write a conforming Rigor strategy folder from a saved portfolio definition.

    Parameters
    ----------
    portfolio:
        A saved-portfolio definition dict as returned by ``load_portfolio`` (or the
        ``dict`` written to ``portfolio.json``).  Must contain ``"name"`` and
        ``"strategies"`` (list of ``{slug, weight}`` dicts).
    dest_dir:
        The **parent** directory to write into.  The strategy folder is created at
        ``dest_dir/<slug>/``.  This is **never** implicitly set to the repo's
        ``strategies/`` tree — the caller chooses the location.
    slug:
        Override the auto-derived slug.  Must match ``[a-z][a-z0-9_]*``.  If
        omitted, derived from ``portfolio["name"]`` via ``_safe_slug``.
    category:
        The ``category`` field written into ``config.json`` (default ``"portfolio"``).
    overwrite:
        If ``False`` (default) and the target folder is non-empty, raise
        ``FileExistsError``.  If ``True``, overwrite existing files in place.

    Returns
    -------
    Path
        The created strategy folder (``dest_dir/<slug>/``).
    """
    name: str = portfolio.get("name", "Exported Portfolio")
    members: list[dict] = portfolio.get("strategies", [])

    # Derive / validate slug
    if slug is None:
        slug = _safe_slug(name)
    else:
        from rigor.project.layout import is_valid_slug  # type: ignore[import]

        if not is_valid_slug(slug):
            raise ValueError(
                f"slug {slug!r} is not valid — must match [a-z][a-z0-9_]*"
            )

    folder = Path(dest_dir) / slug

    # Safety guard: refuse to clobber a non-empty folder unless overwrite=True.
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Strategy folder already exists and is non-empty: {folder}.  "
            "Pass overwrite=True to replace it."
        )

    folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. config.json
    # ------------------------------------------------------------------
    config_data: dict = {
        "name": name,
        "slug": slug,
        "category": category,
        "status": "idle",
        "version": "v1",
        "extra": {
            "allocation_mode": portfolio.get("allocation_mode", "manual"),
            "source_portfolio_created": portfolio.get("created"),
            "source_portfolio_as_of": portfolio.get("as_of"),
            "members": [
                {"slug": m["slug"], "weight": round(float(m["weight"]), 8)}
                for m in members
            ],
        },
    }
    (folder / "config.json").write_text(
        json.dumps(config_data, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 2. strategy.py
    # ------------------------------------------------------------------
    (folder / "strategy.py").write_text(_make_strategy_py(name), encoding="utf-8")

    # ------------------------------------------------------------------
    # 3. README.md
    # ------------------------------------------------------------------
    weight_lines = "\n".join(
        f"| `{m['slug']}` | {float(m['weight']):.2%} |" for m in members
    )
    readme = (
        f"# {name}\n\n"
        "This strategy folder was **auto-generated** by `portfolio_io.export_as_strategy`.\n"
        "It is a composition of the member strategies listed below, reweighted according\n"
        "to the stored allocation.\n\n"
        f"- **Category**: `{category}`\n"
        f"- **Slug**: `{slug}`\n"
        f"- **Allocation mode**: `{portfolio.get('allocation_mode', 'manual')}`\n"
        f"- **Members**: {len(members)}\n\n"
        "## Member weights\n\n"
        "| Strategy slug | Weight |\n"
        "| --- | --- |\n"
        f"{weight_lines}\n\n"
        "## Running\n\n"
        "```bash\n"
        f"rigor run strategies/{category}/{slug} --as-of <YYYY-MM-DD>\n"
        "```\n\n"
        "> **Note**: each member strategy must have its `artifacts/<slug>_returns.csv`\n"
        "> committed before running this composed strategy.  Run `rigor run` on each\n"
        "> member first if the artifact is missing.\n"
    )
    (folder / "README.md").write_text(readme, encoding="utf-8")

    return folder


def _equity_png_b64(returns: pd.Series) -> str:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3), tight_layout=True)
    ax = fig.add_subplot(111)
    equity = (1.0 + returns.dropna()).cumprod()
    ax.plot(equity.index, equity.to_numpy(), color="#1f6feb", linewidth=1.3)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log)")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tearsheet_html(definition: dict, result: BacktestResult) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                   for k, v in definition["metrics"].items())
    wrows = "".join(f"<tr><td>{s['slug']}</td><td>{s['weight']:.1%}</td></tr>"
                    for s in definition["strategies"])
    img = _equity_png_b64(result.portfolio_returns)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{definition['name']} — portfolio</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;max-width:860px;margin:2rem auto}}
table{{border-collapse:collapse;margin:1rem 0}}td{{border:1px solid #30363d;padding:4px 12px}}
h1,h2{{color:#f0f6fc}}img{{max-width:100%;border:1px solid #30363d;border-radius:6px}}</style></head>
<body><h1>{definition['name']}</h1>
<p>Allocation: <b>{definition['allocation_mode']}</b> · created {definition['created']}
 · as-of {definition.get('as_of') or 'live'}</p>
<img src="data:image/png;base64,{img}"/>
<h2>Metrics</h2><table>{rows}</table>
<h2>Weights ({len(definition['strategies'])} strategies)</h2><table>{wrows}</table>
</body></html>"""
