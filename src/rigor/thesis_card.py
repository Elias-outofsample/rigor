"""Adapter: turn a BacktestResult into an academic thesis PDF.

Bridges a backtest result to the PDF thesis generator (``rigor.thesis``).
We use the content-driven ``GenThesisConfig`` / ``GenThesisGenerator`` path
(``paper_type="empirical_study"``) — narrative text + DataFrame figures + tables,
no CSV contract and no Fama-French web fetch.

The narrative is auto-drafted from the metrics as a publishable skeleton; the
researcher refines it (or supplies overrides via ``config.extra["thesis"]``).
Figures/tables come straight from the BacktestResult, so the numbers in the
thesis are the SAME single-source metrics as the report card.

Note: the report card (HTML) remains the byte-deterministic source of truth.
The PDF embeds a ReportLab creation timestamp, so it is content-stable but not
guaranteed byte-identical across runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import BacktestResult
from .thesis import FigureSpec, GenThesisConfig, GenThesisGenerator, TableSpec


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _draft_narrative(name: str, m: dict, start: str, end: str) -> dict[str, str]:
    """Auto-draft thesis sections from the metrics (a skeleton to refine)."""
    abstract = (
        f"This study documents the historical risk-adjusted performance of the "
        f"{name} strategy over the period {start} to {end}. Across {m['n_obs']:,} "
        f"trading days the strategy realised an annualised Sharpe ratio of "
        f"{m['sharpe']:.2f}, a compound annual growth rate of {_pct(m['cagr'])}, "
        f"and a maximum drawdown of {_pct(m['max_drawdown'])}. All results are "
        f"computed on a single-source, point-in-time, survivorship-bias-free "
        f"dataset with deterministic corporate-action adjustment, and are fully "
        f"reproducible from a pinned data as-of date."
    )
    introduction = (
        "## 1.1 Motivation\n\n"
        f"We evaluate {name} as a candidate for the strategy book. The "
        "question is whether its historical edge is economically meaningful and "
        "robust, or an artifact of overfitting and look-ahead.\n\n"
        "## 1.2 Preview of findings\n\n"
        f"Over the sample the strategy earns a Sharpe ratio of {m['sharpe']:.2f} "
        f"(Sortino {m['sortino']:.2f}, Calmar {m['calmar']:.2f}) with annualised "
        f"volatility of {_pct(m['volatility'])}. _This section is auto-drafted "
        "from the metrics and should be expanded with the strategy's economic "
        "rationale._"
    )
    data_methodology = (
        "## 2.1 Data\n\n"
        "Prices are sourced from a single vendor (EODHD) and loaded through the "
        "framework DataLoader, which applies deterministic split/dividend "
        "adjustment from the raw corporate-action feed and resolves point-in-time "
        "index membership so the investable universe reflects only information "
        "available at each date (survivorship-bias-free).\n\n"
        "## 2.2 Backtest methodology\n\n"
        "Signals are converted to target weights and simulated by the shared "
        "engine: positions are held with a one-bar lag (no look-ahead) and "
        "turnover is charged a per-rebalance cost. Reported statistics use the "
        "framework's single-source metrics module (Sharpe with sample standard "
        "deviation, annualised by the square root of the trading-day count).\n\n"
        "$$\\mathrm{Sharpe} = \\frac{\\bar{r}}{\\sigma_r}\\,\\sqrt{P}$$"
    )
    results_analysis = (
        "Table 1 summarises the headline statistics and Figures 1–3 present the "
        "equity curve, drawdown profile, and the distribution of daily returns.\n\n"
        f"The strategy compounds at {_pct(m['cagr'])} per annum with a worst "
        f"peak-to-trough decline of {_pct(m['max_drawdown'])}, implying a Calmar "
        f"ratio of {m['calmar']:.2f}. The realised win rate is {_pct(m['win_rate'])}. "
        "_Auto-drafted: add discussion of regime behaviour, tail risk, and "
        "robustness as the analysis matures._"
    )
    conclusion = (
        f"The {name} strategy delivers a historically attractive, reproducible "
        "risk-adjusted return on a survivorship-bias-free sample. Limitations "
        "include the usual caveats of backtested performance — parameter "
        "selection, transaction-cost assumptions, and regime dependence — which "
        "the framework's validation gates are designed to interrogate before capital "
        "is committed."
    )
    return {
        "abstract": abstract,
        "introduction": introduction,
        "data_methodology": data_methodology,
        "results_analysis": results_analysis,
        "conclusion": conclusion,
    }


def build_thesis_config(
    result: BacktestResult, *, name: str, category: str,
    as_of: str | None = None, author: str = "rigor",
    overrides: dict | None = None,
) -> GenThesisConfig:
    m = result.metrics
    r = result.returns
    start = str(r.index[0].date()) if len(r) else "?"
    end = str(r.index[-1].date()) if len(r) else "?"
    date_str = (pd.Timestamp(as_of) if as_of else r.index[-1]).strftime("%B %Y")

    narrative = _draft_narrative(name, m, start, end)
    narrative.update(overrides or {})

    summary = TableSpec(
        caption="Performance summary statistics over the full backtest period.",
        section_hint="results_analysis",
        data=[
            ["Metric", "Value"],
            ["Period", f"{start} to {end}"],
            ["Observations", f"{m['n_obs']:,}"],
            ["CAGR", _pct(m["cagr"])],
            ["Annualised volatility", _pct(m["volatility"])],
            ["Sharpe ratio", f"{m['sharpe']:.2f}"],
            ["Sortino ratio", f"{m['sortino']:.2f}"],
            ["Calmar ratio", f"{m['calmar']:.2f}"],
            ["Maximum drawdown", _pct(m["max_drawdown"])],
            ["Total return", _pct(m["total_return"])],
            ["Win rate", _pct(m["win_rate"])],
        ],
    )

    equity_df = result.equity.to_frame("Equity")
    underwater_df = (result.equity / result.equity.cummax() - 1.0).to_frame("Drawdown")
    returns_df = r.to_frame("Daily return")
    figures = [
        FigureSpec(caption="Cumulative equity curve (growth of $1 invested).",
                   dataframe=equity_df, chart_type="line", section_hint="results_analysis",
                   chart_config={"ylabel": "Equity", "xlabel": "Date"}),
        FigureSpec(caption="Drawdown (underwater) curve.",
                   dataframe=underwater_df, chart_type="line", section_hint="results_analysis",
                   chart_config={"ylabel": "Drawdown", "xlabel": "Date"}),
        FigureSpec(caption="Distribution of daily returns.",
                   dataframe=returns_df, chart_type="distribution", section_hint="results_analysis",
                   chart_config={"xlabel": "Daily return", "ylabel": "Frequency"}),
    ]

    return GenThesisConfig(
        title=name,
        subtitle=f"A standardized empirical study — {category.replace('_', ' ')}",
        author=author,
        institution="rigor research framework",
        paper_type="empirical_study",
        date=date_str,
        keywords=[category.replace("_", " "), "backtest", "risk-adjusted returns"],
        abstract=narrative["abstract"],
        introduction=narrative["introduction"],
        data_methodology=narrative["data_methodology"],
        results_analysis=narrative["results_analysis"],
        conclusion=narrative["conclusion"],
        tables=[summary],
        figures=figures,
    )


def write_thesis(
    result: BacktestResult, out_dir: str | Path, slug: str, *,
    name: str, category: str, as_of: str | None = None,
    author: str = "rigor", overrides: dict | None = None,
) -> Path:
    """Generate ``<slug>_thesis.pdf`` in ``out_dir`` and return its path."""
    config = build_thesis_config(
        result, name=name, category=category, as_of=as_of,
        author=author, overrides=overrides,
    )
    out_path = Path(out_dir) / f"{slug}_thesis.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    GenThesisGenerator(config).generate(output_path=str(out_path))
    return out_path
