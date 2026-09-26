"""
Generalist Thesis Configuration
================================
Input structures for the GenThesisGenerator.

Two layers:
  - GenThesisBrief   : lightweight input the user/Claude fills out
  - GenThesisConfig  : enriched config produced after Claude has drafted
                       all narrative sections
  - FigureSpec       : unified figure descriptor (path / DataFrame / screenshot)
  - TableSpec        : unified table descriptor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# FigureSpec — unified figure input (three modes)
# ---------------------------------------------------------------------------

@dataclass
class FigureSpec:
    """
    Describes a figure to embed in the thesis.

    Exactly one of (path, dataframe) should be set.
    If both are empty the figure is skipped with a warning.

    Modes:
      path      — existing PNG / JPG / SVG / WEBP file (or screenshot)
      dataframe — pd.DataFrame to render as a chart (chart_type required)
    """

    caption: str = ""
    number: int = 0              # 0 = auto-numbered by the generator

    # --- Mode A : external file ---
    path: str = ""

    # --- Mode B : DataFrame → chart ---
    dataframe: Any = None        # pd.DataFrame
    chart_type: str = "line"     # "line" | "bar" | "scatter" | "heatmap" | "distribution"
    chart_config: dict[str, Any] = field(default_factory=dict)
    # chart_config keys (all optional):
    #   x_col, y_col, color_col (for scatter/line)
    #   title, xlabel, ylabel
    #   figsize (tuple)
    #   palette ("seaborn" | "academic")

    # --- Placement ---
    section_hint: str = ""       # e.g. "stylized_facts" — which section it belongs to
    width_fraction: float = 1.0  # 1.0 = full text width, 0.7 = 70%

    def is_valid(self) -> bool:
        return bool(self.path) or (self.dataframe is not None)

    def source_type(self) -> str:
        if self.path:
            return "path"
        if self.dataframe is not None:
            return "dataframe"
        return "empty"


# ---------------------------------------------------------------------------
# TableSpec
# ---------------------------------------------------------------------------

@dataclass
class TableSpec:
    """
    Describes a data table to embed in the thesis.

    data can be:
      - list of lists  (raw)
      - pd.DataFrame   (auto-converted to list of lists)
    """

    caption: str = ""
    number: int = 0              # 0 = auto-numbered

    data: Any = None             # list[list] or pd.DataFrame
    format_pct_cols: list[str] = field(default_factory=list)
    # Columns that contain fractions to display as "x.xx%"
    format_ratio_cols: list[str] = field(default_factory=list)
    # Columns that contain ratios to display with 2 decimal places
    highlight_rows: list[int] = field(default_factory=list)
    # Row indices (0-based, header excluded) to bold/highlight

    section_hint: str = ""

    # CHANGE-001 — table note
    note: str = ""
    # Optional italic footnote rendered under the table in small font
    # (JFE/RFS academic convention).


# ---------------------------------------------------------------------------
# FormalHypothesis
# ---------------------------------------------------------------------------

# CHANGE-002 — multi-appendices: optional structured appendix container
@dataclass
class AppendixSpec:
    """One labeled appendix (Appendix A, B, C, ...) with title, body, and optional figures/tables.

    Per CHANGE-002 (multi-appendices support), a thesis can specify a list of
    AppendixSpec objects in GenThesisConfig.appendices. Each appendix is rendered
    as a labeled, top-level appendix with auto-assigned letter (A, B, C, ...).

    Backward-compatible: if config.appendices is empty, the existing AppendixGenSection
    falls back to dumping unassigned figures+tables in a single "Appendix" block.
    """
    title: str = ""               # Heading text (e.g., "Functional Non-Stationarity Test")
    body: str = ""                # Main narrative (paragraphs, math, sub-headings ## / ###)
    label: str = ""               # Optional explicit label (e.g., "app:hd"); auto-letter if empty
    figures: list[FigureSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)


@dataclass
class FormalHypothesis:
    label: str               # "H1", "H2", ...
    null_text: str           # "The variance risk premium is zero..."
    alt_text: str            # "The VRP is significantly positive..."


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    authors: str             # "Bollerslev, T., Tauchen, G. and Zhou, H."
    year: int
    title: str
    journal: str = ""
    doi: str = ""
    note: str = ""           # e.g. "Working paper"

    def apa_inline(self) -> str:
        """Return inline citation string, e.g. 'Bollerslev et al. (2009)'."""
        author_parts = self.authors.split(",")
        if len(author_parts) > 2:
            first = author_parts[0].strip().split()[-1]
            return f"{first} et al. ({self.year})"
        first = author_parts[0].strip().split()[-1]
        return f"{first} ({self.year})"


# ---------------------------------------------------------------------------
# GenThesisBrief — what the user hands to Claude
# ---------------------------------------------------------------------------

@dataclass
class GenThesisBrief:
    """
    Lightweight input structure that the user (or user+Claude Code session)
    fills out before Claude drafts the full narrative.

    This is the 'briefing document' — not the final config.
    GenThesisGenerator accepts either a GenThesisBrief (auto-drafts via Claude)
    or a fully populated GenThesisConfig (skip drafting).
    """

    # === Paper type ===
    paper_type: str = "market_phenomenon"
    # "market_phenomenon" | "factor_anomaly" | "empirical_study" | "strategy_backtest"

    # === Identity ===
    title: str = ""
    subtitle: str = ""
    author: str = ""
    institution: str = ""
    date: str = ""            # Auto = current month/year

    # === Core inputs ===

    # For strategy_backtest: path to backtest output directory
    strategy_dir: str = ""
    # Equivalent to ThesisConfig.strategy_dir — the generator loads CSV backtest outputs

    # For concept/phenomenon/factor papers: user's free-form description
    # This is what you tell Claude: "here is what I found empirically..."
    user_description: str = ""
    # Free-form: describe the phenomenon, data, period, key findings, intuition

    key_findings: list[str] = field(default_factory=list)
    # Bullet-form findings, e.g.:
    # ["The variance risk premium is 3x larger post-2022",
    #  "The effect is concentrated in the first 30 minutes post-open",
    #  "Conditioning on overnight gap sign explains 40% of VRP variation"]

    data_description: str = ""
    # What data: source, period, frequency, universe, exclusions, preprocessing

    methodology_notes: str = ""
    # How you tested: statistical tests used, robustness checks run, caveats

    # === Hypotheses ===
    hypotheses: list[FormalHypothesis] = field(default_factory=list)

    # === References hint ===
    references_hint: list[Reference] = field(default_factory=list)
    # Key papers to cite (Claude will integrate into appropriate sections)

    # === Figures and tables ===
    figures: list[FigureSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)

    # === Drafting instructions ===
    tone: str = "academic"
    # "academic" = JFE/RFS style | "practitioner" = Risk/JPM style
    target_length: str = "standard"
    # "short" (~10p) | "standard" (~20p) | "full" (~35p)
    language: str = "en"
    # "en" | "fr"

    # === Optional extra sections ===
    include_appendix: bool = True
    include_robustness: bool = True
    include_implications: bool = True   # Type 2 only


# ---------------------------------------------------------------------------
# GenThesisConfig — the fully drafted config ready for PDF generation
# ---------------------------------------------------------------------------

@dataclass
class GenThesisConfig:
    """
    Complete configuration for a generalist thesis PDF.
    Typically produced by Claude Code after reading a GenThesisBrief.

    Sections are plain text (or light HTML for <b>, <i> tags).
    Multi-paragraph text should use \\n\\n between paragraphs.
    """

    # === Identity ===
    title: str = ""
    subtitle: str = ""
    author: str = ""
    institution: str = ""
    date: str = ""
    paper_type: str = "market_phenomenon"

    # === Universal sections (all paper types) ===
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    # JEL codes optional
    jel_codes: list[str] = field(default_factory=list)

    introduction: str = ""
    # 3-5 paragraphs: motivation → preview of findings → contribution → roadmap

    # Data & methodology (universal, always present)
    data_methodology: str = ""
    # Data source, sample period, variables constructed, statistical tests

    robustness: str = ""
    discussion: str = ""
    conclusion: str = ""
    # Summary, limitations, future work — auto-generated if empty

    # === Type 2 — Market phenomenon ===
    institutional_background: str = ""
    # Market structure, institutional context, why this setting matters

    stylized_facts: str = ""
    # The empirical patterns: numbers, tables, significance, sub-period breakdowns

    mechanism: str = ""
    # Economic explanation: who trades, why, what creates the pattern

    implications: str = ""
    # For traders, market makers, regulators, researchers

    # === Type 3 — Factor / anomaly ===
    related_literature: str = ""
    theoretical_framework: str = ""
    portfolio_formation: str = ""
    risk_adjusted_results: str = ""
    cost_capacity: str = ""

    # === Type 4 — Short empirical study ===
    results_analysis: str = ""

    # === Type 1 — Strategy backtest (passes through to existing generator) ===
    strategy_dir: str = ""
    strategy_type: str = "equity"   # "equity" | "fx_research"
    benchmark_ticker: str = "SPY"

    # === Hypotheses ===
    hypotheses: list[FormalHypothesis] = field(default_factory=list)

    # === Media ===
    figures: list[FigureSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)

    # === References ===
    references: list[Reference] = field(default_factory=list)

    # === Multi-appendices (CHANGE-002) ===
    # Optional list of structured appendices (Appendix A, B, C, ...).
    # If empty, AppendixGenSection falls back to its original single-block behaviour
    # (dumping all unassigned figures+tables under one "Appendix" heading).
    appendices: list[AppendixSpec] = field(default_factory=list)

    # === Drafting metadata (carried from brief) ===
    tone: str = "academic"
    language: str = "en"

    # === Display ===
    number_format_pct: int = 2
    figure_dpi: int = 150

    def __post_init__(self):
        # No wall-clock date anywhere: the caller passes an as-of-derived date, so theses
        # are deterministic. Empty fallback keeps output reproducible if a date is omitted.
        self.date = self.date or ""

    # --- Convenience helpers ---

    def has_section(self, field_name: str) -> bool:
        """Return True if the named section has non-empty content."""
        val = getattr(self, field_name, "")
        return bool(str(val).strip())

    def section_figures(self, section_hint: str) -> list[FigureSpec]:
        """Return figures tagged for a specific section."""
        tagged = [f for f in self.figures if f.section_hint == section_hint]
        untagged = [f for f in self.figures if not f.section_hint]
        return tagged if tagged else []

    def all_figures_unassigned(self) -> list[FigureSpec]:
        """Figures without a section_hint (rendered in Appendix or end of last main section)."""
        return [f for f in self.figures if not f.section_hint]

    def ordered_references_apa(self) -> list[str]:
        """Return references as formatted APA strings, sorted by author+year."""
        refs = sorted(self.references, key=lambda r: (r.authors.split(",")[0].strip(), r.year))
        lines = []
        for r in refs:
            parts = [f"{r.authors} ({r.year}). {r.title}."]
            if r.journal:
                parts.append(f"<i>{r.journal}</i>.")
            if r.doi:
                parts.append(f"DOI: {r.doi}")
            if r.note:
                parts.append(r.note)
            lines.append(" ".join(parts))
        return lines
