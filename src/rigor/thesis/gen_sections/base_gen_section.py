"""Base class for generalist thesis sections."""

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gen_thesis_config import GenThesisConfig
    from ..pdf.pdf_builder import ReportLabPDFBuilder

# Regex for display math: $$...$$
_DISPLAY_MATH_RE = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
# Regex for inline math: $...$  (not greedy, no nested $$)
_INLINE_MATH_RE = re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)')


class BaseGenSection(ABC):
    """Abstract base for all generalist thesis sections.

    Each section reads narrative text from GenThesisConfig fields
    and renders it through the ReportLabPDFBuilder API.

    Text conventions supported in section content:
      - ``\\n\\n`` between paragraphs
      - ``## Title`` for level-2 sub-headings
      - ``### Title`` for level-3 sub-headings
      - ``$$...$$`` for display math (rendered via matplotlib)
      - ``$...$`` for inline math (rendered via matplotlib)
      - ``--`` auto-replaced with em-dash
      - ``<b>``, ``<i>`` HTML tags for bold/italic
    """

    name: str = ""
    config_field: str = ""  # GenThesisConfig field name for has_content check

    @abstractmethod
    def render(self, builder: "ReportLabPDFBuilder", config: "GenThesisConfig"):
        """Render this section into the PDF builder."""
        pass

    def has_content(self, config: "GenThesisConfig") -> bool:
        """Return True if this section should be included.
        Override for sections that are always present (title page, etc.).
        """
        if not self.config_field:
            return True
        return config.has_section(self.config_field)

    def _render_paragraphs(self, builder, text: str, style: str = "body"):
        """Split multi-paragraph text on double newlines and render each.

        Supports markdown sub-headings (## / ###), display math ($$...$$),
        inline math ($...$), and em-dash replacement (--).
        """
        if not text or not text.strip():
            return
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Replace single newlines within a paragraph with spaces
            para = para.replace("\n", " ")
            # Replace ASCII double-dash with proper em-dash
            para = para.replace(" -- ", "\u2014")
            para = para.replace("--", "\u2014")

            # Sub-heading detection (### before ## to avoid false match)
            if para.startswith("### "):
                builder.add_heading(para[4:], level=3)
                continue
            if para.startswith("## "):
                builder.add_heading(para[3:], level=2)
                continue

            # Display math: $$...$$
            if '$$' in para:
                self._render_paragraph_with_display_math(builder, para, style)
                continue

            # Display math: $$...$$
            # NOTE: Inline math $...$ is NOT supported (ReportLab <img> breaks text flow).
            # Use display math $$...$$ for equations. For inline symbols, write them in
            # plain text (e.g., "beta_5" instead of "$\beta_5$").

            builder.add_paragraph(para, style)

    def _render_paragraph_with_display_math(self, builder, para: str, style: str):
        """Handle a paragraph containing $$...$$ display math blocks."""
        parts = _DISPLAY_MATH_RE.split(para)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 0:
                # Text segment
                builder.add_paragraph(part, style)
            else:
                # Math segment — render as display equation
                builder.add_display_math(part)

    def _render_section_figures(self, builder, config: "GenThesisConfig",
                                 section_hint: str):
        """Render figures tagged for this section."""
        figures = config.section_figures(section_hint)
        for fig in figures:
            if fig.source_type() == "path" and fig.path:
                width = builder.TEXT_WIDTH * fig.width_fraction if fig.width_fraction < 1.0 else None
                builder.add_figure(fig.path, fig.caption, width=width)
            elif fig.source_type() == "dataframe":
                # Generate chart from DataFrame
                img_path = self._render_dataframe_figure(fig, config)
                if img_path:
                    width = builder.TEXT_WIDTH * fig.width_fraction if fig.width_fraction < 1.0 else None
                    builder.add_figure(img_path, fig.caption, width=width)
                    builder.register_temp_file(img_path)

    def _render_section_tables(self, builder, config: "GenThesisConfig",
                                section_hint: str):
        """Render tables tagged for this section."""
        tables = [t for t in config.tables if t.section_hint == section_hint]
        for tbl in tables:
            if tbl.data is None:
                continue
            data = self._table_data_to_lists(tbl)
            if data:
                # CHANGE-001 — table note: pass through TableSpec.note
                builder.add_table(data, tbl.caption, note=getattr(tbl, "note", ""))

    @staticmethod
    def _table_data_to_lists(tbl) -> list:
        """Convert TableSpec.data to list-of-lists for the builder."""
        data = tbl.data
        if data is None:
            return []
        # If it's a DataFrame, convert
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                header = [""] + list(data.columns)
                rows = []
                for idx, row in data.iterrows():
                    row_data = [str(idx)]
                    for col in data.columns:
                        val = row[col]
                        if col in tbl.format_pct_cols:
                            row_data.append(f"{val * 100:.2f}%")
                        elif col in tbl.format_ratio_cols:
                            row_data.append(f"{val:.2f}")
                        else:
                            row_data.append(str(val))
                    rows.append(row_data)
                return [header] + rows
        except ImportError:
            pass
        # Already list-of-lists
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _render_dataframe_figure(fig, config) -> str:
        """Render a FigureSpec with a dataframe to a temp PNG. Returns path."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import tempfile

            import matplotlib.pyplot as plt

            df = fig.dataframe
            cc = fig.chart_config
            figsize = cc.get("figsize", (10, 5))

            f, ax = plt.subplots(figsize=figsize)
            chart_type = fig.chart_type

            if chart_type == "line":
                x_col = cc.get("x_col")
                y_col = cc.get("y_col")
                if x_col and y_col:
                    ax.plot(df[x_col], df[y_col])
                else:
                    df.plot(ax=ax)
            elif chart_type == "bar":
                df.plot.bar(ax=ax)
            elif chart_type == "scatter":
                x_col = cc.get("x_col", df.columns[0])
                y_col = cc.get("y_col", df.columns[1])
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=20)
            elif chart_type == "heatmap":
                im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn")
                ax.set_xticks(range(len(df.columns)))
                ax.set_xticklabels(df.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(df.index)))
                ax.set_yticklabels(df.index)
                plt.colorbar(im, ax=ax)
            elif chart_type == "distribution":
                for col in df.columns:
                    ax.hist(df[col].dropna(), bins=50, alpha=0.6, label=col)
                ax.legend()

            ax.set_title(cc.get("title", ""), fontsize=11)
            ax.set_xlabel(cc.get("xlabel", ""), fontsize=10)
            ax.set_ylabel(cc.get("ylabel", ""), fontsize=10)
            ax.tick_params(labelsize=9)
            plt.tight_layout()

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.savefig(tmp.name, dpi=config.figure_dpi, bbox_inches="tight")
            plt.close(f)
            return tmp.name
        except Exception:
            return ""
