"""Generalist Appendix — renders unassigned figures and tables.

CHANGE-002 — multi-appendices: if config.appendices is non-empty, render each
AppendixSpec as a labeled appendix (Appendix A, B, C, ...) with its own title,
body, figures, and tables. Otherwise fall back to the original behaviour
(single "Appendix" block dumping unassigned content).
"""

from .base_gen_section import BaseGenSection


class AppendixGenSection(BaseGenSection):
    name = "Appendix"

    def has_content(self, config) -> bool:
        # CHANGE-002 — multi-appendices: also include if any structured appendix exists
        if getattr(config, "appendices", None):
            return True
        return bool(config.all_figures_unassigned()) or bool(
            [t for t in config.tables if not t.section_hint]
        )

    def render(self, builder, config):
        builder.add_page_break()

        # CHANGE-002 — multi-appendices: structured path
        appendices = getattr(config, "appendices", []) or []
        if appendices:
            self._render_structured_appendices(builder, config, appendices)
            # Also render unassigned figures/tables (if any) as a final "Additional Material"
            unassigned_figs = config.all_figures_unassigned()
            unassigned_tbls = [t for t in config.tables if not t.section_hint]
            if unassigned_figs or unassigned_tbls:
                self._render_unassigned_dump(
                    builder, config, unassigned_figs, unassigned_tbls,
                    heading="Additional Material",
                )
            return

        # Fallback: original single-block behaviour
        builder.add_heading("Appendix", level=1)
        self._render_unassigned_dump(
            builder, config,
            config.all_figures_unassigned(),
            [t for t in config.tables if not t.section_hint],
            heading=None,
        )

    # CHANGE-002 — multi-appendices: helper for structured appendices
    def _render_structured_appendices(self, builder, config, appendices):
        for idx, app in enumerate(appendices):
            letter = chr(ord("A") + idx)
            heading = f"Appendix {letter}: {app.title}" if app.title else f"Appendix {letter}"
            if idx > 0:
                builder.add_page_break()
            builder.add_heading(heading, level=1)
            if app.body:
                self._render_paragraphs(builder, app.body)
            # Render appendix-specific figures
            for fig in (app.figures or []):
                if fig.source_type() == "path" and fig.path:
                    width = (
                        builder.TEXT_WIDTH * fig.width_fraction
                        if fig.width_fraction < 1.0 else None
                    )
                    builder.add_figure(fig.path, fig.caption, width=width)
            # Render appendix-specific tables
            for tbl in (app.tables or []):
                data = self._table_data_to_lists(tbl)
                if data:
                    # CHANGE-001 — pass through TableSpec.note
                    builder.add_table(data, tbl.caption, note=getattr(tbl, "note", ""))

    # CHANGE-002 — multi-appendices: helper for the original dump path
    def _render_unassigned_dump(self, builder, config, figs, tbls, heading=None):
        if heading:
            builder.add_heading(heading, level=1)
        for fig in figs:
            if fig.source_type() == "path" and fig.path:
                width = (
                    builder.TEXT_WIDTH * fig.width_fraction
                    if fig.width_fraction < 1.0
                    else None
                )
                builder.add_figure(fig.path, fig.caption, width=width)
            elif fig.source_type() == "dataframe":
                img_path = self._render_dataframe_figure(fig, config)
                if img_path:
                    width = (
                        builder.TEXT_WIDTH * fig.width_fraction
                        if fig.width_fraction < 1.0
                        else None
                    )
                    builder.add_figure(img_path, fig.caption, width=width)
                    builder.register_temp_file(img_path)
        for tbl in tbls:
            data = self._table_data_to_lists(tbl)
            if data:
                # CHANGE-001 — pass through TableSpec.note
                builder.add_table(data, tbl.caption, note=getattr(tbl, "note", ""))
