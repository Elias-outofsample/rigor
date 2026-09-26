"""Type 2 — Implications section (trading, regulatory, research)."""

from .base_gen_section import BaseGenSection


class ImplicationsSection(BaseGenSection):
    name = "Implications"
    config_field = "implications"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Implications", level=1)
        self._render_paragraphs(builder, config.implications)
        self._render_section_figures(builder, config, "implications")
        self._render_section_tables(builder, config, "implications")
