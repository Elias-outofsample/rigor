"""Type 2 — Institutional Background & Market Structure section."""

from .base_gen_section import BaseGenSection


class InstitutionalBackgroundSection(BaseGenSection):
    name = "Institutional Background"
    config_field = "institutional_background"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Institutional Background and Market Structure", level=1)
        self._render_paragraphs(builder, config.institutional_background)
        self._render_section_figures(builder, config, "institutional_background")
        self._render_section_tables(builder, config, "institutional_background")
