"""Type 2 — Economic Mechanism & Interpretation section."""

from .base_gen_section import BaseGenSection


class MechanismSection(BaseGenSection):
    name = "Economic Mechanism"
    config_field = "mechanism"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Economic Mechanism and Interpretation", level=1)
        self._render_paragraphs(builder, config.mechanism)
        self._render_section_figures(builder, config, "mechanism")
        self._render_section_tables(builder, config, "mechanism")
