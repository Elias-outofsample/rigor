"""Generalist Robustness Checks section."""

from .base_gen_section import BaseGenSection


class RobustnessGenSection(BaseGenSection):
    name = "Robustness"
    config_field = "robustness"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Robustness Checks", level=1)
        self._render_paragraphs(builder, config.robustness)
        self._render_section_figures(builder, config, "robustness")
        self._render_section_tables(builder, config, "robustness")
