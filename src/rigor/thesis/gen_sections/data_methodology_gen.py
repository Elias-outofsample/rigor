"""Generalist Data & Methodology section."""

from .base_gen_section import BaseGenSection


class DataMethodologyGenSection(BaseGenSection):
    name = "Data & Methodology"
    config_field = "data_methodology"

    def has_content(self, config) -> bool:
        return True  # Always present — even if empty, render placeholder

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Data and Methodology", level=1)

        if config.has_section("data_methodology"):
            self._render_paragraphs(builder, config.data_methodology)
        else:
            builder.add_paragraph(
                "<i>[Data and methodology section to be completed.]</i>",
                "body_italic",
            )

        self._render_section_figures(builder, config, "data_methodology")
        self._render_section_tables(builder, config, "data_methodology")

