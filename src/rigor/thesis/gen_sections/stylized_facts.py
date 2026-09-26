"""Type 2 — Stylized Facts & Empirical Patterns section."""

from .base_gen_section import BaseGenSection


class StylizedFactsSection(BaseGenSection):
    name = "Stylized Facts"
    config_field = "stylized_facts"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Stylized Facts and Empirical Patterns", level=1)
        self._render_paragraphs(builder, config.stylized_facts)
        self._render_section_figures(builder, config, "stylized_facts")
        self._render_section_tables(builder, config, "stylized_facts")
