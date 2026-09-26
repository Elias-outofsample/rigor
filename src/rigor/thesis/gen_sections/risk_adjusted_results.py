"""Type 3 — Main Results: Risk-Adjusted Returns section."""

from .base_gen_section import BaseGenSection


class RiskAdjustedResultsSection(BaseGenSection):
    name = "Risk-Adjusted Results"
    config_field = "risk_adjusted_results"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Main Results \u2014 Risk-Adjusted Returns", level=1)
        self._render_paragraphs(builder, config.risk_adjusted_results)
        self._render_section_figures(builder, config, "risk_adjusted_results")
        self._render_section_tables(builder, config, "risk_adjusted_results")
