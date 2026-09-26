"""Type 4 — Results & Analysis section (short empirical study)."""

from .base_gen_section import BaseGenSection


class ResultsAnalysisSection(BaseGenSection):
    name = "Results & Analysis"
    config_field = "results_analysis"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Results and Analysis", level=1)
        self._render_paragraphs(builder, config.results_analysis)
        self._render_section_figures(builder, config, "results_analysis")
        self._render_section_tables(builder, config, "results_analysis")
