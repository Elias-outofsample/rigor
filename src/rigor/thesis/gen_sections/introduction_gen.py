"""Generalist Introduction section."""

from .base_gen_section import BaseGenSection


class IntroductionGenSection(BaseGenSection):
    name = "Introduction"
    config_field = "introduction"

    def render(self, builder, config):
        builder.add_page_break()
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Introduction", level=1)
        self._render_paragraphs(builder, config.introduction)

        # Formal hypotheses (if any)
        if config.hypotheses:
            builder.add_spacer(8)
            builder.add_heading("1.1 Research Hypotheses", level=2)
            for h in config.hypotheses:
                builder.add_paragraph(
                    f"<b>{h.label}</b> (null): {h.null_text}", "body"
                )
                builder.add_paragraph(
                    f"<b>{h.label}</b> (alternative): {h.alt_text}", "body"
                )
                builder.add_spacer(4)

        self._render_section_figures(builder, config, "introduction")
        self._render_section_tables(builder, config, "introduction")
