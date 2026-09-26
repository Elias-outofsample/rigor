"""Type 3 — Theoretical Framework & Hypotheses section."""

from .base_gen_section import BaseGenSection


class TheoreticalFrameworkSection(BaseGenSection):
    name = "Theoretical Framework"
    config_field = "theoretical_framework"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Theoretical Framework and Hypotheses", level=1)
        self._render_paragraphs(builder, config.theoretical_framework)

        # Formal hypotheses (also rendered here for Type 3)
        if config.hypotheses:
            builder.add_spacer(8)
            for h in config.hypotheses:
                builder.add_paragraph(
                    f"<b>{h.label}</b> (null): {h.null_text}", "body"
                )
                builder.add_paragraph(
                    f"<b>{h.label}</b> (alternative): {h.alt_text}", "body"
                )
                builder.add_spacer(4)
