"""Generalist Conclusion section."""

from .base_gen_section import BaseGenSection


class ConclusionGenSection(BaseGenSection):
    name = "Conclusion"
    config_field = "conclusion"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Conclusion", level=1)

        if config.has_section("conclusion"):
            self._render_paragraphs(builder, config.conclusion)
        else:
            builder.add_paragraph(
                "<i>[Conclusion to be completed.]</i>", "body_italic"
            )
