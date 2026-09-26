"""Generalist References section."""

from .base_gen_section import BaseGenSection


class ReferencesGenSection(BaseGenSection):
    name = "References"

    def has_content(self, config) -> bool:
        return bool(config.references)

    def render(self, builder, config):
        builder.add_page_break()
        builder.add_heading("References", level=1)

        for ref_line in config.ordered_references_apa():
            builder.add_paragraph(ref_line, "body")
            builder.add_spacer(4)
