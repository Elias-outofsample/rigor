"""Type 3 — Related Literature section (factor/anomaly papers)."""

from .base_gen_section import BaseGenSection


class RelatedLiteratureSection(BaseGenSection):
    name = "Related Literature"
    config_field = "related_literature"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Related Literature", level=1)
        self._render_paragraphs(builder, config.related_literature)
