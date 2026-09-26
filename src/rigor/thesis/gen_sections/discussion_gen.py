"""Generalist Discussion section."""

from .base_gen_section import BaseGenSection


class DiscussionGenSection(BaseGenSection):
    name = "Discussion"
    config_field = "discussion"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(f"{num}. Discussion", level=1)
        self._render_paragraphs(builder, config.discussion)
