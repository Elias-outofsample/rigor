"""Generalist Abstract section (standalone page, separate from title page)."""

from .base_gen_section import BaseGenSection


class AbstractGenSection(BaseGenSection):
    name = "Abstract"
    config_field = "abstract"

    def has_content(self, config) -> bool:
        # Abstract is rendered on title page; this section is a no-op
        # but kept in the chain so paper_types can reference it
        return False

    def render(self, builder, config):
        # Abstract is already rendered on the title page by TitlePageGenSection
        pass
