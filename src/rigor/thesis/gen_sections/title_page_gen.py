"""Generalist Title Page — title, subtitle, author, date, abstract, keywords."""

from .base_gen_section import BaseGenSection


class TitlePageGenSection(BaseGenSection):
    name = "Title Page"

    def has_content(self, config) -> bool:
        return True  # Always present

    def render(self, builder, config):
        # Compacted spacers: top 72->36, post-title 24->12, post-date 30->14,
        # pre-disclaimer 16->6. Saves ~92 points = 1.3 inches, keeps Keywords +
        # JEL + Disclaimer on the title page when the abstract is long.
        builder.add_spacer(36)
        builder.add_paragraph(config.title, "title")

        if config.subtitle:
            builder.add_paragraph(config.subtitle, "subtitle")

        builder.add_spacer(12)

        if config.author:
            builder.add_paragraph(config.author, "author")
        if config.institution:
            builder.add_paragraph(config.institution, "date_style")

        builder.add_spacer(6)
        builder.add_paragraph(config.date, "date_style")

        builder.add_spacer(14)

        # Abstract on title page
        if config.abstract:
            builder.add_paragraph("Abstract", "abstract_heading")
            self._render_paragraphs(builder, config.abstract, "abstract")

        # Keywords + JEL
        if config.keywords:
            kw_text = f"<b>Keywords:</b> {', '.join(config.keywords)}"
            builder.add_paragraph(kw_text, "abstract")
        if config.jel_codes:
            jel_text = f"<b>JEL Classification:</b> {', '.join(config.jel_codes)}"
            builder.add_paragraph(jel_text, "abstract")

        builder.add_spacer(6)
        builder.add_paragraph(
            "<i>Disclaimer: This document is for informational and educational purposes only "
            "and does not constitute investment advice. Past performance, including backtested "
            "results, is not indicative of future results.</i>",
            "disclaimer",
        )
