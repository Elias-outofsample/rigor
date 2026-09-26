"""Table of Contents — extracts sub-headings from section text for a complete TOC."""

from .base_gen_section import BaseGenSection

# Map section class config_field names to their GenThesisConfig attribute
_SECTION_TO_CONFIG_FIELD = {
    "Introduction": "introduction",
    "Institutional Background": "institutional_background",
    "Data & Methodology": "data_methodology",
    "Stylized Facts": "stylized_facts",
    "Economic Mechanism": "mechanism",
    "Robustness": "robustness",
    "Implications": "implications",
    "Conclusion": "conclusion",
    "Related Literature": "related_literature",
    "Theoretical Framework": "theoretical_framework",
    "Risk-Adjusted Results": "risk_adjusted_results",
    "Costs & Capacity": "cost_capacity",
    "Results & Analysis": "results_analysis",
    "Discussion": "discussion",
}


def _extract_subheadings(text: str) -> list[tuple[int, str]]:
    """Extract ## and ### headings from section text.

    Returns list of (level, title) tuples.
    Level 2 = ##, Level 3 = ###.
    """
    if not text:
        return []
    subheadings = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("### "):
            subheadings.append((3, line[4:].strip()))
        elif line.startswith("## "):
            subheadings.append((2, line[3:].strip()))
    return subheadings


class TOCGenSection(BaseGenSection):
    """Render a complete Table of Contents with sub-sections.

    Parses ## and ### markers from each section's narrative text
    to build a hierarchical TOC.
    """

    name = "Table of Contents"

    def has_content(self, config) -> bool:
        return True

    def render(self, builder, config):
        builder.add_page_break()
        builder.add_heading("Table of Contents", level=1)
        builder.add_spacer(12)

        # Get the section list from the paper type
        from ..paper_types import get_paper_type
        paper_type = get_paper_type(config.paper_type)
        sections = paper_type.get_sections(config)

        section_num = 0
        for section in sections:
            name = section.name

            # Skip non-content sections
            if name in ("Title Page", "Abstract", "Table of Contents"):
                continue

            # Unnumbered sections
            if name in ("References", "Appendix"):
                builder.add_paragraph(f"<b>{name}</b>", "toc_h1")
                continue

            # Numbered sections
            section_num += 1
            builder.add_paragraph(f"<b>{section_num}. {name}</b>", "toc_h1")

            # Extract and render sub-headings from the section text
            config_field = _SECTION_TO_CONFIG_FIELD.get(name, "")
            if config_field:
                section_text = getattr(config, config_field, "")
                subheadings = _extract_subheadings(section_text)
                for level, title in subheadings:
                    if level == 2:
                        builder.add_paragraph(title, "toc_h2")
                    elif level == 3:
                        builder.add_paragraph(f"    {title}", "toc_h2")
