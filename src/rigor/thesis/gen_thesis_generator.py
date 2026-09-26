"""
Generalist Thesis Generator
============================
Main orchestrator for generalist academic paper PDF generation.

Unlike the strategy-specific ThesisGenerator (which loads backtest CSVs),
this generator works from narrative text provided in a GenThesisConfig.

The intended workflow:
  1. User fills a GenThesisBrief with raw findings, data description, etc.
  2. Claude Code drafts each section (using DraftingPrompts as guide)
  3. Drafted text is stored in a GenThesisConfig
  4. GenThesisGenerator renders the config into a publication-quality PDF

Usage:
    from thesis_gen.gen_thesis_generator import GenThesisGenerator
    from thesis_gen.gen_thesis_config import GenThesisConfig

    config = GenThesisConfig(
        title="My Research Paper",
        paper_type="market_phenomenon",
        abstract="We document...",
        introduction="...",
        ...
    )
    generator = GenThesisGenerator(config)
    pdf_path = generator.generate(output_path="output/paper.pdf")
"""

import os

from .gen_thesis_config import GenThesisBrief, GenThesisConfig
from .paper_types import get_paper_type
from .pdf.pdf_builder import ReportLabPDFBuilder


class GenThesisGenerator:
    """Main orchestrator for generalist thesis PDF generation.

    Accepts a GenThesisConfig (fully drafted narrative) and produces
    a publication-ready PDF using the same styles as the strategy thesis.
    """

    def __init__(self, config: GenThesisConfig):
        self.config = config
        self.paper_type = get_paper_type(config.paper_type)

    def generate(self, output_path: str | None = None) -> str:
        """Generate the complete thesis PDF.

        Args:
            output_path: Path for the output PDF file. If None (default), the
                         thesis is saved under thesis_gen/output/<slug>/<slug>.pdf,
                         where <slug> is derived from the paper title — i.e. a
                         dedicated, self-describing folder per paper.

        Returns:
            Path to the generated PDF file.
        """
        if output_path is None:
            from .paths import OUTPUT_DIR, slugify
            slug = slugify(self.config.title)
            output_path = str(OUTPUT_DIR / slug / f"{slug}.pdf")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Auto-number figures and tables if not set
        self._auto_number_media()

        # Get ordered sections from paper type
        sections = self.paper_type.get_sections(self.config)

        print(f"Generating: {self.config.title}")
        print(f"Paper type: {self.config.paper_type} ({self.paper_type.description})")
        print(f"Sections: {len(sections)}")

        # Create PDF builder
        builder = ReportLabPDFBuilder(
            output_path=output_path,
            title=self.config.title,
            author=self.config.author,
        )

        # Render each section
        for section in sections:
            print(f"  Rendering: {section.name}")
            try:
                section.render(builder, self.config)
            except Exception as e:
                print(f"  WARNING: Error rendering {section.name}: {e}")
                builder.add_paragraph(
                    f'[Section "{section.name}" could not be rendered: {e}]',
                    "body_italic",
                )

        # Build PDF
        print("Building PDF...")
        builder.build()
        print(f"PDF generated: {output_path}")

        return output_path

    def _auto_number_media(self):
        """Assign sequential numbers to figures and tables with number=0."""
        fig_num = 0
        for fig in self.config.figures:
            if fig.number == 0:
                fig_num += 1
                fig.number = fig_num
            else:
                fig_num = max(fig_num, fig.number)

        tbl_num = 0
        for tbl in self.config.tables:
            if tbl.number == 0:
                tbl_num += 1
                tbl.number = tbl_num
            else:
                tbl_num = max(tbl_num, tbl.number)

    @classmethod
    def from_brief(cls, brief: GenThesisBrief) -> "GenThesisGenerator":
        """Create a generator from a brief, converting to config.

        This is a convenience method that copies brief fields into a
        GenThesisConfig. The narrative sections must still be populated
        (either manually or by Claude Code) before calling generate().
        """
        config = GenThesisConfig(
            title=brief.title,
            subtitle=brief.subtitle,
            author=brief.author,
            institution=brief.institution,
            date=brief.date,
            paper_type=brief.paper_type,
            hypotheses=brief.hypotheses,
            figures=brief.figures,
            tables=brief.tables,
            references=brief.references_hint,
            tone=brief.tone,
            language=brief.language,
        )
        return cls(config)

    def get_drafting_order(self) -> list[str]:
        """Return the section names in the order Claude should draft them."""
        return self.paper_type.drafting_order

    def get_empty_sections(self) -> list[str]:
        """Return section names that are in the drafting order but not yet populated."""
        return [
            name for name in self.paper_type.drafting_order
            if not self.config.has_section(name)
        ]
